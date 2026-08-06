#!/usr/bin/env python3
"""GUI client for the ESP32-C3 MPU6050 UDP quaternion server."""

from __future__ import annotations

import math
import queue
import socket
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk
from typing import Final

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import colors as mpl_colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_DEVICE_IP: Final = "192.168.1.100"
DEFAULT_DEVICE_PORT: Final = 4210
CUBE_COUNT: Final = 5
IDENTITY_QUATERNION: Final = (1.0, 0.0, 0.0, 0.0)

CUBE_VERTICES: Final = (
    (-0.7, -0.7, -0.7),
    (0.7, -0.7, -0.7),
    (0.7, 0.7, -0.7),
    (-0.7, 0.7, -0.7),
    (-0.7, -0.7, 0.7),
    (0.7, -0.7, 0.7),
    (0.7, 0.7, 0.7),
    (-0.7, 0.7, 0.7),
)
CUBE_FACES: Final = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 3, 7, 4),
)
CUBE_CENTERS: Final = ((-4.8, 0.0, 0.0), (-2.4, 0.0, 0.0), (0.0, 0.0, 0.0),
                       (2.4, 0.0, 0.0), (4.8, 0.0, 0.0))
CUBE_COLORS: Final = ("#4ea1ff", "#ff826e", "#63d297", "#f6c85f", "#a68cf2")


def normalize_quaternion(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in values))
    if norm < 1e-9 or not math.isfinite(norm):
        raise ValueError("нулевая или некорректная норма кватерниона")
    return tuple(component / norm for component in values)  # type: ignore[return-value]


def quaternion_matrix(
    quaternion: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = quaternion
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def rotate_vector(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        sum(matrix[0][i] * vector[i] for i in range(3)),
        sum(matrix[1][i] * vector[i] for i in range(3)),
        sum(matrix[2][i] * vector[i] for i in range(3)),
    )


def make_face_palette(base_color: str) -> list[tuple[float, float, float, float]]:
    red, green, blue = mpl_colors.to_rgb(base_color)
    factors = (0.52, 1.18, 0.78, 0.92, 1.04, 0.66)
    return [
        (min(red * factor, 1.0), min(green * factor, 1.0), min(blue * factor, 1.0), 0.88)
        for factor in factors
    ]


class QuaternionViewer:
    """Tk application that receives UDP frames and renders five oriented cubes."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MPU6050 UDP — визуализация кватернионов")
        self.root.geometry("1500x860")
        self.root.minsize(1120, 680)

        self.sock: socket.socket | None = None
        self.receiver_thread: threading.Thread | None = None
        self.stop_receiver = threading.Event()
        self.events: queue.Queue[tuple[str, float, object, object]] = queue.Queue()
        self.quaternions: dict[int, tuple[float, float, float, float]] = {}
        self.needs_redraw = True
        self.rx_packet_count = 0
        self.rx_quaternion_count = 0

        self.device_ip = tk.StringVar(value=DEFAULT_DEVICE_IP)
        self.device_port = tk.StringVar(value=str(DEFAULT_DEVICE_PORT))
        self.connection_status = tk.StringVar(value="Не подключено")
        self.packet_status = tk.StringVar(value="UDP-пакетов: 0 · кватернионов: 0")
        self.rate_hz = tk.StringVar(value="10")
        self.manual_command = tk.StringVar()
        self.sensor_mapping = [tk.IntVar(value=index + 1) for index in range(CUBE_COUNT)]

        self.cube_collections: list[Poly3DCollection] = []
        self.cube_axes: list[list[object]] = []
        self.cube_labels: list[object] = []

        self._configure_style()
        self._build_controls()
        self._build_main_area()
        self._configure_plot()

        for variable in self.sensor_mapping:
            variable.trace_add("write", self._mapping_changed)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(20, self._process_events)
        self.root.after(33, self._refresh_plot)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Header.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("Status.TLabel", padding=(8, 4))

    def _build_controls(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="IP устройства:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.device_ip, width=16).grid(
            row=0, column=1, padx=(5, 12), sticky=tk.W
        )
        ttk.Label(top, text="UDP-порт:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(top, textvariable=self.device_port, width=7).grid(
            row=0, column=3, padx=(5, 12), sticky=tk.W
        )

        self.connect_button = ttk.Button(top, text="Подключиться", command=self.connect_device)
        self.connect_button.grid(row=0, column=4, padx=(0, 5))
        self.disconnect_button = ttk.Button(
            top, text="Отключиться", command=self.disconnect_device, state=tk.DISABLED
        )
        self.disconnect_button.grid(row=0, column=5, padx=(0, 14))

        ttk.Separator(top, orient=tk.VERTICAL).grid(
            row=0, column=6, rowspan=2, sticky=tk.NS, padx=4
        )
        ttk.Button(top, text="START", command=lambda: self.send_command("START")).grid(
            row=0, column=7, padx=3
        )
        ttk.Button(top, text="STOP", command=lambda: self.send_command("STOP")).grid(
            row=0, column=8, padx=3
        )
        ttk.Button(top, text="STATUS", command=lambda: self.send_command("STATUS")).grid(
            row=0, column=9, padx=3
        )
        ttk.Button(top, text="Калибровка", command=self.calibrate).grid(
            row=0, column=10, padx=(3, 12)
        )

        ttk.Label(top, text="Частота, Гц:").grid(row=0, column=11, sticky=tk.E)
        ttk.Spinbox(top, from_=1, to=100, textvariable=self.rate_hz, width=5).grid(
            row=0, column=12, padx=5
        )
        ttk.Button(top, text="Применить", command=self.apply_rate).grid(row=0, column=13)

        status_frame = ttk.Frame(top)
        status_frame.grid(row=1, column=0, columnspan=14, sticky=tk.EW, pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.connection_status, style="Status.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(status_frame, textvariable=self.packet_status, style="Status.TLabel").pack(
            side=tk.RIGHT
        )
        top.columnconfigure(13, weight=1)

    def _build_main_area(self) -> None:
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        visual_frame = ttk.Frame(paned)
        monitor_frame = ttk.Frame(paned, width=450)
        paned.add(visual_frame, weight=3)
        paned.add(monitor_frame, weight=2)

        mapping_frame = ttk.LabelFrame(visual_frame, text="Назначение датчиков", padding=(8, 6))
        mapping_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        for index, variable in enumerate(self.sensor_mapping):
            item = ttk.Frame(mapping_frame)
            item.pack(side=tk.LEFT, padx=(0, 18))
            ttk.Label(item, text=f"Куб {index + 1} ← датчик").pack(side=tk.LEFT)
            ttk.Spinbox(item, from_=1, to=255, textvariable=variable, width=4).pack(
                side=tk.LEFT, padx=(5, 0)
            )

        figure_frame = ttk.Frame(visual_frame)
        figure_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(9.2, 6.6), dpi=100)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, figure_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        monitor_header = ttk.Frame(monitor_frame)
        monitor_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(monitor_header, text="Монитор UDP-пакетов", style="Header.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Button(monitor_header, text="Очистить", command=self.clear_monitor).pack(side=tk.RIGHT)

        self.monitor = scrolledtext.ScrolledText(
            monitor_frame,
            wrap=tk.NONE,
            width=52,
            height=30,
            font=("TkFixedFont", 9),
            state=tk.DISABLED,
        )
        self.monitor.pack(fill=tk.BOTH, expand=True)

        command_frame = ttk.LabelFrame(monitor_frame, text="Произвольная команда", padding=6)
        command_frame.pack(fill=tk.X, pady=(6, 0))
        command_entry = ttk.Entry(command_frame, textvariable=self.manual_command)
        command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        command_entry.bind("<Return>", lambda _event: self.send_manual_command())
        ttk.Button(command_frame, text="Отправить", command=self.send_manual_command).pack(
            side=tk.LEFT, padx=(6, 0)
        )

    def _configure_plot(self) -> None:
        self.figure.patch.set_facecolor("#f4f6f8")
        self.axes.set_facecolor("#f4f6f8")
        self.axes.set_title("Ориентация датчиков MPU6050", pad=16)
        self.axes.set_xlim(-6.2, 6.2)
        self.axes.set_ylim(-2.2, 2.2)
        self.axes.set_zlim(-2.2, 2.2)
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_zlabel("Z")
        self.axes.set_box_aspect((12.4, 4.4, 4.4))
        self.axes.view_init(elev=22, azim=-62)
        self.axes.grid(True, alpha=0.3)

        for index, (center, color) in enumerate(zip(CUBE_CENTERS, CUBE_COLORS, strict=True)):
            vertices = [
                (x + center[0], y + center[1], z + center[2]) for x, y, z in CUBE_VERTICES
            ]
            faces = [[vertices[vertex_index] for vertex_index in face] for face in CUBE_FACES]
            collection = Poly3DCollection(
                faces,
                facecolors=make_face_palette(color),
                edgecolors="#18212b",
                linewidths=1.0,
            )
            self.axes.add_collection3d(collection)
            self.cube_collections.append(collection)

            axis_lines = [
                self.axes.plot([], [], [], color="#d62728", linewidth=2.4)[0],
                self.axes.plot([], [], [], color="#2ca02c", linewidth=2.4)[0],
                self.axes.plot([], [], [], color="#1f77b4", linewidth=2.4)[0],
            ]
            self.cube_axes.append(axis_lines)
            self.cube_labels.append(
                self.axes.text(center[0], center[1], -1.35, f"Куб {index + 1}\nдатчик {index + 1}",
                               ha="center", va="top", fontsize=9)
            )

        self.canvas.draw()

    def connect_device(self) -> None:
        host = self.device_ip.get().strip()
        try:
            port = int(self.device_port.get().strip())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror("Некорректный порт", "UDP-порт должен быть числом от 1 до 65535.")
            return
        if not host:
            messagebox.showerror("Некорректный адрес", "Введите IP-адрес ESP32.")
            return

        self.disconnect_device(log=False)
        try:
            endpoint = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", 0))
            sock.connect(endpoint)
            sock.settimeout(0.25)
        except OSError as error:
            messagebox.showerror("Ошибка UDP", str(error))
            return

        self.sock = sock
        self.stop_receiver.clear()
        self.receiver_thread = threading.Thread(
            target=self._receive_loop, args=(sock,), name="udp-receiver", daemon=True
        )
        self.receiver_thread.start()

        local_ip, local_port = sock.getsockname()
        self.connection_status.set(
            f"UDP {endpoint[0]}:{endpoint[1]} · локальный порт {local_port} · ожидание ответа"
        )
        self.connect_button.configure(state=tk.DISABLED)
        self.disconnect_button.configure(state=tk.NORMAL)
        self._append_monitor(
            f"[{self._clock()}] OPEN {local_ip}:{local_port} → {endpoint[0]}:{endpoint[1]}\n"
        )
        self.send_command("HELLO")
        self.root.after(150, lambda: self.send_command("STATUS") if self.sock is sock else None)

    def disconnect_device(self, log: bool = True) -> None:
        sock = self.sock
        self.sock = None
        self.stop_receiver.set()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self.receiver_thread = None
        self.connect_button.configure(state=tk.NORMAL)
        self.disconnect_button.configure(state=tk.DISABLED)
        self.connection_status.set("Не подключено")
        if log and sock is not None:
            self._append_monitor(f"[{self._clock()}] CLOSE локальный UDP-сокет\n")

    def _receive_loop(self, sock: socket.socket) -> None:
        while not self.stop_receiver.is_set() and self.sock is sock:
            try:
                payload, address = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as error:
                if not self.stop_receiver.is_set() and self.sock is sock:
                    self.events.put(("error", time.time(), str(error), None))
                break
            self.events.put(("packet", time.time(), payload, address))

    def send_command(self, command: str) -> None:
        command = command.strip()
        if not command:
            return
        sock = self.sock
        if sock is None:
            messagebox.showwarning("Нет соединения", "Сначала укажите адрес и подключитесь к ESP32.")
            return
        try:
            payload = (command + "\n").encode("ascii")
        except UnicodeEncodeError:
            messagebox.showerror("Некорректная команда", "Команда должна содержать ASCII-символы.")
            return
        try:
            sock.send(payload)
        except OSError as error:
            self._append_monitor(f"[{self._clock()}] TX ERROR {error}\n")
            return
        self._append_monitor(f"[{self._clock()}] TX ({len(payload)} B)\n{command}\n")

    def send_manual_command(self) -> None:
        command = self.manual_command.get()
        if command.strip():
            self.send_command(command)
            self.manual_command.set("")

    def apply_rate(self) -> None:
        try:
            rate = int(self.rate_hz.get().strip())
            if not 1 <= rate <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Некорректная частота", "Допустимый диапазон: 1–100 Гц.")
            return
        self.send_command(f"SET_RATE {rate}")

    def calibrate(self) -> None:
        confirmed = messagebox.askyesno(
            "Калибровка MPU6050",
            "Положите все датчики неподвижно. Запустить калибровку?",
        )
        if confirmed:
            self.send_command("CALIB")

    def _process_events(self) -> None:
        for _ in range(250):
            try:
                event_type, timestamp, payload, address = self.events.get_nowait()
            except queue.Empty:
                break

            if event_type == "error":
                self._append_monitor(
                    f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}] "
                    f"RX ERROR {payload}\n"
                )
                continue

            raw = payload if isinstance(payload, bytes) else bytes()
            remote = address if isinstance(address, tuple) else ("?", 0)
            text = raw.decode("utf-8", errors="replace").rstrip("\x00\r\n")
            self.rx_packet_count += 1
            self.connection_status.set(f"Подключено к {remote[0]}:{remote[1]}")
            self._append_monitor(
                f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}] "
                f"RX {remote[0]}:{remote[1]} ({len(raw)} B)\n{text}\n"
            )
            self._parse_packet(text)

        self.packet_status.set(
            f"UDP-пакетов: {self.rx_packet_count} · кватернионов: {self.rx_quaternion_count}"
        )
        self.root.after(20, self._process_events)

    def _parse_packet(self, text: str) -> None:
        received_quaternion = False
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 6 or parts[0].upper() not in {"Q", "QUAT"}:
                continue
            try:
                sensor_id = int(parts[1])
                values = tuple(float(value) for value in parts[2:6])
                quaternion = normalize_quaternion(values)  # type: ignore[arg-type]
            except (ValueError, OverflowError):
                continue
            self.quaternions[sensor_id] = quaternion
            self.rx_quaternion_count += 1
            received_quaternion = True

        if received_quaternion:
            self.needs_redraw = True

    def _mapping_changed(self, *_args: object) -> None:
        self.needs_redraw = True

    def _refresh_plot(self) -> None:
        if self.needs_redraw:
            for cube_index, center in enumerate(CUBE_CENTERS):
                try:
                    sensor_id = int(self.sensor_mapping[cube_index].get())
                except (tk.TclError, ValueError):
                    sensor_id = cube_index + 1

                quaternion = self.quaternions.get(sensor_id, IDENTITY_QUATERNION)
                matrix = quaternion_matrix(quaternion)
                rotated_vertices = []
                for vertex in CUBE_VERTICES:
                    x, y, z = rotate_vector(matrix, vertex)
                    rotated_vertices.append((x + center[0], y + center[1], z + center[2]))

                faces = [
                    [rotated_vertices[vertex_index] for vertex_index in face] for face in CUBE_FACES
                ]
                self.cube_collections[cube_index].set_verts(faces)

                local_axes = ((1.05, 0.0, 0.0), (0.0, 1.05, 0.0), (0.0, 0.0, 1.05))
                for line, local_axis in zip(
                    self.cube_axes[cube_index], local_axes, strict=True
                ):
                    axis = rotate_vector(matrix, local_axis)
                    line.set_data_3d(
                        (center[0], center[0] + axis[0]),
                        (center[1], center[1] + axis[1]),
                        (center[2], center[2] + axis[2]),
                    )

                self.cube_labels[cube_index].set_text(
                    f"Куб {cube_index + 1}\nдатчик {sensor_id}"
                )

            self.canvas.draw_idle()
            self.needs_redraw = False

        self.root.after(33, self._refresh_plot)

    def _append_monitor(self, text: str) -> None:
        self.monitor.configure(state=tk.NORMAL)
        self.monitor.insert(tk.END, text)
        line_count = int(self.monitor.index("end-1c").split(".")[0])
        if line_count > 4000:
            self.monitor.delete("1.0", "501.0")
        self.monitor.see(tk.END)
        self.monitor.configure(state=tk.DISABLED)

    def clear_monitor(self) -> None:
        self.monitor.configure(state=tk.NORMAL)
        self.monitor.delete("1.0", tk.END)
        self.monitor.configure(state=tk.DISABLED)

    @staticmethod
    def _clock() -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def close(self) -> None:
        self.disconnect_device(log=False)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    QuaternionViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
