import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np

from host.shared.config import DEFAULT_SERIAL_PORT, DEFAULT_UDP_PORT
from host.ingestion.reader import SensorCoordinator
from host.processing.ahrs_processor import AHRSProcessor
from host.viz.visualizer import Visualizer

class IMUApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU 3D Visualizer (IMUFusion AHRS)")
        self.root.geometry("1100x780")

        self.data_lock = threading.Lock()
        self.latest_data = {
            "accel": np.zeros(3), "gyro": np.zeros(3), "mag": np.zeros(3),
            "euler": np.zeros(3), "valid": False, "connected": False,
            "collecting": False, "packets_received": 0
        }

        self.reader = SensorCoordinator(self._on_data_received, self._on_error)
        self.ahrs = AHRSProcessor()

        self.rate_calc_time = time.time()
        self.rate_calc_packets = 0
        self.current_rate = 0.0

        self._build_ui()
        self.visualizer = Visualizer(self.plot_frame)

        self._start_gui_loops()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_ui(self):
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(self.control_frame, text="Режим:").grid(row=0, column=0, padx=5)
        self.mode_var = tk.StringVar(value="UDP")
        self.mode_cb = ttk.Combobox(self.control_frame, textvariable=self.mode_var,
                                    values=["Serial", "UDP"], width=8, state="readonly")
        self.mode_cb.grid(row=0, column=1, padx=5)
        self.mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(self.control_frame, text="COM:").grid(row=0, column=2, padx=5)
        self.port_entry = ttk.Entry(self.control_frame, width=8)
        self.port_entry.insert(0, DEFAULT_SERIAL_PORT)
        self.port_entry.grid(row=0, column=3, padx=5)

        ttk.Label(self.control_frame, text="UDP:").grid(row=0, column=4, padx=5)
        self.udp_entry = ttk.Entry(self.control_frame, width=8)
        self.udp_entry.insert(0, str(DEFAULT_UDP_PORT))
        self.udp_entry.grid(row=0, column=5, padx=5)

        self.connect_btn = ttk.Button(self.control_frame, text="🔌 Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=6, padx=5)

        ttk.Label(self.control_frame, text="Точек:").grid(row=0, column=7, padx=5)
        self.points_entry = ttk.Entry(self.control_frame, width=8)
        self.points_entry.insert(0, "100")
        self.points_entry.grid(row=0, column=8, padx=5)

        self.start_btn = ttk.Button(self.control_frame, text="▶ Start", command=self.start_collection, state=tk.DISABLED)
        self.start_btn.grid(row=0, column=9, padx=5)

        self.stop_btn = ttk.Button(self.control_frame, text="⏹ Stop", command=self.stop_collection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=10, padx=5)

        self._on_mode_change(None)

        self.data_frame = ttk.Frame(self.root, padding="10")
        self.data_frame.pack(side=tk.TOP, fill=tk.X)

        self.roll_label = ttk.Label(self.data_frame, text="Roll: ---°", font=("Arial", 12, "bold"))
        self.pitch_label = ttk.Label(self.data_frame, text="Pitch: ---°", font=("Arial", 12, "bold"))
        self.yaw_label = ttk.Label(self.data_frame, text="Yaw: ---°", font=("Arial", 12, "bold"))
        self.count_label = ttk.Label(self.data_frame, text="Packets: 0", font=("Arial", 12))
        self.rate_label = ttk.Label(self.data_frame, text="Rate: 0.0 Hz", font=("Arial", 12, "bold"), foreground="darkgreen")
        self.status_label = ttk.Label(self.data_frame, text="Status: ○ Disconnected", font=("Arial", 12), foreground="red")

        for lbl in [self.roll_label, self.pitch_label, self.yaw_label, self.count_label, self.rate_label, self.status_label]:
            lbl.pack(side=tk.LEFT, padx=10)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def _on_mode_change(self, event):
        mode = self.mode_var.get()
        self.port_entry.config(state=tk.NORMAL if mode == "Serial" else tk.DISABLED)
        self.udp_entry.config(state=tk.NORMAL if mode == "UDP" else tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL if (mode == "Serial" and self.latest_data["connected"]) else tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

    def _start_gui_loops(self):
        self._animate()
        self._update_labels()

    def _animate(self):
        with self.data_lock:
            valid = self.latest_data["valid"]
            euler = self.latest_data["euler"].copy()
        self.visualizer.draw(valid, euler)
        self.root.after(50, self._animate)

    def _update_labels(self):
        with self.data_lock:
            valid = self.latest_data["valid"]
            connected = self.latest_data["connected"]
            collecting = self.latest_data["collecting"]
            euler = self.latest_data["euler"]
            packets = self.latest_data["packets_received"]

        now = time.time()
        dt = now - self.rate_calc_time
        if dt >= 1.0:
            self.current_rate = (packets - self.rate_calc_packets) / dt
            self.rate_calc_time = now
            self.rate_calc_packets = packets

        self.rate_label.config(text=f"Rate: {self.current_rate:4.1f} Hz")
        self.count_label.config(text=f"Packets: {packets}")

        if valid:
            roll, pitch, yaw = euler
            self.roll_label.config(text=f"Roll: {roll:6.2f}°", foreground="green")
            self.pitch_label.config(text=f"Pitch: {pitch:6.2f}°", foreground="green")
            self.yaw_label.config(text=f"Yaw: {yaw:6.2f}°", foreground="green")
            status = "Status: ● Collecting (AHRS)" if collecting else f"Status: ● {self.mode_var.get()} Connected"
            self.status_label.config(text=status, foreground="green" if collecting else "blue")
        else:
            self.roll_label.config(text="Roll: ---°", foreground="gray")
            self.pitch_label.config(text="Pitch: ---°", foreground="gray")
            self.yaw_label.config(text="Yaw: ---°", foreground="gray")
            if connected:
                self.status_label.config(text="Status: ⏳ Waiting for data...", foreground="orange")
            else:
                self.status_label.config(text="Status: ○ Disconnected", foreground="red")
                self.rate_label.config(text="Rate: 0.0 Hz", foreground="gray")

        self.root.after(100, self._update_labels)

    def toggle_connection(self):
        if self.reader.backend and self.reader.thread and self.reader.thread.is_alive():
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        mode = self.mode_var.get()
        try:
            if mode == "Serial":
                self.reader.connect_serial(self.port_entry.get().strip())
                self.start_btn.config(state=tk.NORMAL)
            elif mode == "UDP":
                self.reader.connect_udp(int(self.udp_entry.get().strip()))
            self.rate_calc_time = time.time()
            self.rate_calc_packets = 0
            self.current_rate = 0.0
            with self.data_lock:
                self.latest_data["connected"] = True
                self.latest_data["valid"] = False
                self.latest_data["packets_received"] = 0
            self.connect_btn.config(text="🔴 Disconnect")
            print(f"Connected via {mode}")
        except Exception as e:
            print(f"Connection error: {e}")
            self.disconnect()

    def disconnect(self):
        self.reader.disconnect()
        with self.data_lock:
            self.latest_data.update({"connected": False, "valid": False, "collecting": False})
        self.connect_btn.config(text="🔌 Connect")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        print("Disconnected")

    def start_collection(self):
        if not self.reader.backend or self.mode_var.get() != "Serial":
            return
        try:
            num_points = int(self.points_entry.get().strip())
            if not (1 <= num_points <= 10000):
                print("Points must be 1-10000")
                return
            self.reader.send_command(f"S{num_points}")
            with self.data_lock:
                self.latest_data["collecting"] = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        except Exception as e:
            print(f"Send error: {e}")

    def stop_collection(self):
        if not self.reader.backend or self.mode_var.get() != "Serial":
            return
        try:
            self.reader.send_command("P")
            with self.data_lock:
                self.latest_data["collecting"] = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Stop error: {e}")

    def _on_data_received(self, accel, gyro, mag):
        try:
            euler = self.ahrs.update(gyro, accel, mag)
            with self.data_lock:
                self.latest_data.update({
                    "accel": accel, "gyro": gyro, "mag": mag,
                    "euler": euler, "valid": True,
                    "packets_received": self.latest_data["packets_received"] + 1
                })
        except Exception as e:
            print(f"AHRS error: {e}")

    def _on_error(self, msg: str):
        print(msg)

    def on_closing(self):
        self.disconnect()
        self.root.destroy()