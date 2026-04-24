import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

# 🔧 Авто-добавление корня проекта в sys.path
CURRENT_FILE = Path(__file__).resolve()
for parent in [CURRENT_FILE] + list(CURRENT_FILE.parents):
    if (parent / 'host').is_dir() and (parent / 'firmware').is_dir():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ✅ Импорт агрегатора из ingestion (не переопределяем его здесь!)
from host.ingestion.reader import SensorCoordinator


class RawUDPVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Raw IMU Data (UDP)")
        self.root.geometry("1000x750")

        self.max_points = 200
        self.lock = threading.Lock()
        self._reset_buffers()
        self.reader = None

        self._build_ui()
        self._init_plots()
        
        self.root.after(50, self._update_gui)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _reset_buffers(self):
        self.buffers = {
            "accel": {"x": deque(maxlen=self.max_points), "y": deque(maxlen=self.max_points), "z": deque(maxlen=self.max_points)},
            "gyro":  {"x": deque(maxlen=self.max_points), "y": deque(maxlen=self.max_points), "z": deque(maxlen=self.max_points)},
            "mag":   {"x": deque(maxlen=self.max_points), "y": deque(maxlen=self.max_points), "z": deque(maxlen=self.max_points)}
        }

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding="10")
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="UDP Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = ttk.Entry(ctrl, width=8)
        self.port_entry.insert(0, "1399")
        self.port_entry.pack(side=tk.LEFT, padx=5)

        self.btn_connect = ttk.Button(ctrl, text="🔌 Connect", command=self._toggle_connection)
        self.btn_connect.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Max Points:").pack(side=tk.LEFT, padx=(15, 5))
        self.points_entry = ttk.Entry(ctrl, width=8)
        self.points_entry.insert(0, "200")
        self.points_entry.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(ctrl, text="Status: Disconnected", foreground="red")
        self.status_label.pack(side=tk.RIGHT, padx=5)

    def _init_plots(self):
        self.fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.subplots_adjust(hspace=0.4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.axes = axes
        titles = ["Accelerometer (m/s²)", "Gyroscope (°/s)", "Magnetometer (μT)"]
        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(["X", "Y", "Z"], loc="upper right")

    def _toggle_connection(self):
        if self.reader and self.reader.is_connected():
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        try:
            port = int(self.port_entry.get().strip())
            self.max_points = int(self.points_entry.get().strip())
            self._reset_buffers()

            self.reader = SensorCoordinator(
                data_callback=self._on_data_received,
                error_callback=self._on_error,
                status_callback=lambda msg: self.status_label.config(text=f"Status: {msg}", foreground="blue")
            )
            # ✅ Используем правильный метод
            self.reader.connect_udp(port=port)

            self.btn_connect.config(text="⏹ Disconnect")
            self.status_label.config(text="Status: Listening...", foreground="green")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", foreground="red")

    def _disconnect(self):
        if self.reader:
            self.reader.disconnect()
        self.reader = None
        self.btn_connect.config(text="🔌 Connect")
        self.status_label.config(text="Status: Disconnected", foreground="red")

    def _on_data_received(self, accel, gyro, mag):
        with self.lock:
            for i, key in enumerate(["x", "y", "z"]):
                self.buffers["accel"][key].append(accel[i])
                self.buffers["gyro"][key].append(gyro[i])
                self.buffers["mag"][key].append(mag[i])

    def _on_error(self, msg: str):
        print(f"[UDP] {msg}")
        self.root.after(0, lambda: self.status_label.config(text=f"Error: {msg}", foreground="red"))

    def _update_gui(self):
        with self.lock:
            n = len(self.buffers["accel"]["x"])
            if n > 0:
                x_vals = list(range(n))
                titles = ["Accelerometer (m/s²)", "Gyroscope (°/s)", "Magnetometer (μT)"]
                for i, sensor in enumerate(["accel", "gyro", "mag"]):
                    ax = self.axes[i]
                    ax.clear()
                    ax.set_xlim(max(0, n - self.max_points), n)
                    ax.set_title(titles[i])
                    ax.grid(True, alpha=0.3)
                    ax.legend(["X", "Y", "Z"], loc="upper right")
                    for key, color in zip(["x", "y", "z"], ["r", "g", "b"]):
                        ax.plot(x_vals, list(self.buffers[sensor][key]), color=color, linewidth=1.5)
                self.canvas.draw()
        self.root.after(50, self._update_gui)

    def _on_close(self):
        self._disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RawUDPVisualizer(root)
    root.mainloop()