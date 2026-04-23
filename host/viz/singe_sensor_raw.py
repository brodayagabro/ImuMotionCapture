import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
from collections import deque

class SingleSensorVizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Single IMU Visualizer (Raw + Processed)")
        self.root.geometry("1300x850")

        self.max_points = 200
        self.raw_buf = {
            "accel": {"x": deque(maxlen=self.max_points), "y": deque(maxlen=self.max_points), "z": deque(maxlen=self.max_points)},
            "gyro":  {"x": deque(maxlen=self.max_points), "y": deque(maxlen=self.max_points), "z": deque(maxlen=self.max_points)},
            "mag":   {"x": deque(maxlen=self.max_points), "y": deque(maxlen=self.max_points), "z": deque(maxlen=self.max_points)},
        }
        self.euler_buf = {"roll": deque(maxlen=self.max_points), "pitch": deque(maxlen=self.max_points), "yaw": deque(maxlen=self.max_points)}
        
        self._running = False
        self._sim_thread = None

        self._build_ui()
        self._init_plots()

        self.root.after(50, self._gui_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding="10")
        ctrl.pack(fill=tk.X)
        self.btn_toggle = ttk.Button(ctrl, text="▶ Start Simulation", command=self._toggle_sim)
        self.btn_toggle.pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(self.root)
        self.tab_raw = ttk.Frame(self.notebook)
        self.tab_proc = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_raw, text="Raw Data (Time Series)")
        self.notebook.add(self.tab_proc, text="Processed (3D + Euler)")
        self.notebook.pack(fill=tk.BOTH, expand=True)

    def _init_plots(self):
        self.fig_raw, self.axs_raw = plt.subplots(3, 1, figsize=(11, 7))
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.tab_raw)
        self.canvas_raw.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        titles = ["Accelerometer (m/s²)", "Gyroscope (°/s)", "Magnetometer (μT)"]
        for ax, title in zip(self.axs_raw, titles):
            ax.set_title(title)
            ax.set_xlim(0, self.max_points)
            ax.grid(True, alpha=0.4)
            ax.legend(["X", "Y", "Z"], loc="upper right")

        self.fig_proc = plt.Figure(figsize=(11, 7))
        self.ax_3d = self.fig_proc.add_subplot(121, projection="3d")
        self.ax_euler = self.fig_proc.add_subplot(122)
        self.canvas_proc = FigureCanvasTkAgg(self.fig_proc, master=self.tab_proc)
        self.canvas_proc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cube = np.array([
            [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
            [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
        ])
        self.edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        self.ax_3d.set_xlim([-2,2]); self.ax_3d.set_ylim([-2,2]); self.ax_3d.set_zlim([-2,2])
        self.ax_3d.set_title("3D Orientation")
        self.ax_euler.set_xlim(0, self.max_points)
        self.ax_euler.set_title("Euler Angles (°)")
        self.ax_euler.grid(True, alpha=0.4)
        self.ax_euler.legend(["Roll", "Pitch", "Yaw"], loc="upper right")

    def _gui_loop(self):
        self._draw_raw()
        self._draw_processed()
        self.canvas_raw.draw()
        self.canvas_proc.draw()
        self.root.after(50, self._gui_loop)

    def _draw_raw(self):
        for i, ax in enumerate(self.axs_raw):
            ax.clear()
            ax.set_title(["Accelerometer", "Gyroscope", "Magnetometer"][i])
            ax.set_xlim(0, self.max_points)
            ax.grid(True, alpha=0.4)
            for label, color in zip(["X","Y","Z"], ["r","g","b"]):
                ax.plot(self.raw_buf[["accel","gyro","mag"][i]][label.lower()], color=color, linewidth=1.2)

    def _draw_processed(self):
        self.ax_3d.clear()
        rotated = self.cube.copy()
        for edge in self.edges:
            pts = rotated[edge]
            self.ax_3d.plot(pts[:,0], pts[:,1], pts[:,2], color="b", linewidth=2, alpha=0.8)
        self.ax_3d.set_title("3D Orientation (Stub)")

        self.ax_euler.clear()
        self.ax_euler.plot(self.euler_buf["roll"], color="r", linewidth=1.2)
        self.ax_euler.plot(self.euler_buf["pitch"], color="g", linewidth=1.2)
        self.ax_euler.plot(self.euler_buf["yaw"], color="b", linewidth=1.2)
        self.ax_euler.set_xlim(0, self.max_points)
        self.ax_euler.grid(True, alpha=0.4)

    def _toggle_sim(self):
        if self._running:
            self._running = False
            self.btn_toggle.config(text="▶ Start Simulation")
        else:
            self._running = True
            self.btn_toggle.config(text="⏹ Stop Simulation")
            threading.Thread(target=self._simulate, daemon=True).start()

    def _simulate(self):
        t = 0.0
        while self._running:
            acc = np.array([np.sin(t)*0.5, np.cos(t)*0.5, 1.0]) * 9.81
            gyr = np.array([np.sin(t*2.0), np.cos(t*1.5), 0.3]) * 15.0
            mag = np.array([np.cos(t), np.sin(t), 0.2])
            euler = np.degrees([np.sin(t)*0.5, np.cos(t)*0.3, (t*0.1) % 360])
            self._push_data(acc, gyr, mag, euler)
            t += 0.02
            time.sleep(0.015)

    def _push_data(self, acc, gyr, mag, euler=None):
        for ax, val in zip(["x","y","z"], acc): self.raw_buf["accel"][ax].append(val)
        for ax, val in zip(["x","y","z"], gyr): self.raw_buf["gyro"][ax].append(val)
        for ax, val in zip(["x","y","z"], mag):  self.raw_buf["mag"][ax].append(val)
        if euler is not None:
            for lbl, val in zip(["roll","pitch","yaw"], euler):
                self.euler_buf[lbl].append(val)

    def feed_from_pipeline(self, raw_acc, raw_gyro, raw_mag, processed_euler=None, processed_quat=None):
        self._push_data(raw_acc, raw_gyro, raw_mag, processed_euler)

    def on_close(self):
        self._running = False
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=0.5)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SingleSensorVizApp(root)
    root.mainloop()