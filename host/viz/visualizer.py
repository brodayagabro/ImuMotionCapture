import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class Visualizer:
    def __init__(self, parent_frame):
        self.fig = plt.Figure(figsize=(8, 6), facecolor='white')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        r = 1
        self.cube_vertices = np.array([
            [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
            [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]
        ])
        self.cube_edges = [
            [0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        self.axis_length = 2.5

    def draw(self, euler_valid: bool, euler_deg: np.ndarray):
        self.ax.clear()
        if euler_valid:
            roll, pitch, yaw = np.radians(euler_deg[0]), np.radians(euler_deg[1]), np.radians(euler_deg[2])
            Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            rotated = self.cube_vertices @ R.T
            color = plt.cm.viridis((euler_deg[2] % 360) / 360)
        else:
            rotated = self.cube_vertices
            color = 'gray'

        for edge in self.cube_edges:
            pts = rotated[edge]
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2, alpha=0.8)

        self.ax.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c='red', s=50, alpha=0.9)
        self.ax.plot([0, self.axis_length], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.5)
        self.ax.plot([0, 0], [0, self.axis_length], [0, 0], 'g-', linewidth=2, alpha=0.5)
        self.ax.plot([0, 0], [0, 0], [0, self.axis_length], 'b-', linewidth=2, alpha=0.5)

        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylim([-2.5, 2.5])
        self.ax.set_zlim([-2.5, 2.5])
        self.ax.set_xlabel('X (Red)')
        self.ax.set_ylabel('Y (Green)')
        self.ax.set_zlabel('Z (Blue)')
        self.ax.set_title('IMU Orientation (IMUFusion AHRS)', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()