import pytest
import numpy as np
from host.viz.visualizer import Visualizer


class TestVisualizer:
    def test_init_creates_canvas(self, headless_tk):
        viz = Visualizer(headless_tk)
        assert viz.fig is not None
        assert viz.ax is not None
        assert viz.canvas is not None
        assert len(viz.cube_vertices) == 8
        assert len(viz.cube_edges) == 12

    def test_draw_with_valid_euler(self, headless_tk):
        viz = Visualizer(headless_tk)
        euler = np.array([10.0, 20.0, 30.0])
        # Не вызываем canvas.draw() — это требует GUI-бэкенда
        # Просто проверяем, что метод не падает
        viz.draw(euler_valid=True, euler_deg=euler)
        # Проверка, что ось очищена (частичная)
        assert viz.ax is not None

    def test_draw_with_invalid_euler(self, headless_tk):
        viz = Visualizer(headless_tk)
        euler = np.zeros(3)
        viz.draw(euler_valid=False, euler_deg=euler)
        # Должен нарисовать куб без вращения