"""Regression tests for the Tk cube viewer sensor mapping."""

from __future__ import annotations

import unittest

from cube_viewer.mpu_udp_viewer_tk import DEFAULT_SENSOR_IDS, QuaternionViewer


class ParserState:
    def __init__(self) -> None:
        self.quaternions: dict[int, tuple[float, float, float, float]] = {}
        self.active_sensor_id_mode: str | None = None
        self.rx_quaternion_count = 0
        self.needs_redraw = False


class TkCubeMappingTests(unittest.TestCase):
    def test_default_mapping_matches_physical_tca_channels(self) -> None:
        self.assertEqual(DEFAULT_SENSOR_IDS, (0, 1, 2, 6, 7))

    def test_one_physical_frame_updates_all_five_ids(self) -> None:
        state = ParserState()
        packet = "\n".join(
            (
                "FRAME 1 100 5",
                "Q 0 1 0 0 0",
                "Q 1 1 0 0 0",
                "Q 2 1 0 0 0",
                "Q 6 1 0 0 0",
                "Q 7 1 0 0 0",
            )
        )
        QuaternionViewer._parse_packet(state, packet)  # type: ignore[arg-type]
        self.assertEqual(set(state.quaternions), set(DEFAULT_SENSOR_IDS))
        self.assertEqual(state.active_sensor_id_mode, "tca_channel")
        self.assertEqual(state.rx_quaternion_count, 5)
        self.assertTrue(state.needs_redraw)

    def test_sequential_frame_is_mapped_to_all_five_cubes(self) -> None:
        state = ParserState()
        packet = "\n".join(
            (
                "FRAME 1 100 5",
                "Q 1 1 0 0 0",
                "Q 2 1 0 0 0",
                "Q 3 1 0 0 0",
                "Q 4 1 0 0 0",
                "Q 5 1 0 0 0",
            )
        )
        QuaternionViewer._parse_packet(state, packet)  # type: ignore[arg-type]
        self.assertEqual(set(state.quaternions), set(DEFAULT_SENSOR_IDS))
        self.assertEqual(state.active_sensor_id_mode, "sequential")
        self.assertEqual(state.rx_quaternion_count, 5)
        self.assertTrue(state.needs_redraw)


if __name__ == "__main__":
    unittest.main()
