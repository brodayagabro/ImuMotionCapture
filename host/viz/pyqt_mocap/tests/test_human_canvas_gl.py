import numpy as np

from pyqt_mocap.human_canvas_gl import _segment_positions


def test_segment_positions_flattens_pairs_for_gl_lines() -> None:
    segments = (
        (np.array((0.0, 1.0, 2.0)), np.array((3.0, 4.0, 5.0))),
        (np.array((6.0, 7.0, 8.0)), np.array((9.0, 10.0, 11.0))),
    )

    positions = _segment_positions(segments)

    assert positions.dtype == np.float32
    assert positions.shape == (4, 3)
    np.testing.assert_array_equal(
        positions,
        np.array(
            (
                (0.0, 1.0, 2.0),
                (3.0, 4.0, 5.0),
                (6.0, 7.0, 8.0),
                (9.0, 10.0, 11.0),
            ),
            dtype=np.float32,
        ),
    )
