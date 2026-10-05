"""Explicit constant extrinsics, independent of reference resets and filtering."""
from dataclasses import dataclass
from .quaternion_utils import compose, inverse, normalize


@dataclass(frozen=True)
class SegmentCalibration:
    """W' -> experiment W and segment B -> sensor S, all Hamilton wxyz.

    q_WB = q_WW' * q_W'S * q_SB. The right-hand mounting offset is distinct
    from the left-hand world alignment. A reference pose solves q_SB once.
    """
    q_world_alignment: tuple[float, float, float, float]
    q_sensor_segment: tuple[float, float, float, float]

    @classmethod
    def from_pose(cls, q_world_sensor, q_segment_target=(1., 0., 0., 0.),
                  q_world_alignment=(1., 0., 0., 0.)):
        alignment = normalize(q_world_alignment)
        mounting = compose(inverse(q_world_sensor), compose(inverse(alignment), q_segment_target))
        return cls(tuple(float(v) for v in alignment), tuple(float(v) for v in mounting))

    def apply(self, q_world_sensor):
        return compose(self.q_world_alignment, compose(q_world_sensor, self.q_sensor_segment))
