"""Russian semaphore, anatomical arms in the participant's frontal plane.

Angles: 0=participant's right, 90=up, 180=left, 270=down.
Source: N. Serebryany / B. Zhdanov, Signalman's Handbook, Russian semaphore
chart: https://flot.com/publications/books/shelf/signalman/83.htm
The source shows the signalman from the front (our mirror_mode=True).
"""
from dataclasses import dataclass
import math

POSE_SOURCE = "https://flot.com/publications/books/shelf/signalman/83.htm"
LOWER_LETTERS = frozenset("АТЦЧОНБГВД")
UPPER_LETTERS = frozenset("УЬЕЖЗЭЮЯФЫШЩКХЛМПР")


@dataclass(frozen=True)
class SemaphorePose:
    left_angle_deg: float
    right_angle_deg: float

    def __post_init__(self):
        if not all(math.isfinite(a) for a in (self.left_angle_deg, self.right_angle_deg)):
            raise ValueError("Углы позы должны быть конечными числами")


NEUTRAL = SemaphorePose(270., 270.)
SEMAPHORE_POSES = {
    "А": SemaphorePose(225, 315), "Б": SemaphorePose(315, 0),
    "В": SemaphorePose(270, 0), "Г": SemaphorePose(180, 270),
    "Д": SemaphorePose(180, 225), "Е": SemaphorePose(270, 45),
    "Ж": SemaphorePose(180, 45), "З": SemaphorePose(135, 0),
    "К": SemaphorePose(135, 225), "Л": SemaphorePose(225, 45),
    "М": SemaphorePose(135, 315), "Н": SemaphorePose(270, 315),
    "О": SemaphorePose(225, 270), "П": SemaphorePose(180, 90),
    "Р": SemaphorePose(90, 0), "Т": SemaphorePose(180, 0),
    "У": SemaphorePose(135, 45), "Ф": SemaphorePose(225, 90),
    "Х": SemaphorePose(315, 45), "Ц": SemaphorePose(225, 0),
    "Ч": SemaphorePose(180, 315), "Ш": SemaphorePose(135, 90),
    "Щ": SemaphorePose(90, 45), "Ь": SemaphorePose(90, 90),
    "Ы": SemaphorePose(90, 315), "Э": SemaphorePose(270, 45),
    "Ю": SemaphorePose(0, 45), "Я": SemaphorePose(135, 180),
}
# Cross-body poses still refer to anatomical arms: for Х and Ю the RIGHT
# arm points up-right (45°); the LEFT crosses down-right (315°) / right (0°).
# Horizontal image mirroring never swaps those arm identities.


def letter_group(letter):
    if letter in LOWER_LETTERS:
        return "lower"
    if letter in UPPER_LETTERS:
        return "upper"
    raise ValueError(f"Неизвестная буква семафора: {letter}")


def interpolate_pose(start, end, fraction):
    t = min(1., max(0., fraction))
    smooth = t * t * (3 - 2 * t)
    def angle(a, b):
        delta = (b - a + 180) % 360 - 180
        return (a + smooth * delta) % 360
    return SemaphorePose(angle(start.left_angle_deg, end.left_angle_deg),
                         angle(start.right_angle_deg, end.right_angle_deg))


def arm_geometry(pose, mirror_mode=True):
    """Normalized screen-plane coordinates (Y up), preserving arm identity.

    The requested true convention puts the participant's LEFT shoulder on
    the RIGHT side of the image. False reflects the entire image horizontally.
    """
    flip = -1 if mirror_mode else 1
    result = {}
    for name, x, angle in (("left", -.23, pose.left_angle_deg), ("right", .23, pose.right_angle_deg)):
        radians = math.radians(angle)
        start = (flip * x, .25)
        end = (flip * (x + .8 * math.cos(radians)), .25 + .8 * math.sin(radians))
        result[name] = (start, end)
    return result
