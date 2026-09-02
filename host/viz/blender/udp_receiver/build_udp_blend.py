"""Build the cleaned Human_spine_UDP.blend project.

Usage:
    blender --background --python blender/build_udp_blend.py -- \
        blender/Human_spine_N_sensors.blend1 \
        blender/Human_spine_UDP.blend \
        blender/udp_mocap.py
"""

import os
import sys

import bpy
from mathutils import Quaternion


def arguments():
    if "--" not in sys.argv:
        raise SystemExit("expected source.blend output.blend udp_mocap.py after --")
    values = sys.argv[sys.argv.index("--") + 1 :]
    if len(values) != 3:
        raise SystemExit("expected exactly three arguments after --")
    return tuple(os.path.abspath(value) for value in values)


source_blend, output_blend, driver_path = arguments()
print("[build_udp_blend] source:", source_blend)
print("[build_udp_blend] output:", output_blend)
print("[build_udp_blend] driver:", driver_path)

if not os.path.isfile(source_blend):
    raise FileNotFoundError(source_blend)
if not os.path.isfile(driver_path):
    raise FileNotFoundError(driver_path)

bpy.ops.wm.open_mainfile(filepath=source_blend)

rig = bpy.data.objects.get("Human_Rig")
if rig is None or rig.type != "ARMATURE":
    raise RuntimeError("Human_Rig armature was not found")

required_bones = {
    "spine",
    "shoulder.L",
    "forearm.L",
    "shoulder.R",
    "forearm.R",
}
missing_bones = sorted(required_bones.difference(rig.pose.bones.keys()))
if missing_bones:
    raise RuntimeError("required pose bones were not found: %s" % ", ".join(missing_bones))

# The original creation script left Blender's default bone beside the actual rig.
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
default_bone = rig.data.edit_bones.get("Bone")
if default_bone is not None:
    rig.data.edit_bones.remove(default_bone)
    print("[build_udp_blend] removed unused default bone 'Bone'")
bpy.ops.object.mode_set(mode="POSE")

# Save a deterministic rest pose and quaternion mode.
for bone in rig.pose.bones:
    bone.rotation_mode = "QUATERNION"
    bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
bpy.ops.object.mode_set(mode="OBJECT")
rig.show_in_front = True

# Remove obsolete serial experiments and duplicated scripts from the cleaned copy.
for text in list(bpy.data.texts):
    bpy.data.texts.remove(text)

with open(driver_path, "r", encoding="utf-8") as source:
    driver_source = source.read()

# Reject a broken external driver before modifying and saving the deliverable.
compile(driver_source, driver_path, "exec")

driver_text = bpy.data.texts.new("udp_mocap.py")
driver_text.write(driver_source)

status_text = bpy.data.texts.new("UDP_MOCAP_STATUS")
status_text.write(
    "Run udp_mocap.py from the Scripting workspace.\n"
    "Edit DEVICE_IP and AXIS_MAPS at the top of that text first.\n\n"
    "Canonical TCA channel map (raw IDs 1..5 are auto-detected):\n"
    "  0 -> shoulder.L\n"
    "  1 -> forearm.L\n"
    "  2 -> spine\n"
    "  6 -> forearm.R\n"
    "  7 -> shoulder.R\n"
)

# Make the cleaned driver visible when the Scripting workspace is opened.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "TEXT_EDITOR":
            area.spaces.active.text = driver_text
            area.spaces.active.top = 0

if bpy.context.window is not None:
    scripting = bpy.data.workspaces.get("Scripting")
    if scripting is not None:
        bpy.context.window.workspace = scripting

rig["udp_mocap_driver"] = "udp_mocap.py"
rig["sensor_0"] = "shoulder.L"
rig["sensor_1"] = "forearm.L"
rig["sensor_2"] = "spine"
rig["sensor_6"] = "forearm.R"
rig["sensor_7"] = "shoulder.R"
rig["udp_port"] = 4210
rig["sensor_id_mode"] = "auto: tca_channel or sequential"

scene = bpy.context.scene
scene["neuromorph_udp_mocap"] = True
scene["udp_mocap_build_version"] = 3
scene["udp_mocap_protocol"] = "FRAME + Q, auto IDs: TCA 0/1/2/6/7 or sequential 1..5"
scene["calibration_pose"] = "N-pose: upright, arms down, palms toward body"

os.makedirs(os.path.dirname(output_blend), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=output_blend, check_existing=False)
print("[build_udp_blend] saved:", bpy.data.filepath)
