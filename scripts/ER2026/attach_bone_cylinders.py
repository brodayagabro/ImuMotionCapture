"""Создать цветные цилиндры вдоль костей всех арматур текущей сцены.

Blender: Scripting > Open > этот файл > Run Script (Alt+P).
Сохраните .blend обычным способом, чтобы сохранить созданные объекты.
Повторный запуск пересоздаёт только цилиндры, созданные этим скриптом
для выбранных арматур. Исходные кости, анимация и другие меши не меняются.

Привязка работает без обработчиков кадров и без повторного запуска скрипта:
https://docs.blender.org/api/main/bpy.types.CopyTransformsConstraint.html
"""

import colorsys
import math

import bpy


# Пустой кортеж означает все арматуры сцены.
# Для конкретных арматур: ARMATURE_NAMES = ("custom", "noitom")
ARMATURE_NAMES = ()
RADIUS = 0.015  # Радиус в локальных единицах арматуры, масштаб наследуется.
SIDES = 24
SHOW_COLORS_IN_SOLID = True
COLORS = {
    "custom": (0.03, 0.43, 0.63, 1.0),
    "noitom": (0.87, 0.24, 0.07, 1.0),
}

_TAG = "imu_bone_cylinders_generator"
_VERSION = "1"
_FOLLOW = "IMU: follow bone"


def _is_generated(item):
    return item.get(_TAG) == _VERSION


def _collection(scene):
    collection = next(
        (item for item in bpy.data.collections if _is_generated(item)), None
    )
    if collection is None:
        collection = bpy.data.collections.new("IMU Bone Cylinders")
        collection[_TAG] = _VERSION
    if collection not in scene.collection.children_recursive:
        scene.collection.children.link(collection)
    return collection


def _color(rig, index):
    if rig.name in COLORS:
        return COLORS[rig.name]
    hue = (0.16 + index * 0.61803398875) % 1.0
    return (*colorsys.hsv_to_rgb(hue, 0.70, 0.85), 1.0)


def _material(rig, color):
    name = f"IMU Cylinder Color - {rig.name}"
    material = bpy.data.materials.get(name)
    if material is None or not _is_generated(material):
        material = bpy.data.materials.new(name)
        material[_TAG] = _VERSION
    material.diffuse_color = color
    material.use_nodes = True
    shader = next(
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 0.45
    return material


def _remove_previous(rigs):
    for obj in list(bpy.data.objects):
        if not _is_generated(obj):
            continue
        follow = obj.constraints.get(_FOLLOW)
        if follow is None or follow.type != "COPY_TRANSFORMS":
            continue
        if follow.target not in rigs:
            continue
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0 and _is_generated(mesh):
            bpy.data.meshes.remove(mesh)


def _cylinder_mesh(name, length, radius, sides):
    # Локальная ось Y кости идёт от head к tail. Начало меша находится в head.
    vertices = [
        (radius * math.cos(2 * math.pi * i / sides), y,
         radius * math.sin(2 * math.pi * i / sides))
        for y in (0.0, length)
        for i in range(sides)
    ]
    faces = [
        (i, sides + i, sides + (i + 1) % sides, (i + 1) % sides)
        for i in range(sides)
    ]
    faces.extend((tuple(range(sides)), tuple(reversed(range(sides, 2 * sides)))))
    mesh = bpy.data.meshes.new(name)
    mesh[_TAG] = _VERSION
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = polygon.index < sides
    return mesh


def main(armature_names=ARMATURE_NAMES, radius=RADIUS, sides=SIDES):
    """Создать по цилиндру на кость и вернуть список созданных объектов."""
    if bpy.context.mode == "EDIT_ARMATURE":
        raise RuntimeError("Сначала выйдите из Edit Mode арматуры в Object или Pose Mode.")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("RADIUS должен быть положительным конечным числом.")
    if not isinstance(sides, int) or sides < 3:
        raise ValueError("SIDES должен быть целым числом не меньше 3.")

    scene = bpy.context.scene
    rigs = sorted(
        (obj for obj in scene.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )
    if armature_names:
        missing = set(armature_names) - {rig.name for rig in rigs}
        if missing:
            raise ValueError("Арматуры не найдены в сцене: " + ", ".join(sorted(missing)))
        rigs = [rig for rig in rigs if rig.name in armature_names]
    if not rigs:
        raise RuntimeError("В текущей сцене нет арматур.")

    _remove_previous(rigs)
    collection = _collection(scene)
    created = []
    for index, rig in enumerate(rigs):
        color = _color(rig, index)
        material = _material(rig, color)
        for bone in rig.data.bones:
            if bone.length < 1e-8:
                print(f"Пропущена кость нулевой длины: {rig.name} / {bone.name}")
                continue
            name = f"IMU_Cyl_{rig.name}_{bone.name}"
            mesh = _cylinder_mesh(name, bone.length, radius, sides)
            obj = bpy.data.objects.new(name, mesh)
            obj[_TAG] = _VERSION
            obj.color = color
            mesh.materials.append(material)
            collection.objects.link(obj)

            # Матрица кости уже включает анимацию, положение и масштаб арматуры.
            follow = obj.constraints.new("COPY_TRANSFORMS")
            follow.name = _FOLLOW
            follow.target = rig
            follow.subtarget = bone.name
            follow.head_tail = 0.0
            follow.target_space = "WORLD"
            follow.owner_space = "WORLD"
            follow.mix_mode = "REPLACE"
            follow.use_bbone_shape = False
            follow.remove_target_shear = False
            created.append(obj)

    # Цвет виден и в Solid, и в Material Preview / финальном рендере.
    if SHOW_COLORS_IN_SOLID:
        scene.display.shading.color_type = "MATERIAL"
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.shading.color_type = "MATERIAL"
    bpy.context.view_layer.update()
    print(f"Создано цилиндров: {len(created)}. Арматур: {len(rigs)}.")
    return created


if __name__ == "__main__":
    main()
