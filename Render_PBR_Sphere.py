"""Render generated PBR texture samples on a sphere with Blender Cycles.

The generated texture folders are expected to contain PBR map files such as:

    BaseColor.png, Roughness.png, Metallic.png, Normal.png, Displacement.png

This script can render one material folder directly with
``render_single_pbr_sphere`` or batch-render one sample from every generated PBR
model when executed as a Blender Python script.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random

import bpy
from mathutils import Vector


# Batch-render defaults. Change these paths for your local generated dataset.
DEFAULT_BATCH_ROOT = "output_pbrs"
DEFAULT_HDRI_FILE = Path("HDRI/cannon_4k.exr")
DEFAULT_SAMPLE_SUBDIR = Path("textures") / "sample_002"
DEFAULT_OUTPUT_NAME = "render_sphere1.jpg"

# Texture names produced by the generated PBR scripts.
TEXTURE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DISPLACEMENT_FILE_EXTENSIONS = TEXTURE_FILE_EXTENSIONS + (".exr",)
BASE_COLOR_NAMES = ("BaseColor", "basecolor", "Albedo", "albedo", "Diffuse", "diffuse")
ROUGHNESS_NAMES = ("Roughness", "roughness")
METALLIC_NAMES = ("Metallic", "metallic", "Metalness", "metalness")
NORMAL_NAMES = ("Normal", "normal", "NormalGL", "normalgl")
DISPLACEMENT_NAMES = ("Displacement", "displacement", "Height", "height")


def clear_scene() -> None:
    """Reset Blender to an empty scene before each render."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def enable_cycles_displacement_mode(scene) -> None:
    """Enable old Blender experimental displacement mode when it exists."""
    if not hasattr(scene.cycles, "feature_set"):
        return

    try:
        scene.cycles.feature_set = "EXPERIMENTAL"
    except Exception:
        pass


def set_material_displacement_method_safe(material, method: str = "DISPLACEMENT") -> None:
    """Set material displacement mode on Blender versions that still expose it."""
    if not hasattr(material, "cycles") or not hasattr(material.cycles, "displacement_method"):
        return

    try:
        material.cycles.displacement_method = method
    except Exception:
        pass


def get_material_node_tree(material):
    """Return a material node tree without touching deprecated APIs when possible."""
    node_tree = getattr(material, "node_tree", None)
    if node_tree is not None:
        return node_tree

    material.use_nodes = True
    return material.node_tree


def get_world_node_tree(world):
    """Return a world node tree without touching deprecated APIs when possible."""
    node_tree = getattr(world, "node_tree", None)
    if node_tree is not None:
        return node_tree

    world.use_nodes = True
    return world.node_tree


def setup_cycles(cycles_samples: int = 64) -> None:
    """Configure Cycles and use the first available GPU backend."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cycles_samples
    enable_cycles_displacement_mode(scene)

    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        scene.cycles.device = "CPU"
        print("Cycles preferences unavailable. Using CPU.")
        return

    for backend in ("OPTIX", "CUDA", "HIP", "METAL"):
        try:
            cycles_prefs.compute_device_type = backend
            cycles_prefs.get_devices()
            for device in cycles_prefs.devices:
                device.use = True
            scene.cycles.device = "GPU"
            print(f"Using Cycles backend: {backend}")
            return
        except Exception as err:
            print(f"Backend {backend} unavailable: {err}")

    scene.cycles.device = "CPU"
    print("No GPU backend available. Using CPU.")


def set_render_output(scene, output_path: str, image_size: int) -> None:
    """Set square render resolution and image format from the output extension."""
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    suffix = Path(output_path).suffix.lower()
    scene.render.image_settings.file_format = "PNG" if suffix == ".png" else "JPEG"


def ensure_parent_dir(path: str | None) -> None:
    """Create the parent folder for an output path if one was provided."""
    if path is None:
        return

    parent = Path(path).parent
    if str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def load_object(object_path: str, use_displacement: bool = True):
    """Import an OBJ/GLTF mesh and join selected mesh parts into one object."""
    ext = Path(object_path).suffix.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=object_path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=object_path)
    else:
        print(f"Unsupported object format: {object_path}")
        return None

    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not mesh_objects:
        print(f"No mesh objects found in: {object_path}")
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    obj = bpy.context.active_object

    if use_displacement:
        add_subdivision_modifier(obj, "Real_Displacement_Subdivision", levels=2, render_levels=5)
    return obj


def add_subdivision_modifier(obj, name: str, levels: int, render_levels: int):
    """Add Catmull-Clark subdivision when either subdivision level is positive."""
    if levels <= 0 and render_levels <= 0:
        return None

    modifier = obj.modifiers.new(name=name, type="SUBSURF")
    modifier.subdivision_type = "CATMULL_CLARK"
    modifier.levels = levels
    modifier.render_levels = render_levels

    if hasattr(modifier, "use_adaptive_subdivision"):
        try:
            modifier.use_adaptive_subdivision = True
        except Exception:
            pass
    return modifier


def normalize_object(obj, target_size: float = 1.0) -> None:
    """Scale an object to a maximum dimension and center it at the origin."""
    bpy.context.view_layer.update()
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_corner = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    size = max_corner - min_corner
    max_dim = max(size.x, size.y, size.z)

    if max_dim == 0:
        return

    obj.scale *= target_size / max_dim
    bpy.context.view_layer.update()
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    obj.location -= center


def frame_camera(obj, margin: float = 1.3, randomize: bool = False):
    """Create a camera that frames the object and tracks its center."""
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    bpy.context.view_layer.update()
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    radius = max((corner - center).length for corner in bbox)
    distance = (radius * margin) / math.tan(camera.data.angle / 2.0)

    if randomize:
        direction = Vector(
            (
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(0.3, 1.0),
            )
        ).normalized()
    else:
        direction = Vector((0.0, 0.6, 0.0)).normalized()

    camera.location = center + direction * distance
    track = camera.constraints.new(type="TRACK_TO")
    track.target = obj
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return camera


def setup_hdri(
    hdri_file: str,
    black_background: bool = False,
    background_color=(0.0, 0.0, 0.0, 1.0),
    hdri_strength: float = 1.0,
) -> None:
    """Use HDRI lighting and optionally hide the HDRI from the camera."""
    world = bpy.data.worlds.new("HDRI_World")
    bpy.context.scene.world = world
    node_tree = get_world_node_tree(world)

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(hdri_file)

    hdri_background = nodes.new("ShaderNodeBackground")
    hdri_background.inputs["Strength"].default_value = hdri_strength
    links.new(environment.outputs["Color"], hdri_background.inputs["Color"])

    if not black_background:
        links.new(hdri_background.outputs["Background"], output.inputs["Surface"])
        return

    camera_background = nodes.new("ShaderNodeBackground")
    camera_background.inputs["Color"].default_value = background_color
    camera_background.inputs["Strength"].default_value = 1.0

    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(hdri_background.outputs["Background"], mix.inputs[1])
    links.new(camera_background.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def texture_name_candidates(possible_names, extensions):
    """Yield common filename variants for one texture map type."""
    seen = set()
    for name in possible_names:
        root, ext = os.path.splitext(name)
        if ext:
            candidates = [name]
            if ext.lower() not in extensions:
                candidates = []
        else:
            root = name
            candidates = []

        candidates.extend(root + extension for extension in extensions)
        candidates.extend(root + extension.upper() for extension in extensions)

        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def find_texture(pbr_dir: str, possible_names, extensions=TEXTURE_FILE_EXTENSIONS):
    """Find the first matching texture file in a PBR sample directory."""
    for name in texture_name_candidates(possible_names, extensions):
        path = Path(pbr_dir) / name
        if path.exists():
            return str(path)
    return None


def get_displacement_texture_path(pbr_dir: str):
    """Find the height/displacement texture for geometry displacement."""
    return find_texture(pbr_dir, DISPLACEMENT_NAMES, extensions=DISPLACEMENT_FILE_EXTENSIONS)


def create_texture_node(nodes, links, mapping, image_path: str, is_data: bool = False):
    """Create an image texture node and connect camera mapping to it."""
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(image_path)

    if is_data:
        texture.image.colorspace_settings.name = "Non-Color"

    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    print(f"Loaded texture: {image_path}")
    return texture


def create_camera_mapping_nodes(nodes, links, camera):
    """Use camera coordinates so every generated texture gets consistent preview mapping."""
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    texcoord.object = camera
    mapping.inputs["Scale"].default_value = (1.0, -1.0, 1.0)
    links.new(texcoord.outputs["Camera"], mapping.inputs["Vector"])
    return mapping


def create_pbr_material(
    pbr_dir: str,
    camera,
    use_displacement: bool = True,
    displacement_scale: float = 0.03,
):
    """Build a Blender material from the PBR maps found in one sample folder."""
    material = bpy.data.materials.new("PBR_Mat")
    node_tree = get_material_node_tree(material)

    if use_displacement:
        set_material_displacement_method_safe(material)

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    mapping = create_camera_mapping_nodes(nodes, links, camera)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")

    base_path = find_texture(pbr_dir, BASE_COLOR_NAMES)
    if base_path:
        base = create_texture_node(nodes, links, mapping, base_path, is_data=False)
        links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    roughness_path = find_texture(pbr_dir, ROUGHNESS_NAMES)
    if roughness_path:
        roughness = create_texture_node(nodes, links, mapping, roughness_path, is_data=True)
        links.new(roughness.outputs["Color"], bsdf.inputs["Roughness"])

    metallic_path = find_texture(pbr_dir, METALLIC_NAMES)
    if metallic_path:
        metallic = create_texture_node(nodes, links, mapping, metallic_path, is_data=True)
        links.new(metallic.outputs["Color"], bsdf.inputs["Metallic"])

    normal_path = find_texture(pbr_dir, NORMAL_NAMES)
    if normal_path:
        normal = create_texture_node(nodes, links, mapping, normal_path, is_data=True)
        normal_map = nodes.new("ShaderNodeNormalMap")
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    if use_displacement:
        displacement_path = get_displacement_texture_path(pbr_dir)
        if displacement_path:
            displacement = create_texture_node(nodes, links, mapping, displacement_path, is_data=True)
            displacement_node = nodes.new("ShaderNodeDisplacement")
            displacement_node.inputs["Scale"].default_value = displacement_scale
            displacement_node.inputs["Midlevel"].default_value = 0.5
            links.new(displacement.outputs["Color"], displacement_node.inputs["Height"])
            links.new(displacement_node.outputs["Displacement"], output.inputs["Displacement"])
        else:
            print("use_displacement=True, but no height/displacement texture found.")

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def apply_material(obj, material) -> None:
    """Replace all object materials with the provided material."""
    obj.data.materials.clear()
    obj.data.materials.append(material)


def create_mask_material(
    pbr_dir: str,
    camera,
    use_displacement: bool = True,
    displacement_scale: float = 0.03,
):
    """Create a white emission material for object-mask renders."""
    material = bpy.data.materials.new("Mask_Mat")
    node_tree = get_material_node_tree(material)

    if use_displacement:
        set_material_displacement_method_safe(material)

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    mapping = create_camera_mapping_nodes(nodes, links, camera)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    if use_displacement:
        displacement_path = get_displacement_texture_path(pbr_dir)
        if displacement_path:
            displacement = create_texture_node(nodes, links, mapping, displacement_path, is_data=True)
            displacement_node = nodes.new("ShaderNodeDisplacement")
            displacement_node.inputs["Scale"].default_value = displacement_scale
            displacement_node.inputs["Midlevel"].default_value = 0.5
            links.new(displacement.outputs["Color"], displacement_node.inputs["Height"])
            links.new(displacement_node.outputs["Displacement"], output.inputs["Displacement"])

    return material


def render_mask(
    scene,
    obj,
    camera,
    pbr_dir: str,
    output_mask_path: str,
    use_displacement: bool = True,
    displacement_scale: float = 0.03,
) -> None:
    """Render a white object mask on a black background."""
    original_world = scene.world
    original_materials = obj.data.materials[:]
    original_filepath = scene.render.filepath
    original_file_format = scene.render.image_settings.file_format
    original_film_transparent = scene.render.film_transparent

    world = bpy.data.worlds.new("Mask_Black_World")
    scene.world = world
    node_tree = get_world_node_tree(world)

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])

    mask_material = create_mask_material(
        pbr_dir=pbr_dir,
        camera=camera,
        use_displacement=use_displacement,
        displacement_scale=displacement_scale,
    )
    apply_material(obj, mask_material)

    scene.render.filepath = output_mask_path
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    bpy.ops.render.render(write_still=True)

    obj.data.materials.clear()
    for material in original_materials:
        obj.data.materials.append(material)

    scene.world = original_world
    scene.render.filepath = original_filepath
    scene.render.image_settings.file_format = original_file_format
    scene.render.film_transparent = original_film_transparent


def add_real_displacement(
    obj,
    pbr_dir: str,
    texture_name: str,
    displacement_scale: float,
    subdivision_levels: int,
    subdivision_render_levels: int,
    apply_displacement_modifier: bool,
):
    """Add optional subdivision and an image-based Displace modifier to a mesh."""
    displacement_path = get_displacement_texture_path(pbr_dir)
    if displacement_path is None:
        print("use_displacement=True, but no Height/Displacement texture found.")
        return None

    add_subdivision_modifier(
        obj,
        name=f"{texture_name}_Subdivision",
        levels=subdivision_levels,
        render_levels=subdivision_render_levels,
    )

    print(f"Using real mesh displacement texture: {displacement_path}")
    print(f"Displacement strength: {displacement_scale}")
    image = bpy.data.images.load(displacement_path)
    image.colorspace_settings.name = "Non-Color"

    texture = bpy.data.textures.new(name=f"{texture_name}_Displacement_Texture", type="IMAGE")
    texture.image = image

    displacement_modifier = obj.modifiers.new(name=f"{texture_name}_Displacement", type="DISPLACE")
    displacement_modifier.texture = texture
    displacement_modifier.strength = displacement_scale
    displacement_modifier.mid_level = 0.5

    if hasattr(displacement_modifier, "texture_coords"):
        try:
            displacement_modifier.texture_coords = "UV"
        except Exception:
            pass

    if apply_displacement_modifier:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for modifier in list(obj.modifiers):
            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except Exception as err:
                print(f"Could not apply modifier {modifier.name}: {err}")
        print("Subdivision/displacement modifiers applied to geometry.")

    return displacement_path


def render_single_pbr_object(
    hdri_file: str,
    pbr_dir: str,
    object_path: str,
    output_image_path: str,
    output_mask_path: str | None = None,
    image_size: int = 512,
    cycles_samples: int = 64,
    use_displacement: bool = True,
    displacement_scale: float = 0.02,
    normalize_target_size: float = 1.0,
    camera_margin: float = 1.3,
    subdivision_levels: int = 1,
    subdivision_render_levels: int = 2,
    apply_displacement_modifier: bool = True,
    black_background: bool = False,
    background_color=(0.0, 0.0, 0.0, 1.0),
    hdri_strength: float = 1.0,
):
    """Render one imported object with one PBR material folder."""
    clear_scene()
    ensure_parent_dir(output_image_path)
    ensure_parent_dir(output_mask_path)
    setup_cycles(cycles_samples=cycles_samples)

    scene = bpy.context.scene
    set_render_output(scene, output_image_path, image_size)

    obj = load_object(object_path=object_path, use_displacement=False)
    if obj is None:
        raise RuntimeError(f"Could not load object: {object_path}")

    normalize_object(obj=obj, target_size=normalize_target_size)

    displacement_texture_path = None
    if use_displacement:
        displacement_texture_path = add_real_displacement(
            obj=obj,
            pbr_dir=pbr_dir,
            texture_name="Object_Real",
            displacement_scale=displacement_scale,
            subdivision_levels=subdivision_levels,
            subdivision_render_levels=subdivision_render_levels,
            apply_displacement_modifier=apply_displacement_modifier,
        )

    camera = frame_camera(obj=obj, margin=camera_margin)
    setup_hdri(
        hdri_file=hdri_file,
        black_background=black_background,
        background_color=background_color,
        hdri_strength=hdri_strength,
    )

    material = create_pbr_material(
        pbr_dir=pbr_dir,
        camera=camera,
        use_displacement=False,
        displacement_scale=displacement_scale,
    )
    apply_material(obj, material)

    scene.render.filepath = output_image_path
    print(f"Rendering RGB to: {output_image_path}")
    bpy.ops.render.render(write_still=True)

    if output_mask_path is not None:
        print(f"Rendering mask to: {output_mask_path}")
        render_mask(
            scene=scene,
            obj=obj,
            camera=camera,
            pbr_dir=pbr_dir,
            output_mask_path=output_mask_path,
            use_displacement=False,
            displacement_scale=displacement_scale,
        )

    print("Object render complete.")
    return {
        "image": output_image_path,
        "mask": output_mask_path,
        "object": object_path,
        "hdri": hdri_file,
        "pbr_dir": pbr_dir,
        "cycles_samples": cycles_samples,
        "use_displacement": use_displacement,
        "displacement_scale": displacement_scale,
        "displacement_texture": displacement_texture_path,
        "black_background": black_background,
        "background_color": background_color,
        "hdri_strength": hdri_strength,
        "subdivision_levels": subdivision_levels,
        "subdivision_render_levels": subdivision_render_levels,
        "applied_displacement_modifier": apply_displacement_modifier,
    }


def create_uv_sphere(radius: float, segments: int, rings: int):
    """Create a smooth UV sphere for material preview renders."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=radius,
        location=(0.0, 0.0, 0.0),
    )
    sphere = bpy.context.active_object
    sphere.name = "PBR_Render_Sphere"
    bpy.ops.object.shade_smooth()
    return sphere


def render_single_pbr_sphere(
    hdri_file: str,
    pbr_dir: str,
    output_image_path: str,
    output_mask_path: str | None = None,
    image_size: int = 512,
    cycles_samples: int = 64,
    use_displacement: bool = True,
    displacement_scale: float = 0.03,
    sphere_radius: float = 0.5,
    sphere_segments: int = 256,
    sphere_rings: int = 128,
    subdivision_levels: int = 1,
    subdivision_render_levels: int = 2,
    camera_margin: float = 1.3,
    apply_displacement_modifier: bool = True,
    black_background: bool = True,
    background_color=(0.0, 0.0, 0.0, 1.0),
    hdri_strength: float = 1.0,
):
    """Render one PBR material on a UV sphere.

    When ``use_displacement`` is true, displacement is applied as real geometry
    before rendering. The material then disables shader displacement to avoid
    applying the same height map twice.
    """
    clear_scene()
    ensure_parent_dir(output_image_path)
    ensure_parent_dir(output_mask_path)
    setup_cycles(cycles_samples=cycles_samples)

    scene = bpy.context.scene
    set_render_output(scene, output_image_path, image_size)

    sphere = create_uv_sphere(
        radius=sphere_radius,
        segments=sphere_segments,
        rings=sphere_rings,
    )

    displacement_texture_path = None
    if use_displacement:
        displacement_texture_path = add_real_displacement(
            obj=sphere,
            pbr_dir=pbr_dir,
            texture_name="Sphere_Real",
            displacement_scale=displacement_scale,
            subdivision_levels=subdivision_levels,
            subdivision_render_levels=subdivision_render_levels,
            apply_displacement_modifier=apply_displacement_modifier,
        )
    else:
        add_subdivision_modifier(
            sphere,
            name="Sphere_Subdivision",
            levels=subdivision_levels,
            render_levels=subdivision_render_levels,
        )

    camera = frame_camera(obj=sphere, margin=camera_margin)
    setup_hdri(
        hdri_file=hdri_file,
        black_background=black_background,
        background_color=background_color,
        hdri_strength=hdri_strength,
    )

    material = create_pbr_material(
        pbr_dir=pbr_dir,
        camera=camera,
        use_displacement=False,
        displacement_scale=displacement_scale,
    )
    apply_material(sphere, material)

    scene.render.filepath = output_image_path
    print(f"Rendering sphere RGB to: {output_image_path}")
    bpy.ops.render.render(write_still=True)

    if output_mask_path is not None:
        print(f"Rendering sphere mask to: {output_mask_path}")
        render_mask(
            scene=scene,
            obj=sphere,
            camera=camera,
            pbr_dir=pbr_dir,
            output_mask_path=output_mask_path,
            use_displacement=False,
            displacement_scale=displacement_scale,
        )

    print("Sphere render complete.")
    return {
        "image": output_image_path,
        "mask": output_mask_path,
        "hdri": hdri_file,
        "pbr_dir": pbr_dir,
        "cycles_samples": cycles_samples,
        "use_displacement": use_displacement,
        "displacement_scale": displacement_scale,
        "sphere_radius": sphere_radius,
        "sphere_segments": sphere_segments,
        "sphere_rings": sphere_rings,
        "subdivision_levels": subdivision_levels,
        "subdivision_render_levels": subdivision_render_levels,
        "displacement_texture": displacement_texture_path,
        "applied_displacement_modifier": apply_displacement_modifier,
    }


def iter_generated_model_dirs(batch_root: Path):
    """Yield every generated model directory under a topic/model tree."""
    for topic_dir in sorted(batch_root.iterdir()):
        if not topic_dir.is_dir():
            continue

        for model_dir in sorted(topic_dir.iterdir()):
            if model_dir.is_dir():
                yield model_dir


def render_batch_spheres(
    batch_root: Path | None = DEFAULT_BATCH_ROOT,
    hdri_file: Path = DEFAULT_HDRI_FILE,
    sample_subdir: Path = DEFAULT_SAMPLE_SUBDIR,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> None:
    """Render one selected sample folder from every generated PBR model."""
    if batch_root is None:
        raise ValueError("Set DEFAULT_BATCH_ROOT or pass batch_root before running batch renders.")
    batch_root = Path(batch_root)
    hdri_file = Path(hdri_file)
    sample_subdir = Path(sample_subdir)

    if not batch_root.is_dir():
        raise FileNotFoundError(f"Batch root does not exist: {batch_root}")
    if not hdri_file.is_file():
        raise FileNotFoundError(f"HDRI file does not exist: {hdri_file}")

    rendered_count = 0
    for workdir in iter_generated_model_dirs(batch_root):
        pbr_dir = workdir / sample_subdir
        output_image_path = workdir / output_name

        if output_image_path.exists() or not pbr_dir.is_dir():
            continue

        rendered_count += 1
        render_single_pbr_sphere(
            hdri_file=str(hdri_file),
            pbr_dir=str(pbr_dir),
            output_image_path=str(output_image_path),
            image_size=500,
            cycles_samples=100,
            use_displacement=True,
            sphere_radius=0.5,
            displacement_scale=0.014,
            camera_margin=0.8,
        )
        print(workdir)
        print(f"{rendered_count}) Finished {output_image_path}")


if __name__ == "__main__":
    render_batch_spheres()
