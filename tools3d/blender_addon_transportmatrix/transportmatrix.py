bl_info = {
    "name": "Generate Transport Matrix",
    "author": "Yannick Hold-Geoffroy",
    "version": (0, 1, 0),
    "blender": (2, 7, 9),
    "category": "Import-Export",
    "location": "File > Export > Generate Transport Matrix",
    "description": "Export the current camera viewport to a transport matrix.",
}

import bpy
from mathutils import Vector
import bpy_extras
import numpy as np

# TODOs:
# 1. Generate a mask to shrink the matrix size (remove pixels that doesn't intersect the scene)
# 2. Add support for (lambertian) albedo from blender's UI
# 3. Add anti-aliasing


# For each pixel, check intersection
# Inspirations:
# https://blender.stackexchange.com/questions/90698/projecting-onto-a-mirror-and-back/91019#91019
# https://blender.stackexchange.com/questions/13510/get-position-of-the-first-solid-crossing-limits-line-of-a-camera
def getClosestIntersection(clip_end, ray_begin, ray_direction):
    objs = bpy.data.objects

    min_location = None
    min_normal = None
    min_ = (clip_end, None, None)
    for obj in objs:
        if obj.type != 'MESH':
            continue
        #v1 = obj.matrix_world.inverted() * cam.matrix_world * Vector((0.0, 0.0, -cam.data.clip_end))

        mat_inv = obj.matrix_world.inverted()
        ray_origin = mat_inv * ray_begin
        ray_end = mat_inv * ray_direction.copy()
        #ray_obj_direction = (ray_end - ray_origin) * cam.data.clip_end
        success, location, normal, index = obj.ray_cast(ray_origin, ray_end - ray_origin)

        if success and index != -1:
            mag = (obj.matrix_world * (location - ray_origin)).magnitude
            if mag < min_[0]:
                min_ = (mag, obj, location)
                min_normal = obj.matrix_world.to_3x3().normalized() * normal.copy()
                min_location = obj.matrix_world * location.copy()

    return min_normal, min_location


# Taken from https://github.com/soravux/skylibs/blob/master/envmap/projections.py
def latlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def skylatlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v / 2

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def getEnvmapDirections(envmap_size, envmap_type):
    """envmap_size = (rows, columns)
    envmap_type in ["latlong", "skylatlong"]"""
    cols = np.linspace(0, 1, envmap_size[1]*2 + 1)
    rows = np.linspace(0, 1, envmap_size[0]*2 + 1)

    cols = cols[1::2]
    rows = rows[1::2]

    u, v = [d.astype('float32') for d in np.meshgrid(cols, rows)]
    if envmap_type.lower() == "latlong":
        x, y, z, valid = latlong2world(u, v)
    elif envmap_type.lower() == "skylatlong":
        x, y, z, valid = skylatlong2world(u, v)
    else:
        raise Exception("Unknown format: {}. Should be either \"latlong\" or "
                        "\"skylatlong\".".format(envmap_type))

    return np.asarray([x, y, z]).reshape((3,-1)).T


class GenerateTransportMatrix(bpy.types.Operator):
    bl_idname = "export.generate_transport_matrix"
    bl_label = "Generate Transport Matrix"
    bl_options = {'REGISTER'}

    filepath = bpy.props.StringProperty(default="scene.msgpack", subtype="FILE_PATH")
    envmap_type = bpy.props.EnumProperty(name="Environment Map Type",
                                         items=[("SKYLATLONG", "skylatlong", "width should be 4x the height", "", 1),
                                                ("LATLONG", "latlong", "width should be 2x the height", "", 2)])
    envmap_height = bpy.props.IntProperty(name="Environment Map Height (px)", default=64)
    only_surface_normals = bpy.props.BoolProperty(name="Output only surface normals", default=False)

    def execute(self, context):

        cam = bpy.data.objects['Camera']

        envmap_size = (self.envmap_height, 2*self.envmap_height if self.envmap_type.lower() == "latlong" else 4*self.envmap_height)
        envmap_coords = cam.data.clip_end * getEnvmapDirections(envmap_size, self.envmap_type)
        envmap_coords_blender = envmap_coords[:,[0,2,1]].copy()
        envmap_coords_blender[:,1] = -envmap_coords_blender[:,1]

        # Get the render resolution
        resp = bpy.data.scenes["Scene"].render.resolution_percentage / 100.
        resx = int(bpy.data.scenes["Scene"].render.resolution_x * resp)
        resy = int(bpy.data.scenes["Scene"].render.resolution_y * resp)

        # Get the camera viewport corner 3D coordinates
        frame = cam.data.view_frame(bpy.context.scene)
        tr, br, bl, tl = [cam.matrix_world * corner for corner in frame]
        x = br - bl
        dx =  x.normalized()
        y = tl - bl
        dy = y.normalized()

        pixels = []
        whole_normals = []
        # TODO: Shouldn't we use the center of the pixel instead of the corner?
        for s in range(resy):
            py = tl - s * y.length / float(resy) * dy
            normals_row = []
            for b in range(resx):
                ray_direction = py + b * x.length / float(resx) * dx
                #ray_target = (p - cam.location) * cam.data.clip_end + cam.location
                #hit, n, f = obj_raycast(obj, ray_origin, ray_target)
                normal, location = getClosestIntersection(cam.data.clip_end, cam.location, ray_direction)

                # This coordinates system converts from blender's z=up to skylibs y=up
                normals_row.append([normal[0], normal[2], -normal[1]] if normal else [0, 0, 0])

                if self.only_surface_normals:
                    continue

                if normal:
                    # For every envmap pixel
                    normal_np = np.array([normal[0], normal[2], -normal[1]])
                    intensity = envmap_coords.dot(normal_np)
                    # TODO: Add albedo here

                    # Handle occlusions (single bounce)
                    for idx in range(envmap_coords.shape[0]):
                        target_vec = Vector(envmap_coords_blender[idx,:])
                        # Check for occlusions. The 1e-3 is just to be sure the
                        # ray casting does not start right from the surface and
                        # find itself as occlusion...
                        normal_occ, _ = getClosestIntersection(cam.data.clip_end, location + (1/cam.data.clip_end)*1e-3*target_vec, target_vec)

                        if normal_occ:
                            intensity[idx] = 0

                    intensity[intensity < 0] = 0
                else:
                    intensity = np.zeros(envmap_coords.shape[0])

                #row.append(intensity)
                pixels.append(intensity)
                # This line is for outputting normals
                #row.append([p for p in normal] if normal else [0, 0, 0])
            whole_normals.append(normals_row)
            #row = []
            print("{}/{}".format(s, resy))
        pixels = np.asarray(pixels)
        whole_normals = np.asarray(whole_normals)

        print("Saving to ", self.filepath)
        with open(self.filepath, "wb") as fhdl:
            np.savez(fhdl, pixels, whole_normals, np.array([resy, resx]))

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def menu_export(self, context):
    #import os
    #default_path = os.path.splitext(bpy.data.filepath)[0] + ".npz"
    self.layout.operator(GenerateTransportMatrix.bl_idname, text="Transport Matrix (.npz)")#.filepath = default_path


def register():
    bpy.types.INFO_MT_file_export.append(menu_export)
    bpy.utils.register_class(GenerateTransportMatrix)


def unregister():
    bpy.types.INFO_MT_file_export.remove(menu_export)
    bpy.utils.unregister_class(GenerateTransportMatrix)


if __name__ == "__main__":
    register()
    #a = GenerateTransportMatrix()
    #a.execute(bpy.context)
