#!/usr/bin/env python3
"""Build models/samurai-cutout.glb: the samurai girl as a flat cutout standee.

Reads assets/samurai-cutout.png (transparent PNG, background removed from the
source art) and wraps it in a GLB: one double-sided plane, alphaMode MASK so
the renderer discards the transparent pixels, plus a gentle 'sway' clip
(small Y-rock + bob) so she feels alive in the box. Placeholder exhibit until
the real rigged model replaces her.
"""
import json, struct
from PIL import Image

ROOT = __file__.rsplit('/tools/', 1)[0]
PNG_PATH = ROOT + '/assets/samurai-cutout.png'
OUT_PATH = ROOT + '/models/samurai-cutout.glb'

img = Image.open(PNG_PATH)
iw, ih = img.size
H = 2.0
W = H * iw / ih
print(f'texture {iw}x{ih} -> plane {W:.3f} x {H:.3f}')

png = open(PNG_PATH, 'rb').read()

# plane: base at y=0, facing +Z. v=0 at the top edge (WebGL uploads the
# canvas top row first and this viewer does not flip Y).
pos = [(-W/2, 0, 0), (W/2, 0, 0), (W/2, H, 0), (-W/2, H, 0)]
nor = [(0, 0, 1)] * 4
col = [(1, 1, 1)] * 4
uv  = [(0, 1), (1, 1), (1, 0), (0, 0)]
idx = [0, 1, 2, 0, 2, 3]

import math
def quat_y(a):
    return (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
times = [0.0, 1.6, 3.2]
rots = [quat_y(-0.05), quat_y(0.05), quat_y(-0.05)]
bobs = [(0.0, 0.0, 0.0), (0.0, 0.015, 0.0), (0.0, 0.0, 0.0)]

chunks = []   # (bytes, target) target: 34962 ARRAY_BUFFER, 34963 ELEMENT_ARRAY_BUFFER, None=image
views, accessors = [], []

def add(data, target, ctype, atype, count, mn=None, mx=None):
    off = sum(len(c[0]) + (4 - len(c[0]) % 4) % 4 for c in chunks)
    chunks.append((data, target))
    views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(data),
                  **({'target': target} if target else {})})
    acc = {'bufferView': len(views) - 1, 'componentType': ctype, 'count': count, 'type': atype}
    if mn is not None: acc['min'] = mn
    if mx is not None: acc['max'] = mx
    accessors.append(acc)
    return len(accessors) - 1

def f32(vals): return struct.pack('<%df' % len(vals), *vals)
def u16(vals): return struct.pack('<%dH' % len(vals), *vals)

img_view = len(views)
views.append({'buffer': 0, 'byteOffset': 0, 'byteLength': len(png)})
chunks.append((png, None))

a_pos = add(f32([c for v in pos for c in v]), 34962, 5126, 'VEC3', 4,
            mn=[-W/2, 0, 0], mx=[W/2, H, 0])
a_nor = add(f32([c for v in nor for c in v]), 34962, 5126, 'VEC3', 4)
a_col = add(f32([c for v in col for c in v]), 34962, 5126, 'VEC3', 4)
a_uv  = add(f32([c for v in uv for c in v]), 34962, 5126, 'VEC2', 4)
a_idx = add(u16(idx), 34963, 5123, 'SCALAR', 6)
a_t   = add(f32(times), None, 5126, 'SCALAR', 3, mn=[0.0], mx=[3.2])
a_rot = add(f32([c for q in rots for c in q]), None, 5126, 'VEC4', 3)
a_bob = add(f32([c for v in bobs for c in v]), None, 5126, 'VEC3', 3)

gltf = {
    'asset': {'version': '2.0', 'generator': 'dbox make_cutout_glb'},
    'scene': 0,
    'scenes': [{'nodes': [0]}],
    'nodes': [{'mesh': 0, 'name': 'samurai-cutout'}],
    'meshes': [{'name': 'cutout-plane', 'primitives': [{
        'attributes': {'POSITION': a_pos, 'NORMAL': a_nor, 'COLOR_0': a_col, 'TEXCOORD_0': a_uv},
        'indices': a_idx, 'material': 0}]}],
    'materials': [{'name': 'cutout',
                   'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                            'metallicFactor': 0.0, 'roughnessFactor': 1.0},
                   'alphaMode': 'MASK', 'alphaCutoff': 0.45, 'doubleSided': True}],
    'textures': [{'sampler': 0, 'source': 0}],
    'images': [{'bufferView': img_view, 'mimeType': 'image/png', 'name': 'samurai-cutout'}],
    'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}],
    'animations': [{'name': 'sway', 'channels': [
        {'sampler': 0, 'target': {'node': 0, 'path': 'rotation'}},
        {'sampler': 1, 'target': {'node': 0, 'path': 'translation'}}],
        'samplers': [
            {'input': a_t, 'output': a_rot, 'interpolation': 'LINEAR'},
            {'input': a_t, 'output': a_bob, 'interpolation': 'LINEAR'}]}],
    'bufferViews': views,
    'accessors': accessors,
}

bin_data = b''
for data, _ in chunks:
    bin_data += data
    bin_data += b'\x00' * ((4 - len(data) % 4) % 4)
gltf['buffers'] = [{'byteLength': len(bin_data)}]

json_bytes = json.dumps(gltf).encode()
json_bytes += b' ' * ((4 - len(json_bytes) % 4) % 4)
total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
glb = struct.pack('<III', 0x46546C67, 2, total)
glb += struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
glb += struct.pack('<II', len(bin_data), 0x004E4942) + bin_data
open(OUT_PATH, 'wb').write(glb)
print(f'wrote {OUT_PATH} ({len(glb)} bytes)')
