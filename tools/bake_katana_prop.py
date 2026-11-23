#!/usr/bin/env python3
"""Bake the standalone katana GLB into a hand-prop GLB.

Bakes the node's Z-up rotation into the vertices, recenters on the grip
point (middle of the wrapped handle), turns the blade to +Y, and scales
to character units (the girl's GLB is 1.0 tall; a katana is ~0.62 of her
height). Textures/materials/UVs are untouched.

Usage: bake_katana_prop.py <src.glb> <dst.glb>
"""
import json, struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
data = open(src, 'rb').read()
assert data[:4] == b'glTF'
jlen, _ = struct.unpack_from('<II', data, 12)
gltf = json.loads(data[20:20 + jlen])
boff = 20 + jlen
blen, _ = struct.unpack_from('<II', data, boff)
bin_data = bytearray(data[boff + 8: boff + 8 + blen])

def acc_view(ai):
    a = gltf['accessors'][ai]; bv = gltf['bufferViews'][a['bufferView']]
    return a, (bv.get('byteOffset', 0) + a.get('byteOffset', 0))

def read_f32(ai, ncomp):
    a, off = acc_view(ai)
    return np.frombuffer(bytes(bin_data), dtype=np.float32,
                         count=a['count'] * ncomp, offset=off).reshape(a['count'], ncomp).copy()

def write_f32(ai, arr):
    a, off = acc_view(ai)
    bin_data[off:off + arr.size * 4] = arr.astype(np.float32).tobytes()

prim = gltf['meshes'][0]['primitives'][0]
pos = read_f32(prim['attributes']['POSITION'], 3)
nor = read_f32(prim['attributes']['NORMAL'], 3)

# node rotation: quat (x,y,z,w) = (0.7071, 0, 0, 0.7071) -> +90 deg about X
q = gltf['nodes'][0]['rotation']
x, y, z, w = q
R = np.array([
    [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
    [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
    [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
])
pos = pos @ R.T
nor = nor @ R.T

# baked frame: length along X (handle -0.33..-0.09, guard ~-0.09, tip 0.869),
# blade width along Y (0..0.126), thickness along Z
grip = np.array([-0.20, 0.028, 0.0])
pos = pos - grip
# rotate about Z by +90 deg: blade +X -> +Y
Rz = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
pos = pos @ Rz.T
nor = nor @ Rz.T
scale = 0.62 / 1.199
pos = pos * scale
nor = nor / np.linalg.norm(nor, axis=1, keepdims=True)

write_f32(prim['attributes']['POSITION'], pos)
write_f32(prim['attributes']['NORMAL'], nor)
pa = gltf['accessors'][prim['attributes']['POSITION']]
pa['min'] = pos.min(0).tolist(); pa['max'] = pos.max(0).tolist()
gltf['nodes'][0].pop('rotation', None)
gltf['buffers'][0]['byteLength'] = len(bin_data)

js = json.dumps(gltf, separators=(',', ':')).encode()
while len(js) % 4: js += b' '
out = bytearray()
out += struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_data))
out += struct.pack('<II', len(js), 0x4E4F534A) + js
out += struct.pack('<II', len(bin_data), 0x004E4942) + bin_data
open(dst, 'wb').write(out)
print(f'wrote {dst}: {len(out)} bytes; bounds min {pos.min(0).round(3)} max {pos.max(0).round(3)}')
