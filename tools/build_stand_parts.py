#!/usr/bin/env python3
"""Assemble the standalone part models into one standing figure (stand-parts.glb).

Each part was generated on its own (flat-lay / ghost-mannequin), unskinned, in
an arbitrary frame. This bakes every part's node transform, applies a placement
(rotation + uniform scale + anchor translation) measured against the samurai
girl's bind pose, and merges everything into a single multi-primitive GLB with
each part keeping its own texture. Sleeves are split into their two panels by
connected component and placed at her sides.

Usage: python3 build_stand_parts.py [out.glb] [texmax]
"""
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = str(Path(__file__).resolve().parent.parent / 'models')


# --- minimal GLB reading helpers (self-contained) ---
class ss:
    @staticmethod
    def load(path):
        data = Path(path).read_bytes()
        assert data[:4] == b'glTF'
        jlen = struct.unpack('<I', data[12:16])[0]
        gltf = json.loads(data[20:20 + jlen])
        boff = 20 + jlen
        blen = struct.unpack('<I', data[boff:boff + 4])[0]
        return gltf, bytearray(data[boff + 8: boff + 8 + blen])

    @staticmethod
    def acc_np(gltf, binb, i):
        a = gltf['accessors'][i]
        bv = gltf['bufferViews'][a['bufferView']]
        off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        dt = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}[a['componentType']]
        arr = np.frombuffer(bytes(binb[off: off + a['count'] * ncomp * np.dtype(dt).itemsize]),
                            dtype=dt, count=a['count'] * ncomp)
        return arr.reshape(a['count'], ncomp) if ncomp > 1 else arr

    @staticmethod
    def node_worlds(gltf):
        cache = {}

        def world(idx):
            if idx in cache:
                return cache[idx]
            n = gltf['nodes'][idx]
            if 'matrix' in n:
                m = np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
            else:
                t = n.get('translation', [0, 0, 0])
                r = n.get('rotation', [0, 0, 0, 1])
                s = n.get('scale', [1, 1, 1])
                x, y, z, w = r
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
                             dtype=np.float64)
                m = np.eye(4)
                m[:3, :3] = R * np.array(s)
                m[:3, 3] = t
            parent = next((i for i, nn in enumerate(gltf['nodes']) if idx in nn.get('children', [])), None)
            if parent is not None:
                m = world(parent) @ m
            cache[idx] = m
            return m
        return world

    @staticmethod
    def shells_of(tris, nverts):
        parent = np.arange(nverts)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, b, c in tris:
            ra, rb, rc = find(a), find(b), find(c)
            if ra != rb:
                parent[ra] = rb
            if rb != rc:
                parent[rb] = rc
        groups = {}
        for ti, t in enumerate(tris):
            groups.setdefault(find(t[0]), []).append(ti)
        out = []
        for root, tlist in groups.items():
            vset = np.unique(tris[tlist].ravel())
            out.append({'min_vert': int(vset.min()), 'tris': tlist, 'verts': vset})
        out.sort(key=lambda s: s['min_vert'])
        for i, s in enumerate(out):
            s['id'] = i
        return out


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_part(name):
    """Return list of prim dicts with baked world positions/normals."""
    gltf, binb = ss.load(f'{MODELS}/{name}.glb')
    world = ss.node_worlds(gltf)
    prims = []
    for ni, node in enumerate(gltf['nodes']):
        if 'mesh' not in node:
            continue
        M = world(ni)
        R = M[:3, :3]
        for prim in gltf['meshes'][node['mesh']]['primitives']:
            pos = ss.acc_np(gltf, binb, prim['attributes']['POSITION']).astype(np.float64)
            nor = ss.acc_np(gltf, binb, prim['attributes']['NORMAL']).astype(np.float64)
            uv = ss.acc_np(gltf, binb, prim['attributes']['TEXCOORD_0']).astype(np.float32)
            idx = ss.acc_np(gltf, binb, prim['indices']).astype(np.int64)
            prims.append({
                'pos': pos @ R.T + M[:3, 3],
                'nor': nor @ R.T,
                'uv': uv, 'idx': idx,
                'material': prim.get('material'),
                'gltf': gltf, 'bin': binb,
            })
    return prims


def place(prims, R, scale, anchor, target):
    """Rotate (about bounds center), scale, then move `anchor` point to target.
    anchor: (fx, fy, fz) fractions of the bounds box, e.g. (0.5, 1, 0.5) = top center."""
    pos = np.vstack([p['pos'] for p in prims])
    cen = (pos.min(0) + pos.max(0)) / 2
    for p in prims:
        p['pos'] = (p['pos'] - cen) @ R.T
        p['nor'] = p['nor'] @ R.T
    pos = np.vstack([p['pos'] for p in prims])
    lo, hi = pos.min(0), pos.max(0)
    for p in prims:
        p['pos'] = p['pos'] * scale
    a = (lo + (hi - lo) * np.array(anchor)) * scale
    for p in prims:
        p['pos'] = p['pos'] + (np.array(target) - a)
    return prims


def split_shells(prim, keep):
    """Split a prim's triangles by connected component; keep = predicate(shell)."""
    tris = prim['idx'].reshape(-1, 3)
    shells = ss.shells_of(tris, len(prim['pos']))
    out = []
    for s in shells:
        if not keep(s):
            continue
        vmap = {v: i for i, v in enumerate(s['verts'])}
        sub = dict(prim)
        sub['pos'] = prim['pos'][s['verts']]
        sub['nor'] = prim['nor'][s['verts']]
        sub['uv'] = prim['uv'][s['verts']]
        sub['idx'] = np.array([[vmap[v] for v in t] for t in tris[s['tris']]], dtype=np.int64)
        out.append(sub)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else f'{MODELS}/stand-parts.glb'
    texmax = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    jobs = []  # (prims, name)

    hair = load_part('hair')
    jobs += [(p, 'hair') for p in place(hair, rot('z', -90), 0.58, (0.5, 1, 0.5), (-0.02, 1.00, -0.125))]

    hakama = load_part('hakama')
    jobs += [(p, 'hakama') for p in place(hakama, np.eye(3), 0.58, (0.5, 1, 0.5), (0, 0.62, -0.01))]

    top = load_part('top')
    jobs += [(p, 'top') for p in place(top, np.eye(3), 0.28, (0.5, 0.5, 0.5), (0, 0.73, -0.005))]

    sleeves = load_part('sleeves')
    panels = split_shells(sleeves[0], lambda s: len(s['tris']) >= 40)
    panels.sort(key=lambda p: p['pos'][:, 0].mean())
    if len(panels) == 2:
        left, right = panels
        jobs += [(p, 'sleeveL') for p in place([left], np.eye(3), 0.60, (0.5, 1, 0.5), (-0.315, 0.82, -0.01))]
        jobs += [(p, 'sleeveR') for p in place([right], np.eye(3), 0.60, (0.5, 1, 0.5), (0.315, 0.82, -0.01))]
    else:
        print(f'WARN: sleeves split into {len(panels)} panels, placing whole')
        jobs += [(p, 'sleeves') for p in place(sleeves, np.eye(3), 0.55, (0.5, 1, 0.5), (0, 0.84, -0.10))]

    sandals = load_part('sandals')
    jobs += [(p, 'sandals') for p in place(sandals, np.eye(3), 0.21, (0.5, 0, 0.5), (0, 0.0, 0.03))]

    katana = load_part('katana')
    jobs += [(p, 'katana') for p in place(katana, rot('z', -38) @ rot('x', -12), 1.0, (0.5, 0, 0.5), (0.02, 0.52, 0.15))]

    # ---- merge ----
    buf = bytearray()
    views, accessors, images, textures, materials, prims_out = [], [], [], [], [], []

    def add_view(data, target=None):
        while len(buf) % 4:
            buf.append(0)
        off = len(buf)
        buf.extend(data)
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target:
            v['target'] = target
        views.append(v)
        return len(views) - 1

    def add_accessor(view, ctype, count, typ, arr=None):
        a = {'bufferView': view, 'componentType': ctype, 'count': count, 'type': typ}
        if arr is not None:
            a['min'] = arr.min(0).tolist()
            a['max'] = arr.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    mat_remap = {}
    for prim, name in jobs:
        g, b = prim['gltf'], prim['bin']
        key = (id(g), prim['material'])
        if key not in mat_remap:
            src = g['materials'][prim['material']]
            mat = json.loads(json.dumps(src))
            pbr = mat.get('pbrMetallicRoughness', {})
            if 'baseColorTexture' in pbr:
                ti = pbr['baseColorTexture']['index']
                tex = g['textures'][ti]
                img = g['images'][tex['source']]
                bv = g['bufferViews'][img['bufferView']]
                raw = bytes(b[bv.get('byteOffset', 0): bv.get('byteOffset', 0) + bv['byteLength']])
                im = Image.open(io.BytesIO(raw)).convert('RGB')
                if max(im.size) > texmax:
                    im.thumbnail((texmax, texmax), Image.LANCZOS)
                out = io.BytesIO()
                im.save(out, 'JPEG', quality=82)
                iv = add_view(out.getvalue())
                images.append({'bufferView': iv, 'mimeType': 'image/jpeg'})
                textures.append({'source': len(images) - 1, 'sampler': 0})
                pbr['baseColorTexture'] = {'index': len(textures) - 1}
            mat_remap[key] = len(materials)
            materials.append(mat)
        pos = prim['pos'].astype(np.float32)
        nor = prim['nor'].astype(np.float32)
        n = np.linalg.norm(nor, axis=1, keepdims=True)
        nor = (nor / np.maximum(n, 1e-9)).astype(np.float32)
        idx = prim['idx'].astype(np.uint32)
        pa = add_accessor(add_view(pos.tobytes(), 34962), 5126, len(pos), 'VEC3', pos)
        na = add_accessor(add_view(nor.tobytes(), 34962), 5126, len(nor), 'VEC3')
        ua = add_accessor(add_view(prim['uv'].tobytes(), 34962), 5126, len(prim['uv']), 'VEC2')
        ia = add_accessor(add_view(idx.tobytes(), 34963), 5125, len(idx), 'SCALAR')
        prims_out.append({'attributes': {'POSITION': pa, 'NORMAL': na, 'TEXCOORD_0': ua},
                          'indices': ia, 'material': mat_remap[key], 'name': name})
        print(f'{name}: {len(pos)} verts, {len(idx)//3} tris')

    gltf = {
        'asset': {'version': '2.0', 'generator': 'dbox build_stand_parts'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0, 'name': 'stand-parts'}],
        'meshes': [{'name': 'stand-parts', 'primitives': prims_out}],
        'materials': materials,
        'textures': textures,
        'images': images,
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
        'accessors': accessors,
        'bufferViews': views,
        'buffers': [{'byteLength': len(buf)}],
    }
    js = json.dumps(gltf, separators=(',', ':')).encode()
    while len(js) % 4:
        js += b' '
    while len(buf) % 4:
        buf.append(0)
    with open(out_path, 'wb') as f:
        f.write(b'glTF' + struct.pack('<II', 2, 20 + len(js) + 8 + len(buf)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(buf), 0x004E4942) + bytes(buf))
    print('wrote', out_path, len(buf) + len(js) + 28, 'bytes')


if __name__ == '__main__':
    main()
