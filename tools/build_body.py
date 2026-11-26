#!/usr/bin/env python3
"""Build a body-only version of the samurai girl: strip the clothing shells
(hair, sleeves, wrap top, hakama, belt, her own sandals) from the swordless
model, then bake the idle pose (arms down) into a static mesh.

Usage: python3 build_body.py <in.glb> <out.glb>
"""
import io
import json
import struct
import sys

import numpy as np
from PIL import Image


# --- minimal GLB helpers (self-contained) ---
import struct as _struct
from pathlib import Path as _Path


class ss:
    @staticmethod
    def load(path):
        data = _Path(path).read_bytes()
        assert data[:4] == b'glTF'
        jlen = _struct.unpack('<I', data[12:16])[0]
        gltf = json.loads(data[20:20 + jlen])
        boff = 20 + jlen
        blen = _struct.unpack('<I', data[boff:boff + 4])[0]
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

    @staticmethod
    def bind_positions(gltf, binb):
        mesh_node = next(i for i, n in enumerate(gltf['nodes']) if 'skin' in n and 'mesh' in n)
        skin = gltf['skins'][gltf['nodes'][mesh_node]['skin']]
        prim = gltf['meshes'][gltf['nodes'][mesh_node]['mesh']]['primitives'][0]
        pos = ss.acc_np(gltf, binb, prim['attributes']['POSITION']).astype(np.float64)
        joints = ss.acc_np(gltf, binb, prim['attributes']['JOINTS_0']).astype(int)
        weights = ss.acc_np(gltf, binb, prim['attributes']['WEIGHTS_0']).astype(np.float64)
        ibm = ss.acc_np(gltf, binb, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
        world = ss.node_worlds(gltf)
        mats = np.stack([world(jn) @ ibm[j] for j, jn in enumerate(skin['joints'])])
        hom = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
        bind = np.zeros((len(pos), 3))
        for k in range(4):
            bind += np.einsum('nij,nj->ni', mats[joints[:, k]], hom)[:, :3] * weights[:, k:k + 1]
        return prim, bind




def build_body(src, out_path):
    gltf, binb = ss.load(src)
    prim, bind = ss.bind_positions(gltf, binb)
    idx = ss.acc_np(gltf, binb, prim['indices']).astype(np.int64)
    tris = idx.reshape(-1, 3)
    shells = ss.shells_of(tris, len(bind))
    joints = ss.acc_np(gltf, binb, prim['attributes']['JOINTS_0']).astype(int)
    weights = ss.acc_np(gltf, binb, prim['attributes']['WEIGHTS_0']).astype(np.float64)
    uv = ss.acc_np(gltf, binb, prim['attributes']['TEXCOORD_0']).astype(np.float64)
    skin = gltf['skins'][0]
    names = [gltf['nodes'][j].get('name', '') for j in skin['joints']]

    # Body test: a vertex belongs to the body if it lies close to the skeleton
    # core (spine / arm / leg polylines, head sphere) — clothing surfaces sit
    # measurably farther out. A shell is kept when >=45% of its verts qualify.
    world = ss.node_worlds(gltf)

    def jpos(suffix):
        j = next(i for i, n in enumerate(names) if n.endswith(':' + suffix))
        return world(skin['joints'][j])[:3, 3]

    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)

    def poly_dist(p, pts):
        return np.min([seg_dist(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1)], axis=0)

    spine = np.array([jpos(s) for s in ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck')])
    d_spine = poly_dist(bind, spine)
    body = d_spine < 0.10
    body |= np.linalg.norm(bind - jpos('Head'), axis=1) < 0.085
    d_arm = np.full(len(bind), 1e9)
    for side in ('Left', 'Right'):
        arm = np.array([jpos(f'{side}Shoulder'), jpos(f'{side}Arm'),
                        jpos(f'{side}ForeArm'), jpos(f'{side}Hand')])
        leg = np.array([jpos(f'{side}UpLeg'), jpos(f'{side}Leg'), jpos(f'{side}Foot')])
        d_arm = np.minimum(d_arm, poly_dist(bind, arm))
        body |= poly_dist(bind, leg) < 0.068
    body |= d_arm < 0.045
    # Torso band: the wrap top covers y 0.58-0.845 — strip that whole band
    # except the neck column and the shoulder/arm junctions (parts cover it).
    band = (bind[:, 1] >= 0.58) & (bind[:, 1] <= 0.845) & (d_spine < 0.13)
    band &= ~((bind[:, 1] > 0.80) & (d_spine < 0.05))
    band &= ~(d_arm < 0.045)
    body &= ~band
    hand_js = [i for i, n in enumerate(names) if 'Hand' in n]
    whand = np.zeros(len(bind))
    for k4 in range(4):
        for j in hand_js:
            whand += weights[:, k4] * (joints[:, k4] == j)
    body |= whand > 0.4
    body &= ~((bind[:, 1] < 0.09) & (poly_dist(bind, np.array([jpos('LeftUpLeg'), jpos('LeftLeg'), jpos('LeftFoot')])) >= 0.04)
              & (poly_dist(bind, np.array([jpos('RightUpLeg'), jpos('RightLeg'), jpos('RightFoot')])) >= 0.04))

    # mean texture luminance per shell, for the dark-remnant rules below
    mat = gltf['materials'][prim['material']]
    ti = mat['pbrMetallicRoughness']['baseColorTexture']['index']
    img = gltf['images'][gltf['textures'][ti]['source']]
    bv = gltf['bufferViews'][img['bufferView']]
    tex = Image.open(io.BytesIO(bytes(binb[bv.get('byteOffset', 0):
                                      bv.get('byteOffset', 0) + bv['byteLength']]))).convert('RGB')
    ta = np.asarray(tex).astype(np.float64) / 255.0

    def lum_stats(verts):
        u = np.clip(uv[verts, 0], 0, 0.999)
        v = np.clip(uv[verts, 1], 0, 0.999)
        px = ta[(v * ta.shape[0]).astype(int), (u * ta.shape[1]).astype(int)]
        lums = px @ [0.299, 0.587, 0.114]
        return float(lums.mean()), float(np.percentile(lums, 25))

    def mean_lum(verts):
        return lum_stats(verts)[0]

    drop = set()
    for s in shells:
        v = s['verts']
        cen = bind[v].mean(0)
        lum = mean_lum(v)
        if cen[1] > 0.84 and lum < 0.35:
            drop.add(s['id'])                       # hair remnants on the head
            continue
        if 0.52 <= cen[1] <= 0.62 and lum < 0.30:
            drop.add(s['id'])                       # belt ring remnants
            continue
        if 0.60 <= cen[1] <= 0.86 and abs(cen[0]) < 0.17 and cen[2] > -0.02 and lum < 0.40:
            drop.add(s['id'])                       # dark front-torso cloth remnants
            continue
        frac = float(body[v].mean())
        if frac < 0.45:
            drop.add(s['id'])
    ntris = sum(len(shells[i]['tris']) for i in drop)
    print(f'stripping {len(drop)} shells, {ntris} tris; keeping {len(tris) - ntris} tris')

    drop_tris = set()
    for d in drop:
        drop_tris.update(shells[d]['tris'])
    keep_mask = np.array([i not in drop_tris for i in range(len(tris))])
    tris2 = tris[keep_mask]

    # ---- bake the idle pose ----
    anim = next(a for a in gltf['animations'] if a.get('name') == 'idle')
    posed = {}
    for ch in anim['channels']:
        smp = anim['samplers'][ch['sampler']]
        times = ss.acc_np(gltf, binb, smp['input']).astype(np.float64)
        vals = ss.acc_np(gltf, binb, smp['output']).astype(np.float64)
        t = 0.0
        if t <= times[0]:
            val = vals[0]
        elif t >= times[-1]:
            val = vals[-1]
        else:
            k = int(np.searchsorted(times, t)) - 1
            f = (t - times[k]) / (times[k + 1] - times[k])
            a, b = vals[k], vals[k + 1]
            if ch['target']['path'] == 'rotation':
                d = float(np.dot(a, b))
                if d < 0:
                    b = -b
                    d = -d
                if d > 0.9995:
                    val = a + f * (b - a)
                else:
                    th = np.arccos(np.clip(d, -1, 1))
                    val = (np.sin((1 - f) * th) * a + np.sin(f * th) * b) / np.sin(th)
                val = val / np.linalg.norm(val)
            else:
                val = a + f * (b - a)
        posed.setdefault(ch['target']['node'], {})[ch['target']['path']] = val

    def node_local(i):
        n = gltf['nodes'][i]
        if 'matrix' in n and i not in posed:
            return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
        if 'matrix' in n:
            m = np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
            t = m[:3, 3]
            sc = np.linalg.norm(m[:3, :3], axis=0)
            r = None
            s = sc
            Rn = m[:3, :3] / sc
            # rotation matrix -> quaternion
            tr = np.trace(Rn)
            if tr > 0:
                w = np.sqrt(1 + tr) / 2
                r = [(Rn[2, 1] - Rn[1, 2]) / (4 * w), (Rn[0, 2] - Rn[2, 0]) / (4 * w),
                     (Rn[1, 0] - Rn[0, 1]) / (4 * w), w]
            else:
                k = int(np.argmax(np.diag(Rn)))
                if k == 0:
                    x = np.sqrt(1 + Rn[0, 0] - Rn[1, 1] - Rn[2, 2]) / 2
                    r = [x, (Rn[0, 1] + Rn[1, 0]) / (4 * x), (Rn[0, 2] + Rn[2, 0]) / (4 * x),
                         (Rn[2, 1] - Rn[1, 2]) / (4 * x)]
                elif k == 1:
                    y = np.sqrt(1 + Rn[1, 1] - Rn[0, 0] - Rn[2, 2]) / 2
                    r = [(Rn[0, 1] + Rn[1, 0]) / (4 * y), y, (Rn[1, 2] + Rn[2, 1]) / (4 * y),
                         (Rn[0, 2] - Rn[2, 0]) / (4 * y)]
                else:
                    z = np.sqrt(1 + Rn[2, 2] - Rn[0, 0] - Rn[1, 1]) / 2
                    r = [(Rn[0, 2] + Rn[2, 0]) / (4 * z), (Rn[1, 2] + Rn[2, 1]) / (4 * z), z,
                         (Rn[1, 0] - Rn[0, 1]) / (4 * z)]
        else:
            t = n.get('translation', [0, 0, 0])
            r = n.get('rotation', [0, 0, 0, 1])
            s = n.get('scale', [1, 1, 1])
        if i in posed:
            t = posed[i].get('translation', t)
            r = posed[i].get('rotation', r)
            s = posed[i].get('scale', s)
        x, y, z, w = r
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m = np.eye(4)
        m[:3, :3] = R * np.array(s)
        m[:3, 3] = t
        return m

    parents = {}
    for i, n in enumerate(gltf['nodes']):
        for c in n.get('children', []):
            parents[c] = i
    cache = {}

    def world(i):
        if i in cache:
            return cache[i]
        m = node_local(i)
        if i in parents:
            m = world(parents[i]) @ m
        cache[i] = m
        return m

    ibm = ss.acc_np(gltf, binb, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
    mats = np.stack([world(jn) @ ibm[j] for j, jn in enumerate(skin['joints'])])
    pos = ss.acc_np(gltf, binb, prim['attributes']['POSITION']).astype(np.float64)
    nor = ss.acc_np(gltf, binb, prim['attributes']['NORMAL']).astype(np.float64)
    hom = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
    newpos = np.zeros((len(pos), 3))
    newnor = np.zeros((len(pos), 3))
    for k in range(4):
        jk = joints[:, k]
        wk = weights[:, k:k + 1]
        newpos += np.einsum('nij,nj->ni', mats[jk], hom)[:, :3] * wk
        newnor += np.einsum('nij,nj->ni', mats[jk], np.concatenate([nor, np.zeros((len(nor), 1))], axis=1))[:, :3] * wk
    nn = np.linalg.norm(newnor, axis=1, keepdims=True)
    newnor = newnor / np.maximum(nn, 1e-9)

    # ---- compact + write a static GLB (same texture/material) ----
    used = np.unique(tris2.ravel())
    remap = {v: i for i, v in enumerate(used)}
    tris3 = np.array([[remap[v] for v in t] for t in tris2], dtype=np.uint32)
    P = newpos[used].astype(np.float32)
    N = newnor[used].astype(np.float32)
    UV = uv[used].astype(np.float32)

    buf = bytearray(binb)  # keep original buffer (texture etc.), append geometry
    views = list(gltf['bufferViews'])
    accessors = list(gltf['accessors'])

    def add_view(data, target):
        while len(buf) % 4:
            buf.append(0)
        off = len(buf)
        buf.extend(data)
        views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(data), 'target': target})
        return len(views) - 1

    def add_acc(view, ctype, count, typ, arr=None):
        a = {'bufferView': view, 'componentType': ctype, 'count': count, 'type': typ}
        if arr is not None:
            a['min'] = arr.min(0).tolist()
            a['max'] = arr.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    pa = add_acc(add_view(P.tobytes(), 34962), 5126, len(P), 'VEC3', P)
    na = add_acc(add_view(N.tobytes(), 34962), 5126, len(N), 'VEC3')
    ua = add_acc(add_view(UV.tobytes(), 34962), 5126, len(UV), 'VEC2')
    ia = add_acc(add_view(tris3.tobytes(), 34963), 5125, len(tris3) * 3, 'SCALAR')
    out = {
        'asset': gltf['asset'],
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0, 'name': 'body'}],
        'meshes': [{'name': 'body', 'primitives': [
            {'attributes': {'POSITION': pa, 'NORMAL': na, 'TEXCOORD_0': ua},
             'indices': ia, 'material': prim['material']}]}],
        'materials': gltf['materials'],
        'textures': gltf['textures'],
        'images': gltf['images'],
        'samplers': gltf.get('samplers', []),
        'accessors': accessors,
        'bufferViews': views,
        'buffers': [{'byteLength': len(buf)}],
    }
    js = json.dumps(out, separators=(',', ':')).encode()
    while len(js) % 4:
        js += b' '
    while len(buf) % 4:
        buf.append(0)
    with open(out_path, 'wb') as f:
        f.write(b'glTF' + struct.pack('<II', 2, 20 + len(js) + 8 + len(buf)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(buf), 0x004E4942) + bytes(buf))
    print('wrote', out_path, 'verts', len(P), 'tris', len(tris3))


if __name__ == '__main__':
    build_body(sys.argv[1], sys.argv[2])
