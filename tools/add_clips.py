#!/usr/bin/env python3
"""Inject hand-authored animation clips into the Tripo samurai-girl GLB.

Tripo Studio's free-plan GLB exports carry the Mixamo skin but drop the
retargeted animation clips, so this writes two simple clips (idle, slash)
directly into the file: absolute local rotations = bind * euler-offset,
matching how Dbox's renderer samples rotation channels.

Usage: add_clips.py <src.glb> <dst.glb>
"""
import json, math, struct, sys

# ---------- quat helpers (mirror Dbox's Q, [x,y,z,w], Hamilton) ----------
def qmul(a, b):
    return [a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
            a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
            a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
            a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]]

def qeuler(x, y, z):
    qx = [math.sin(x/2), 0, 0, math.cos(x/2)]
    qy = [0, math.sin(y/2), 0, math.cos(y/2)]
    qz = [0, 0, math.sin(z/2), math.cos(z/2)]
    return qmul(qmul(qx, qy), qz)

# ---------- matrix decompose (port of Dbox's decomposeTRS) ----------
def decompose(m):
    t = [m[12], m[13], m[14]]
    sx = math.sqrt(m[0]**2 + m[1]**2 + m[2]**2)
    sy = math.sqrt(m[4]**2 + m[5]**2 + m[6]**2)
    sz = math.sqrt(m[8]**2 + m[9]**2 + m[10]**2)
    det = (m[0]*(m[5]*m[10] - m[6]*m[9]) - m[4]*(m[1]*m[10] - m[2]*m[9])
           + m[8]*(m[1]*m[6] - m[2]*m[5]))
    if det < 0:
        sx = -sx
    r00, r10, r20 = m[0]/sx, m[1]/sx, m[2]/sx
    r01, r11, r21 = m[4]/sy, m[5]/sy, m[6]/sy
    r02, r12, r22 = m[8]/sz, m[9]/sz, m[10]/sz
    tr = r00 + r11 + r22
    if tr > 0:
        s4 = math.sqrt(tr + 1) * 2
        q = [(r21 - r12)/s4, (r02 - r20)/s4, (r10 - r01)/s4, 0.25*s4]
    elif r00 > r11 and r00 > r22:
        s4 = math.sqrt(1 + r00 - r11 - r22) * 2
        q = [0.25*s4, (r01 + r10)/s4, (r02 + r20)/s4, (r21 - r12)/s4]
    elif r11 > r22:
        s4 = math.sqrt(1 + r11 - r00 - r22) * 2
        q = [(r01 + r10)/s4, 0.25*s4, (r12 + r21)/s4, (r02 - r20)/s4]
    else:
        s4 = math.sqrt(1 + r22 - r00 - r11) * 2
        q = [(r02 + r20)/s4, (r12 + r21)/s4, 0.25*s4, (r10 - r01)/s4]
    return t, q, [sx, sy, sz]

# ---------- clip design: per joint, list of (time, euler-offset) or (time, trans-offset) ----------
def idle_keys():
    # 4 s breathing loop; values sampled from gentle sines, first == last
    ts = [0.0, 1.0, 2.0, 3.0, 4.0]
    def s(amp, phase=0.0):
        return [amp * math.sin(2 * math.pi * t / 4.0 + phase) for t in ts]
    return {
        'rot': {
            'mixamorig:Spine':  [(t, (v, 0, 0)) for t, v in zip(ts, s(0.035))],
            'mixamorig:Spine2': [(t, (v, 0, 0)) for t, v in zip(ts, s(0.022))],
            'mixamorig:Head':   [(t, (v, 0, w)) for t, v, w in zip(ts, s(-0.018), s(0.02, 1.2))],
            'mixamorig:LeftArm':  [(t, (v, 0, 0)) for t, v in zip(ts, s(0.045, 0.5))],
            'mixamorig:RightArm': [(t, (v, 0, 0)) for t, v in zip(ts, s(0.045, 0.5))],
        },
        'trans': {
            'mixamorig:Hips': [(t, (0, v, 0)) for t, v in zip(ts, s(0.005))],
        },
    }

def slash_keys():
    ts = [0.0, 0.45, 0.62, 0.95, 1.35, 1.8]
    def track(vals):
        return list(zip(ts, vals))
    return {
        'rot': {
            'mixamorig:LeftArm':    track([(v, 0, 0) for v in (0, -0.28, -2.0, -2.05, -0.9, 0)]),
            'mixamorig:RightArm':   track([(v, 0, 0) for v in (0, -0.28, -2.0, -2.05, -0.9, 0)]),
            'mixamorig:LeftForeArm':  track([(v, 0, 0) for v in (0, -0.15, -0.5, -0.55, -0.25, 0)]),
            'mixamorig:RightForeArm': track([(v, 0, 0) for v in (0, -0.15, -0.5, -0.55, -0.25, 0)]),
            'mixamorig:Spine':  track([(v, 0, 0) for v in (0, 0.06, 0.38, 0.40, 0.15, 0)]),
            'mixamorig:Spine2': track([(v, 0, 0) for v in (0, 0.04, 0.25, 0.26, 0.10, 0)]),
            'mixamorig:Head':   track([(v, 0, 0) for v in (0, 0, -0.15, -0.15, -0.05, 0)]),
            'mixamorig:Hips':   track([(0, v, 0) for v in (0, -0.08, 0.22, 0.22, 0.08, 0)]),
        },
        'trans': {
            'mixamorig:Hips': track([(0, v, 0) for v in (0, 0.004, -0.012, -0.012, -0.004, 0)]),
        },
    }

def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, 'rb').read()
    assert data[:4] == b'glTF'
    jlen, jtype = struct.unpack_from('<II', data, 12)
    gltf = json.loads(data[20:20 + jlen])
    boff = 20 + jlen
    blen, _ = struct.unpack_from('<II', data, boff)
    bin_data = bytearray(data[boff + 8: boff + 8 + blen])

    by_name = {}
    bind = {}
    for i, n in enumerate(gltf['nodes']):
        if n.get('name'):
            by_name[n['name']] = i
        if 'matrix' in n:
            bind[i] = decompose(n['matrix'])
        else:
            bind[i] = (n.get('translation', [0, 0, 0]),
                       n.get('rotation', [0, 0, 0, 1]),
                       n.get('scale', [1, 1, 1]))

    def pad4():
        while len(bin_data) % 4:
            bin_data.append(0)

    def add_accessor(arr, comp_type, acc_type, flat):
        pad4()
        off = len(bin_data)
        bin_data.extend(struct.pack('<%df' % len(flat), *flat))
        gltf['bufferViews'].append({'buffer': 0, 'byteOffset': off,
                                    'byteLength': len(flat) * 4})
        acc = {'bufferView': len(gltf['bufferViews']) - 1,
               'componentType': 5126, 'count': arr[0], 'type': acc_type}
        if comp_type == 'times':
            acc['min'] = [arr[1]]
            acc['max'] = [arr[2]]
        gltf['accessors'].append(acc)
        return len(gltf['accessors']) - 1

    def build_clip(name, design):
        samplers, channels = [], []
        for kind, path, ncomp in (('rot', 'rotation', 4), ('trans', 'translation', 3)):
            for jname, keys in design.get(kind, {}).items():
                ni = by_name[jname]
                bt, bq, _ = bind[ni]
                times = [k[0] for k in keys]
                vals = []
                prev = None
                for _, off in keys:
                    if path == 'rotation':
                        q = qmul(bq, qeuler(*off))
                        if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0:
                            q = [-v for v in q]
                        prev = q
                        vals.extend(q)
                    else:
                        vals.extend([bt[i] + off[i] for i in range(3)])
                ti = add_accessor((len(times), min(times), max(times)), 'times',
                                  'SCALAR', times)
                vi = add_accessor((len(times),), 'vals',
                                  'VEC4' if ncomp == 4 else 'VEC3', vals)
                samplers.append({'input': ti, 'output': vi,
                                 'interpolation': 'LINEAR'})
                channels.append({'sampler': len(samplers) - 1,
                                 'target': {'node': ni, 'path': path}})
        return {'name': name, 'samplers': samplers, 'channels': channels}

    gltf['animations'] = [build_clip('idle', idle_keys()),
                          build_clip('slash', slash_keys())]
    gltf['buffers'][0]['byteLength'] = len(bin_data)

    js = json.dumps(gltf, separators=(',', ':')).encode()
    while len(js) % 4:
        js += b' '
    out = bytearray()
    out += struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_data))
    out += struct.pack('<II', len(js), 0x4E4F534A) + js
    out += struct.pack('<II', len(bin_data), 0x004E4942) + bin_data
    open(dst, 'wb').write(out)
    print(f'wrote {dst}: {len(out)} bytes, clips:',
          [(a['name'], len(a['channels'])) for a in gltf['animations']])

if __name__ == '__main__':
    main()
