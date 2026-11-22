#!/usr/bin/env python3
"""Generate models/sample.glb — a wooden mannequin with a real skeleton,
two animation clips (wave, bow). Rigid-skinned boxes per bone segment.
Used as Dbox's demo model and as the known-good fixture for the viewer.
"""
import json, math, struct, os

OUT = os.path.join(os.path.dirname(__file__), "..", "models", "sample.glb")

# ---------- math helpers ----------
def quat(axis, deg):
    a = math.radians(deg) / 2.0
    s = math.sin(a)
    return [axis[0]*s, axis[1]*s, axis[2]*s, math.cos(a)]

def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz]

# ---------- skeleton ----------
# name, parent, local translation
BONES = [
    ("hips",       None,    (0.0, 0.95, 0.0)),
    ("spine",      "hips",  (0.0, 0.14, 0.0)),
    ("chest",      "spine", (0.0, 0.24, 0.0)),
    ("neck",       "chest", (0.0, 0.30, 0.0)),
    ("head",       "neck",  (0.0, 0.10, 0.0)),
    ("upperArm.L", "chest", (-0.24, 0.24, 0.0)),
    ("foreArm.L",  "upperArm.L", (0.0, -0.28, 0.0)),
    ("upperArm.R", "chest", (0.24, 0.24, 0.0)),
    ("foreArm.R",  "upperArm.R", (0.0, -0.28, 0.0)),
    ("thigh.L",    "hips",  (-0.10, -0.04, 0.0)),
    ("shin.L",     "thigh.L", (0.0, -0.42, 0.0)),
    ("foot.L",     "shin.L",  (0.0, -0.42, 0.0)),
    ("thigh.R",    "hips",  (0.10, -0.04, 0.0)),
    ("shin.R",     "thigh.R", (0.0, -0.42, 0.0)),
    ("foot.R",     "shin.R",  (0.0, -0.42, 0.0)),
]
bidx = {b[0]: i for i, b in enumerate(BONES)}
world = {}
for name, parent, t in BONES:
    if parent is None:
        world[name] = t
    else:
        p = world[parent]
        world[name] = (p[0]+t[0], p[1]+t[1], p[2]+t[2])

# ---------- geometry: boxes in bind-pose world space ----------
pos, nor, col, jnt, wgt, idx = [], [], [], [], [], []
def box(center, size, color, bone):
    cx, cy, cz = center; sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
    faces = [  # (normal, four corners)
        ((0,0,1),  [(-sx,-sy,sz),(sx,-sy,sz),(sx,sy,sz),(-sx,sy,sz)]),
        ((0,0,-1), [(sx,-sy,-sz),(-sx,-sy,-sz),(-sx,sy,-sz),(sx,sy,-sz)]),
        ((1,0,0),  [(sx,-sy,sz),(sx,-sy,-sz),(sx,sy,-sz),(sx,sy,sz)]),
        ((-1,0,0), [(-sx,-sy,-sz),(-sx,-sy,sz),(-sx,sy,sz),(-sx,sy,-sz)]),
        ((0,1,0),  [(-sx,sy,sz),(sx,sy,sz),(sx,sy,-sz),(-sx,sy,-sz)]),
        ((0,-1,0), [(-sx,-sy,-sz),(sx,-sy,-sz),(sx,-sy,sz),(-sx,-sy,sz)]),
    ]
    ji = bidx[bone]
    for n, corners in faces:
        base = len(pos)
        for dx, dy, dz in corners:
            pos.append((cx+dx, cy+dy, cz+dz)); nor.append(n); col.append(color)
            jnt.append((ji, 0, 0, 0)); wgt.append((1.0, 0.0, 0.0, 0.0))
        idx.extend([base, base+1, base+2, base, base+2, base+3])

W  = world
WOOD  = (0.72, 0.52, 0.33)
WOOD2 = (0.62, 0.43, 0.27)
SKIN  = (0.85, 0.65, 0.45)
box((W["hips"][0],  W["hips"][1]+0.02, 0), (0.30, 0.17, 0.19), WOOD2, "hips")
box((0, (W["hips"][1]+W["chest"][1])/2 + 0.03, 0), (0.25, 0.36, 0.16), WOOD, "spine")
box((0, W["chest"][1]+0.08, 0), (0.35, 0.30, 0.20), WOOD, "chest")
box((0, W["neck"][1]+0.03, 0), (0.09, 0.10, 0.09), SKIN, "neck")
box((0, W["head"][1]+0.13, 0), (0.21, 0.25, 0.22), SKIN, "head")
for s, sx in (("L", -1), ("R", 1)):
    ua, fa = W[f"upperArm.{s}"], W[f"foreArm.{s}"]
    box((ua[0], ua[1]-0.14, 0), (0.095, 0.28, 0.095), WOOD, f"upperArm.{s}")
    box((fa[0], fa[1]-0.13, 0), (0.08, 0.26, 0.08), WOOD2, f"foreArm.{s}")
    box((fa[0], fa[1]-0.30, 0), (0.085, 0.10, 0.035), SKIN, f"foreArm.{s}")
    th, sh, ft = W[f"thigh.{s}"], W[f"shin.{s}"], W[f"foot.{s}"]
    box((th[0], th[1]-0.21, 0), (0.135, 0.42, 0.135), WOOD, f"thigh.{s}")
    box((sh[0], sh[1]-0.21, 0), (0.11, 0.42, 0.11), WOOD2, f"shin.{s}")
    box((ft[0], ft[1]-0.015, 0.055), (0.095, 0.06, 0.25), WOOD2, f"foot.{s}")

# ---------- binary buffer ----------
buf = bytearray()
views, accs = [], []
def add_accessor(data, fmt, comp, typ, count, minmax=None):
    global buf
    while len(buf) % 4: buf.append(0)
    off = len(buf)
    for item in data:
        if isinstance(item, (tuple, list)): buf += struct.pack("<" + fmt*len(item), *item)
        else: buf += struct.pack("<" + fmt, item)
    ncomp = {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4, "MAT4":16}[typ]
    views.append({"buffer": 0, "byteOffset": off, "byteLength": len(buf)-off})
    a = {"bufferView": len(views)-1, "componentType": comp, "count": count, "type": typ}
    if minmax: a["min"], a["max"] = minmax
    accs.append(a)
    return len(accs)-1

F32, U16, U32 = 5126, 5123, 5125
mins = [min(p[i] for p in pos) for i in range(3)]; maxs = [max(p[i] for p in pos) for i in range(3)]
a_pos = add_accessor(pos, "f", F32, "VEC3", len(pos), (mins, maxs))
a_nor = add_accessor(nor, "f", F32, "VEC3", len(nor))
a_col = add_accessor(col, "f", F32, "VEC3", len(col))
a_jnt = add_accessor(jnt, "B", 5121, "VEC4", len(jnt))
a_wgt = add_accessor(wgt, "f", F32, "VEC4", len(wgt))
a_idx = add_accessor(idx, "I", U32, "SCALAR", len(idx))

# inverse bind matrices: bind world = pure translation -> IBM = translate(-world)
ibms = []
for name, _, _ in BONES:
    x, y, z = world[name]
    ibms.append((1,0,0,0, 0,1,0,0, 0,0,1,0, -x,-y,-z,1))
a_ibm = add_accessor(ibms, "f", F32, "MAT4", len(ibms))

# ---------- animations ----------
def anim(name, channels):
    """channels: list of (bone, path, times, values) values are tuples."""
    samplers, chans = [], []
    for bone, path, times, values in channels:
        ai = add_accessor(times, "f", F32, "SCALAR", len(times), ([min(times)], [max(times)]))
        typ = "VEC4" if path == "rotation" else "VEC3"
        ao = add_accessor(values, "f", F32, typ, len(values))
        samplers.append({"input": ai, "output": ao, "interpolation": "LINEAR"})
        chans.append({"sampler": len(samplers)-1,
                      "target": {"node": NODE_OF[bidx[bone]], "path": path}})
    return {"name": name, "samplers": samplers, "channels": chans}

Z = (0, 0, 1); X = (1, 0, 0); Y = (0, 1, 0)
# node indices: 0 = root "Mannequin", joints start at 1
NODE_OF = {i: i + 1 for i in range(len(BONES))}

wave = anim("wave", [
    ("upperArm.R", "rotation", [0, 0.5, 1.0, 1.5, 2.0],
     [quat(Z, 10), quat(Z, 145), quat(Z, 118), quat(Z, 145), quat(Z, 118)]),
    ("foreArm.R", "rotation", [0, 0.5, 1.0, 1.5, 2.0],
     [quat(Z, 0), quat(Z, 28), quat(Z, -14), quat(Z, 28), quat(Z, -14)]),
    ("head", "rotation", [0, 1.0, 2.0], [quat(Z, 0), quat(Z, -7), quat(Z, 0)]),
    ("hips", "translation", [0, 0.5, 1.0, 1.5, 2.0],
     [(0, 0.95, 0), (0, 0.965, 0), (0, 0.95, 0), (0, 0.965, 0), (0, 0.95, 0)]),
])
bow = anim("bow", [
    ("spine", "rotation", [0, 0.8, 1.4, 2.2], [quat(X, 0), quat(X, 26), quat(X, 26), quat(X, 0)]),
    ("chest", "rotation", [0, 0.8, 1.4, 2.2], [quat(X, 0), quat(X, 22), quat(X, 22), quat(X, 0)]),
    ("head", "rotation", [0, 0.8, 1.4, 2.2], [quat(X, 0), quat(X, 14), quat(X, 14), quat(X, 0)]),
    ("upperArm.L", "rotation", [0, 0.8, 1.4, 2.2], [quat(X, 0), quat(X, -18), quat(X, -18), quat(X, 0)]),
    ("upperArm.R", "rotation", [0, 0.8, 1.4, 2.2], [quat(X, 0), quat(X, -18), quat(X, -18), quat(X, 0)]),
])

# ---------- assemble glTF ----------
nodes = [{"name": "Mannequin", "children": [NODE_OF[0]]}]
for i, (name, parent, t) in enumerate(BONES):
    n = {"name": name, "translation": list(t)}
    kids = [NODE_OF[bidx[c[0]]] for c in BONES if c[1] == name]
    if kids: n["children"] = kids
    if name == "hips": n["children"] = n.get("children", []) + [len(BONES) + 1]
    nodes.append(n)
nodes.append({  # mesh node, child of hips so it follows the root joint
    "name": "MannequinMesh", "mesh": 0, "skin": 0,
})
gltf = {
    "asset": {"version": "2.0", "generator": "dbox make_sample_glb.py"},
    "scene": 0, "scenes": [{"nodes": [0]}],
    "nodes": nodes,
    "meshes": [{"name": "MannequinMesh", "primitives": [{
        "attributes": {"POSITION": a_pos, "NORMAL": a_nor, "COLOR_0": a_col,
                       "JOINTS_0": a_jnt, "WEIGHTS_0": a_wgt},
        "indices": a_idx, "material": 0}]}],
    "materials": [{"name": "wood", "pbrMetallicRoughness":
                   {"baseColorFactor": [1, 1, 1, 1], "roughnessFactor": 0.85, "metallicFactor": 0.0}}],
    "skins": [{"name": "MannequinSkin", "joints": [NODE_OF[i] for i in range(len(BONES))],
               "inverseBindMatrices": a_ibm, "skeleton": NODE_OF[0]}],
    "animations": [wave, bow],
    "buffers": [{"byteLength": len(buf)}],
    "bufferViews": views, "accessors": accs,
}
js = json.dumps(gltf).encode()
while len(js) % 4: js += b" "
while len(buf) % 4: buf.append(0)
gltf["buffers"][0]["byteLength"] = len(buf)
js = json.dumps(gltf).encode()
while len(js) % 4: js += b" "
glb = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(js) + 8 + len(buf))
glb += struct.pack("<I", len(js)) + b"JSON" + js
glb += struct.pack("<I", len(buf)) + b"BIN\x00" + bytes(buf)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as fh: fh.write(glb)
print("wrote", OUT, len(glb), "bytes | bones:", len(BONES), "| verts:", len(pos), "| clips:", [a["name"] for a in gltf["animations"]])
