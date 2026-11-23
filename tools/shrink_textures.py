#!/usr/bin/env python3
"""Shrink a GLB's embedded textures for web delivery.

Base color is capped at 2048 px and re-encoded as JPEG q90; every other
texture (normal / metallic-roughness / occlusion) is capped at 1024 px,
JPEG q85. Geometry, skin, and materials are untouched — only image
payloads and their bufferViews change.

Usage: shrink_textures.py <src.glb> <dst.glb>
"""
import io, json, struct, sys
from PIL import Image

def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, 'rb').read()
    assert data[:4] == b'glTF'
    jlen, _ = struct.unpack_from('<II', data, 12)
    gltf = json.loads(data[20:20 + jlen])
    boff = 20 + jlen
    blen, _ = struct.unpack_from('<II', data, boff)
    bin_data = data[boff + 8: boff + 8 + blen]

    base_tex = None
    for m in gltf.get('materials', []):
        pbr = m.get('pbrMetallicRoughness', {})
        if 'baseColorTexture' in pbr:
            base_tex = gltf['textures'][pbr['baseColorTexture']['index']]['source']
    img_view = {i: img['bufferView'] for i, img in enumerate(gltf.get('images', []))}

    new_payload = {}
    for i, img in enumerate(gltf.get('images', [])):
        bv = gltf['bufferViews'][img['bufferView']]
        raw = bin_data[(bv.get('byteOffset', 0)):(bv.get('byteOffset', 0)) + bv['byteLength']]
        im = Image.open(io.BytesIO(raw))
        cap, q = (2048, 90) if i == base_tex else (1024, 85)
        if max(im.size) > cap:
            im.thumbnail((cap, cap), Image.LANCZOS)
        out = io.BytesIO()
        im.convert('RGB').save(out, 'JPEG', quality=q)
        new_payload[img['bufferView']] = out.getvalue()
        img['mimeType'] = 'image/jpeg'
        print(f'image {i}: {len(raw)/1e6:.2f} MB -> {len(out.getvalue())/1e6:.2f} MB ({im.size[0]}x{im.size[1]})')

    new_bin = bytearray()
    for vi, bv in enumerate(gltf['bufferViews']):
        while len(new_bin) % 4:
            new_bin.append(0)
        payload = new_payload.get(vi)
        if payload is None:
            payload = bin_data[(bv.get('byteOffset', 0)):(bv.get('byteOffset', 0)) + bv['byteLength']]
        bv['byteOffset'] = len(new_bin)
        bv['byteLength'] = len(payload)
        new_bin.extend(payload)
    gltf['buffers'][0]['byteLength'] = len(new_bin)

    js = json.dumps(gltf, separators=(',', ':')).encode()
    while len(js) % 4:
        js += b' '
    out = bytearray()
    out += struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(new_bin))
    out += struct.pack('<II', len(js), 0x4E4F534A) + js
    out += struct.pack('<II', len(new_bin), 0x004E4942) + new_bin
    open(dst, 'wb').write(out)
    print(f'wrote {dst}: {len(out)/1e6:.2f} MB (was {len(data)/1e6:.2f} MB)')

if __name__ == '__main__':
    main()
