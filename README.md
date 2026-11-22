# Dbox

A box, in 3D. A diorama-style stage for rigged 3D models — one HTML file,
raw WebGL2, no libraries, no build step.

**Live:** GitHub Pages serves `index.html` from this repo as-is.

## What it does

- Presents a `.glb` model standing inside an open-front box with a visible frame
- **Editable backdrop** — the back wall takes a preset image, your own uploaded
  image, or a solid color; brightness and side-wall wrap included
- **Animation** — plays any clips baked into the GLB, with a time scrubber
- **Posing** — rotation sliders generated from the rig's actual bones, so it
  works with any skeleton (Mixamo, RigAnything, hand-made, …)
- **Drag & drop** — drop a `.glb` anywhere to load it; drop an image to make
  it the backdrop
- Orbit camera that stays in front of the box, gentle turntable sway,
  box width/height/depth sliders, PNG snapshot button
- **Mobile friendly** — the control panel becomes a collapsible bottom sheet,
  one finger orbits, pinch zooms, and the framing adapts to portrait screens

## Files

| Path | What |
| --- | --- |
| `index.html` | the whole app (~36 KB, zero dependencies) |
| `models/sample.glb` | wooden mannequin placeholder: 15 bones, `wave` + `bow` clips |
| `assets/` | backdrop presets (skull wall and source art from the exhibit's artwork, generated sunset) |
| `tools/make_sample_glb.py` | regenerates `models/sample.glb` |

## Bring your own model

Drop any `.glb` on the page, or point the URL at one:

```
index.html?model=models/sample.glb&clip=wave&t=1.0&backdrop=sunset
```

URL parameters: `model`, `clip` (clip name), `t` (start time, seconds,
paused), `backdrop` (`skullwall` | `sourceart` | `sunset` | `color`),
`title` (plaque text).

## The exhibit

Dbox was built to present a rigged 3D model of the "Samurai Girl — Waltz"
artwork (AI art by @ai.art_hana): image → Hunyuan mesh → RigAnything rig →
this box. Until the rigged `.glb` arrives, she stands in the box as a flat
cutout (`models/samurai-cutout.glb`, built by `tools/make_cutout_glb.py`
from `assets/samurai-cutout.png`, the source art with its background
removed) with a gentle `sway` clip. Swapping in the real rigged model is a
one-line change in the `CONFIG` block at the top of `index.html`.
