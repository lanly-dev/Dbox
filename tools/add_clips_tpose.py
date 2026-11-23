#!/usr/bin/env python3
"""T-pose variant of add_clips.py: same machinery, mirrored clip designs.

For rigs whose bind pose is a true T-pose (arms horizontal), the Mixamo
arm bones are mirrored: LeftArm lowers with -X, RightArm lowers with +X.
These designs also bake a natural rest (arms at her sides) into both
clips, so idle does not stand in a scarecrow T and slash starts/ends at
that same rest.

Usage: add_clips_tpose.py <src.glb> <dst.glb>
"""
import math, sys
sys.path.insert(0, 'tools')
import add_clips

REST_L, REST_R = -1.22, 1.22          # arms lowered to her sides
REST_FL, REST_FR = -0.12, 0.12        # slight forearm bend

def idle_keys():
    ts = [0.0, 1.0, 2.0, 3.0, 4.0]
    def s(amp, phase=0.0):
        return [amp * math.sin(2 * math.pi * t / 4.0 + phase) for t in ts]
    return {
        'rot': {
            'mixamorig:Spine':  [(t, (v, 0, 0)) for t, v in zip(ts, s(0.035))],
            'mixamorig:Spine2': [(t, (v, 0, 0)) for t, v in zip(ts, s(0.022))],
            'mixamorig:Head':   [(t, (v, 0, w)) for t, v, w in zip(ts, s(-0.018), s(0.02, 1.2))],
            'mixamorig:LeftArm':  [(t, (REST_L + v, 0, 0)) for t, v in zip(ts, s(0.04, 0.5))],
            'mixamorig:RightArm': [(t, (REST_R + v, 0, 0)) for t, v in zip(ts, s(0.04, 0.5))],
            'mixamorig:LeftForeArm':  [(t, (REST_FL, 0, 0)) for t in ts],
            'mixamorig:RightForeArm': [(t, (REST_FR, 0, 0)) for t in ts],
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
            # rest -> overhead windup -> chop down past rest -> hold -> recover to rest
            'mixamorig:LeftArm':    track([(v, 0, 0) for v in (REST_L, 1.9, -1.75, -1.8, -1.3, REST_L)]),
            'mixamorig:RightArm':   track([(v, 0, 0) for v in (REST_R, -1.9, 1.75, 1.8, 1.3, REST_R)]),
            'mixamorig:LeftForeArm':  track([(v, 0, 0) for v in (REST_FL, 0.35, -0.1, -0.1, 0.0, REST_FL)]),
            'mixamorig:RightForeArm': track([(v, 0, 0) for v in (REST_FR, -0.35, 0.1, 0.1, 0.0, REST_FR)]),
            'mixamorig:Spine':  track([(v, 0, 0) for v in (0, -0.06, 0.38, 0.40, 0.15, 0)]),
            'mixamorig:Spine2': track([(v, 0, 0) for v in (0, -0.04, 0.25, 0.26, 0.10, 0)]),
            'mixamorig:Head':   track([(v, 0, 0) for v in (0, 0, -0.15, -0.15, -0.05, 0)]),
            'mixamorig:Hips':   track([(0, v, 0) for v in (0, -0.08, 0.22, 0.22, 0.08, 0)]),
        },
        'trans': {
            'mixamorig:Hips': track([(0, v, 0) for v in (0, 0.004, -0.012, -0.012, -0.004, 0)]),
        },
    }

def stand_keys():
    # The original art (slide 2): a wide planted stance, both arms raised
    # overhead with elbows bent, hands together above her head in a high
    # guard, chin down. Held pose — constant keys, a whisper of breath on
    # the spine only. (Her katana stays sheathed in the mesh, so the guard
    # is empty-handed, same as the slash clip.)
    ts = [0.0, 2.0, 4.0]
    def s(amp, phase=0.0):
        return [amp * math.sin(2 * math.pi * t / 4.0 + phase) for t in ts]
    def held(v):
        return [(t, v) for t in ts]
    return {
        'rot': {
            'mixamorig:Spine':  [(t, (v, 0.05, 0)) for t, v in zip(ts, s(0.008))],
            'mixamorig:Spine2': [(t, (v, 0.03, 0)) for t, v in zip(ts, s(0.005))],
            'mixamorig:Head':   held((-0.08, 0, 0)),
            'mixamorig:LeftArm':  held((2.20, 0, 0)),
            'mixamorig:RightArm': held((-2.20, 0, 0)),
            'mixamorig:LeftForeArm':  held((-1.15, 0, 0)),
            'mixamorig:RightForeArm': held((1.15, 0, 0)),
            # fingers curled as if gripping the (sheathed, so absent) katana
            **{f'mixamorig:{side}Hand{finger}{j}': held((0.5, 0, 0))
               for side in ('Left', 'Right')
               for finger in ('Index', 'Middle', 'Ring', 'Pinky')
               for j in (1, 2, 3)},
            **{f'mixamorig:{side}HandThumb{j}': held((0.3, 0, 0))
               for side in ('Left', 'Right') for j in (1, 2, 3)},
            # Leg splay is the UpLeg X axis on this rig (probed: Z is the
            # sagittal swing, X +/-1.0 does the splits). Feet counter-tilt
            # so the sandals stay flat.
            'mixamorig:LeftUpLeg':  held((0.38, 0, 0)),
            'mixamorig:RightUpLeg': held((-0.38, 0, 0)),
            'mixamorig:LeftFoot':  held((-0.38, 0, 0)),
            'mixamorig:RightFoot': held((0.38, 0, 0)),
        },
        'trans': {
            'mixamorig:Hips': [(t, (0, -0.035 + v, 0)) for t, v in zip(ts, s(0.0015))],
        },
    }

add_clips.idle_keys = idle_keys
add_clips.slash_keys = slash_keys
add_clips.stand_keys = stand_keys

if __name__ == '__main__':
    add_clips.main()
