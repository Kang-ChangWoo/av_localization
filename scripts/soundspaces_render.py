#!/usr/bin/env python3
"""Render RIRs with SoundSpaces 2.0 on a floorplan proxy, over a candidate grid.

This is the high-fidelity acoustic likelihood path: instead of approximating
echo arrivals analytically, every candidate pose gets a room impulse response
from the same engine that produced the observed recordings, on the same
watertight floorplan mesh. Model and observation then share one physics.

Run it through the simulator environment, which sets the required LD_PRELOAD:

    source .envs/ss_env.sh
    $SS_ENV/bin/python scripts/soundspaces_render.py --scene office_4

Four things had to be right before this reproduced the shipped recordings, and
each is a silent failure if wrong:

* ``spec.position = [0, 0, 0]`` -- habitat offsets sensors 1.5 m above the agent
  by default, which puts the receiver through the ceiling and returns silence.
* ``gpu_device_id = -1`` -- this container runs without the NVIDIA graphics
  capability, so only Mesa's software EGL device exists and the CUDA/EGL match
  fails. Audio does not need the GPU.
* ``acousticsConfig.enableMaterials = False`` -- the flag lives on the acoustics
  config, not the sensor spec; left on, the audio sensor asks for a semantic
  mesh the proxy does not have and aborts.
* ``habitat = f3loc + mesh_centre`` -- the proxy's own frame, verified against
  the shipped ``array_center_habitat``.

Verified against the dataset: per-channel energy-envelope correlation 0.999
with the shipped ``floorplan_closed`` RIR at the same pose, up to one global
gain (the engine is stochastic, so waveform correlation is ~0.9).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# EchoScan ring order, matching dataset_meta.json acoustics.ring_angles_rad
RING_ANGLES_RAD = np.array(
    [np.pi, 4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi, np.pi / 3, 2 * np.pi / 3]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--desdf-dir", type=Path, default=None,
                   help="where <scene>/desdf.npy lives; default <dataset-root>/desdf. Validation "
                        "rooms use a cache built by scripts/build_desdf.py")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="default: outputs/acoustic_grid/<scene>.npz")
    p.add_argument("--yaw-bins", type=int, default=1,
                   help="ring orientations to render; 1 is enough while the "
                        "likelihood is empirically flat in yaw")
    p.add_argument("--ring-radius-m", type=float, default=0.05)
    p.add_argument("--layout", default="ring", choices=["ring", "binaural"],
                   help="'ring' reproduces the shipped candidate grids: six "
                        "omnidirectional receivers moved around a 5 cm circle, "
                        "which carries no directivity and therefore no heading. "
                        "'binaural' renders the engine's HRTF pair at the "
                        "head orientations given by --yaw-bins, which does. The "
                        "ring costs six engine calls per cell and binaural one, "
                        "so four head orientations are cheaper than one ring.")
    p.add_argument("--source-height-m", type=float, default=None,
                   help="receiver height in habitat y; default: read it from the "
                        "scene's shipped rir_metadata.json so candidates and "
                        "observations share the same height")
    p.add_argument("--clearance-m", type=float, default=0.25)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--indirect-rays", type=int, default=4096)
    p.add_argument("--ray-depth", type=int, default=6)
    p.add_argument("--source-rays", type=int, default=512)
    p.add_argument("--diffraction", action="store_true",
                   help="match the shipped recordings, which enable it")
    p.add_argument("--max-diffraction-order", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="cells to render (smoke tests)")
    p.add_argument("--shard", type=int, default=0,
                   help="this worker's index; cells are split round-robin so every "
                        "shard covers the whole room and a partial run is still usable")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


def quat_from_yaw(yaw: float):
    """Agent rotation for a yaw about habitat's up axis.

    Habitat's y is up and its yaw runs opposite to the floorplan's, which is the
    same convention the pose conversion elsewhere in this file uses.
    """
    import quaternion
    return quaternion.from_rotation_vector(np.array([0.0, -yaw, 0.0]))


def build_simulator(glb: str, args):
    import habitat_sim
    from habitat_sim.sensor import RLRAudioPropagationChannelLayoutType as LT

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    backend.gpu_device_id = -1          # no NVIDIA EGL in this container; audio is CPU-side

    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.position = np.array([0.0, 0.0, 0.0])   # cancel habitat's 1.5 m sensor offset
    if args.layout == "binaural":
        spec.channelLayout.channelType = LT.Binaural
        spec.channelLayout.channelCount = 2
    else:
        spec.channelLayout.channelType = LT.Mono
        spec.channelLayout.channelCount = 1
    ac = spec.acousticsConfig
    ac.sampleRate = args.sample_rate
    ac.indirectRayCount = args.indirect_rays
    ac.indirectRayDepth = args.ray_depth
    ac.sourceRayCount = args.source_rays
    ac.sourceRayDepth = args.ray_depth
    ac.direct = True
    ac.indirect = True
    # The shipped recordings were re-rendered with diffraction on to order 10.
    # A candidate grid without it is a strictly simpler model than the
    # observation, which is the mismatch that made the matched-domain score
    # collapse from GT rank 6 to 1474; these have to agree.
    ac.diffraction = args.diffraction
    if args.diffraction:
        try:
            ac.maxDiffractionOrder = args.max_diffraction_order
        except AttributeError:
            print("[render] this build exposes no maxDiffractionOrder; "
                  "diffraction is on at the engine default")
    ac.transmission = False
    ac.enableMaterials = False

    agent = habitat_sim.AgentConfiguration()
    agent.sensor_specifications = [spec]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    from track1_core.floorplan import PoseGrid, valid_pose_mask

    root = Path(args.dataset_root)
    scene = args.scene
    glb = str(root / "floorplan_proxy" / scene / "floorplan.glb")

    desdf = np.load((args.desdf_dir or root / "desdf") / scene / "desdf.npy", allow_pickle=True).item()
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    grid = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, grid, clearance_m=args.clearance_m)
    rows, cols = np.nonzero(mask)
    if args.limit:
        rows, cols = rows[: args.limit], cols[: args.limit]
    if args.shards > 1:
        keep = np.arange(len(rows)) % args.shards == args.shard
        rows, cols = rows[keep], cols[keep]
    print(f"[render] {scene}: {len(rows)} candidate cells x {args.yaw_bins} yaw x 6 mics"
          + (f"  [shard {args.shard}/{args.shards}]" if args.shards > 1 else ""))

    sim = build_simulator(glb, args)
    node = sim.get_active_scene_graph().get_root_node().cumulative_bb
    centre = (np.array(node.min) + np.array(node.max)) / 2
    if args.source_height_m is not None:
        height = float(np.array(node.min)[1]) + args.source_height_m
        source = "floor + offset"
    else:
        # match the height the observed recordings were rendered at, otherwise
        # the floor and ceiling reflections land at different delays
        rir_dir = root / "rir" / args.collection / "floorplan_closed" / scene
        sample = sorted(p for p in rir_dir.iterdir() if p.name.startswith("pose_"))[0]
        md = json.loads((sample / "rir_metadata.json").read_text())
        height = float(md["array_center_habitat"][1])
        source = f"observed metadata ({sample.name})"
    print(f"[render] mesh centre {centre.round(3)}  receiver height {height:.3f}  [{source}]")

    audio = sim.get_agent(0)._sensors["audio_sensor"]
    agent = sim.get_agent(0)

    metric = grid.grid_to_metric(np.stack([cols, rows], axis=-1))   # (N, 2) floorplan metres
    yaws = np.arange(args.yaw_bins) / max(args.yaw_bins, 1) * 2 * np.pi

    rirs, index, start = [], [], time.time()
    for i, (mx, my) in enumerate(metric):
        # the proxy's own frame: habitat = f3loc + mesh centre, z flipped
        src = np.array([mx + centre[0], height, -my + centre[2]], dtype=np.float32)
        for yb, yaw in enumerate(yaws):
            if args.layout == "binaural":
                # One head, two ears, rotated to the yaw bin. The HRTF is what
                # makes the two channels differ with heading, which the
                # omnidirectional ring cannot do at any radius.
                st = agent.get_state()
                st.position = src
                st.rotation = quat_from_yaw(yaw)
                agent.set_state(st)
                audio.setAudioSourceTransform(src)
                obs = np.asarray(sim.get_sensor_observations()["audio_sensor"])
                rirs.append(obs[:2].astype(np.float32))
            else:
                angles = RING_ANGLES_RAD + yaw
                channels = []
                for a in angles:
                    off = np.array([args.ring_radius_m * np.cos(a), 0.0,
                                    -args.ring_radius_m * np.sin(a)], dtype=np.float32)
                    st = agent.get_state()
                    st.position = src + off
                    agent.set_state(st)
                    audio.setAudioSourceTransform(src)
                    channels.append(np.asarray(sim.get_sensor_observations()["audio_sensor"])[0])
                n = min(len(c) for c in channels)
                rirs.append(np.stack([c[:n] for c in channels]).astype(np.float32))
            index.append((int(rows[i]), int(cols[i]), yb))
        if (i + 1) % args.progress_every == 0:
            done = i + 1
            rate = done / (time.time() - start)
            print(f"[render] {done}/{len(metric)} cells  {rate:.1f} cells/s  "
                  f"eta {(len(metric)-done)/rate/60:.0f} min")

    sim.close()
    length = min(r.shape[1] for r in rirs)
    stack = np.stack([r[:, :length] for r in rirs])
    out = args.out or REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, rir=stack, index=np.array(index, dtype=np.int32),
        grid=json.dumps(grid.describe()),
        config=json.dumps(vars(args), default=str),
    )
    print(f"[render] wrote {out}  rir{stack.shape}  {stack.nbytes/1e6:.0f} MB "
          f"in {(time.time()-start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
