import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import beatmap, bsor, judge
from .player import HZ, Player, objects_from_map, play

BSOR_HEIGHT = 1.6


def load_player(path, dev="cuda"):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        ck = dict(json.loads(path.with_suffix(".json").read_text()), model=load_file(path))
    else:
        ck = torch.load(path, map_location="cpu")
    model = Player(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"])
    return model.eval()


def map_objects(zip_path, difficulty, characteristic="Standard"):
    with beatmap.open_map(zip_path) as z:
        info = beatmap.read_info(z)
        d = beatmap.read_difficulty(z, info, characteristic, difficulty)
    n_beats = int(max([d.notes["beat"].max() if len(d.notes) else 0] +
                      [(d.walls["beat"] + d.walls["duration"]).max() if len(d.walls) else 0])) + 4
    beat_times = d.tempo.to_seconds(np.arange(n_beats, dtype=float)) + info.song_offset
    return objects_from_map(d.notes, d.bombs, d.walls, beat_times), d.njs or 18.0, info


def from_song_start(motion, t0):
    n0 = int(round(t0 * HZ))
    return np.concatenate([np.repeat(motion[:1], n0, 0), motion]) if n0 >= 0 else motion[-n0:]


def write_replay(path, zip_path, difficulty, motion, info, njs, objects, player="bsgen bot"):
    events = judge.note_events(judge.Notes(objects, njs), motion, BSOR_HEIGHT)
    m = motion.reshape(-1, 3, 9).astype(np.float64)
    quat = Rotation.from_matrix(judge.rotations(m[..., 3:]).reshape(-1, 3, 3)).as_quat().reshape(-1, 3, 4)
    meta = {"player": player, "hash": bsor.map_hash(zip_path, info), "song": info.song_name,
            "mapper": info.mapper, "difficulty": difficulty, "score": judge.replay_score(events),
            "environment": "DefaultEnvironment", "height": BSOR_HEIGHT}
    bsor.write_bsor(path, meta, m[..., :3], quat, events, HZ)
    return events


def summary(events):
    kinds = np.array([e[5] for e in events])
    return {"notes": len(events), "good": int(np.sum(kinds == 0)), "bad": int(np.sum(kinds == 1)),
            "missed": int(np.sum(kinds == 2)), "score": judge.replay_score(events)}


def play_map(model, zip_path, difficulty, out, seed=0, candidates=4, steps=12, height=1.7, progress=None,
             dev="cuda"):
    objects, njs, info = map_objects(zip_path, difficulty)
    motion, t0 = play(model, objects, height, njs, steps=steps, seed=seed, dev=dev, candidates=candidates,
                      progress=progress)
    events = write_replay(out, zip_path, difficulty, from_song_start(motion, t0), info, njs, objects)
    return summary(events)
