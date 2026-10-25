import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsgen import beatmap, export, hub, lights, sample, timing, tokens
from bsgen.tokens import SWING

ROOT = Path(__file__).resolve().parents[1]


def ogg_vorbis(path):
    data = Path(path).read_bytes()
    if data[:4] == b"OggS" and b"vorbis" in data[:64]:
        return data
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-c:a", "libvorbis", "-q:a", "6", "-f", "ogg", "-"],
                          capture_output=True, check=True).stdout


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="map model checkpoint (default: the released model, downloaded once)")
    ap.add_argument("--map", help="take song + timing from this map zip")
    ap.add_argument("--diff", default="ExpertPlus", help="difficulty to read from --map")
    ap.add_argument("--difficulty", nargs="+", choices=list(export.DIFF_RANK),
                    help="one per --stars value (default: from the stars)")
    ap.add_argument("--audio")
    ap.add_argument("--bpm", type=float, help="default: detected from the audio")
    ap.add_argument("--offset-ms", type=float, help="song time of the first beat (with --bpm; default: found in the audio)")
    ap.add_argument("--stars", type=float, nargs="+", required=True,
                    help="one difficulty per value, all in one level (e.g. --stars 4.5 7.5 10)")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--candidates", type=int, default=4,
                    help="samples per window; one whose density is far from the others is passed over")
    ap.add_argument("--njs", type=float, nargs="+",
                    help="note speed, one for every difficulty or one per difficulty (default: the usual one)")
    ap.add_argument("--max-loop", type=float, default=sample.MAX_LOOP,
                    help="beats a pattern may repeat back to back before its continuation is penalized; 0: off")
    ap.add_argument("--dry", type=float, default=1.0,
                    help="penalty on a note that carries on a stretch of notes played earlier (0: off; ~1)")
    ap.add_argument("--xtc", type=float, default=0.3, help="how often the likeliest note choices are excluded (0: off)")
    ap.add_argument("--arc-boost", type=float, default=0.0, help="logits added to arcs where one is allowed")
    ap.add_argument("--arc-rate", type=float, default=0.0,
                    help="arcs per note to steer toward (human maps that have arcs: ~0.025); 0: off")
    ap.add_argument("--min-p", type=float, default=0.0, help="keep tokens at least this share of the likeliest's (0: top-p)")
    ap.add_argument("--style", help="acc, tech, midspeed or speed, or a blend like tech=0.7,speed=0.3")
    ap.add_argument("--mapper", help="map like this mapper (see --list-mappers)")
    ap.add_argument("--list-mappers", action="store_true")
    ap.add_argument("--nps", type=float, help="notes per second over the whole map")
    ap.add_argument("--pass-rating", type=float)
    ap.add_argument("--acc-rating", type=float)
    ap.add_argument("--tech-rating", type=float)
    ap.add_argument("--no-walls", action="store_true")
    ap.add_argument("--no-bombs", action="store_true")
    ap.add_argument("--no-arcs", action="store_true")
    ap.add_argument("--no-lights", action="store_true", help="leave the lightshow out")
    ap.add_argument("--format", choices=["v2", "v3"], default="v2",
                    help="v2: read by every viewer, two wall shapes; v3: exact wall shapes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "out"))
    return ap


def run(args, model=None, log=print):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import librosa
    if args.map:
        with beatmap.open_map(args.map) as z:
            info = beatmap.read_info(z)
            diff = beatmap.read_difficulty(z, info, "Standard", args.diff)
            y, sr = beatmap.read_audio(z, info)
            audio_bytes = z.read(beatmap._member(z, info.audio_file))
        tempo, bpm, title = diff.tempo, info.bpm, info.song_name
        offset, _ = beatmap.grid_offset(diff.notes["beat"])
    else:
        audio_bytes = ogg_vorbis(args.audio)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        title = Path(args.audio).stem
        bpm, first = args.bpm, (args.offset_ms or 0) / 1000
        if not bpm or args.offset_ms is None:
            logmel, fps = sample.song_mel(y, sr)
            local = ROOT / "runs" / "timing"
            root = local if (local / "timing.safetensors").exists() else hub.timing_model()
            bpm, first = timing.detect(logmel, fps, root, args.device, bpm=bpm)
            log(f"timing: {bpm:.3f} BPM{' (given)' if args.bpm else ''}, first beat at {first * 1000:.0f} ms")
        tempo = beatmap.TempoMap(bpm)
        offset = first * bpm / 60
    dur = len(y) / sr
    n_beats = int(tempo.to_beats(np.array([dur]))[0])
    log(f"song: {dur:.0f} s, {n_beats} beats")
    beat_times = tempo.to_seconds(np.arange(0, n_beats + 16, dtype=float))
    logmel, fps = sample.song_mel(y, sr)
    model = model or sample.load_model(args.ckpt or hub.map_model(), args.device)
    if args.list_mappers:
        log("\n".join(model.mappers) or "this model has no mapper control")
        return None
    stars = args.stars
    names = args.difficulty or [export.difficulty_for(x) for x in stars]
    njs = args.njs or [export.DIFF_NJS[n] for n in names]
    if len(names) != len(stars) or len(njs) not in (1, len(stars)):
        raise ValueError("give one --difficulty per --stars value, and one --njs or one per --stars value")
    if len(set(names)) < len(names):
        raise ValueError(f"two difficulties would share a name ({', '.join(names)}); name them with --difficulty")
    njs = njs * len(stars) if len(njs) == 1 else njs
    objects = (0 if args.no_walls else tokens.OBJ_WALLS) | (0 if args.no_bombs else tokens.OBJ_BOMBS) \
        | (0 if args.no_arcs else tokens.OBJ_ARCS)
    styles = None
    if args.style:
        styles = {}
        for part in args.style.split(","):
            name, _, w = part.partition("=")
            if name.strip() not in tokens.STYLES:
                raise ValueError(f"unknown style {name!r}; one of {', '.join(tokens.STYLES)}")
            styles[name.strip()] = float(w or 1)
    if not getattr(model, "controls", False) and (styles or args.mapper or args.nps):
        log("note: this model only takes a star rating; the other controls are ignored")

    changes = list(zip(tempo.beats[1:], tempo.bpms[1:])) if tempo.has_changes else []
    levels = []
    for star, name, speed in zip(stars, names, njs):
        if len(stars) > 1:
            log(f"{name}: {star:g}★, NJS {speed:g}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        controls = dict(njs=speed, nps=args.nps, mapper=args.mapper, objects=objects, acc=args.acc_rating,
                        tech=args.tech_rating, **{"pass": args.pass_rating})
        hooks = []
        ev, b = sample.generate(model, logmel, fps, beat_times, offset, n_beats, star, args.cfg, args.temp,
                                args.top_p, dev=args.device, log=log, candidates=args.candidates,
                                controls=controls, styles=styles, max_loop=args.max_loop, dry=args.dry,
                                xtc=args.xtc, min_p=args.min_p, arc_boost=args.arc_boost,
                                arc_rate=args.arc_rate, hooks=hooks)
        bombs, walls, arcs = tokens.to_objects(ev, offset)
        log(f"bombs {len(bombs)}, walls {len(walls)}, arcs {len(arcs)}")

        notes = tokens.to_notes(ev, offset)
        secs = tempo.to_seconds(notes["beat"])
        span = secs.max() - secs.min() if len(notes) else 1
        log(f"{len(notes)} notes, {len(notes) / span:.2f} notes/s, swings {np.sum(ev['kind'] == SWING)}"
            f" ({np.mean(ev['air'][ev['kind'] == SWING]):.0%} air)")
        log(f"probability the model put on moves the rules blocked: mean {b.mean():.1%}, "
            f"steps where it was over half: {np.mean(b > 0.5):.1%}")
        show = [] if args.no_lights else lights.lightshow(notes, beat_times, logmel, fps, args.seed)
        if show:
            log(f"lights: {len(show)} events, "
                f"{len(show) / max(notes['beat'].max() - notes['beat'].min(), 1):.1f} per beat")
        levels.append(dict(difficulty=name, notes=notes, label=f"bsgen {star:g}★", njs=speed, bombs=bombs,
                           walls=walls, arcs=arcs, lights=show))
    tag = ",".join(f"{x:g}" for x in stars)
    folder = Path(args.out) / export.safe_name(f"{title[:40]} {tag}star seed{args.seed}")
    export.write_pack(folder, audio_bytes, bpm, f"{title} (bsgen {stars[0]:g}★)" if len(stars) == 1 else f"{title} (bsgen)",
                      levels, tempo_changes=changes, fmt=args.format)
    log(f"wrote {folder}")
    return folder.parent / (folder.name + ".zip")


def main():
    run(parser().parse_args(), log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
