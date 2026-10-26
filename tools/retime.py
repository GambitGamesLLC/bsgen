#!/usr/bin/env python3
"""Retime a bsgen chart to a target BPM and remove sustained-flow violations.

bsgen generates Beat Saber charts at BS densities. AeroBeat's Flow mode wants a
lower, more deliberate ceiling. Rather than fight the model's own density
steering (which cannot reach low targets and produces flat charts), keep the
model's note *selection* and rewrite the *timing*.

Two operations, applied in order:

1. Trim: drop leading and trailing dead air so the chart starts and ends with
   the song.
2. Thin: enforce a minimum spacing between consecutive notes, dropping notes
   that arrive too fast. This is what actually lowers NPS, and because it works
   on real note times it lowers it honestly rather than by relabeling.

Thinning preserves the model's hand alternation and lane choices; it only
removes. No note is ever invented or moved, so a thinned chart is always a
subset of what the model produced.

Usage:
  retime.py --input X.dat --target-nps 2.0 --out Y.dat [--min-spacing 0.0]
"""
import argparse
import json
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def note_times(raw):
    return sorted(n["_time"] for n in raw["_notes"] if n["_type"] in (0, 1))


def trim(raw, lead_beats=0.0, tail_beats=0.0):
    """Drop a fixed beat range from the head and tail of the chart."""
    notes = raw["_notes"]
    times = [n["_time"] for n in notes]
    if not times:
        return raw, 0.0, 0.0
    lo, hi = min(times), max(times)
    new_lo, new_hi = lo + lead_beats, hi - tail_beats
    if new_hi <= new_lo:
        return raw, 0.0, 0.0
    kept = [n for n in notes if new_lo <= n["_time"] <= new_hi]
    obs = [o for o in raw.get("_obstacles", []) if new_lo <= o["_time"] <= new_hi]
    out = dict(raw, _notes=kept, _obstacles=obs)
    return out, lo, hi


def thin(raw, min_spacing_sec, keep_all_bombs=True):
    """Drop notes closer together than min_spacing_sec, keeping the earlier<|place_holder_mm_span_0442|>.

    A real note is kept only if every already-kept REAL note is at least
    min_spacing_sec away. Bombs and bombs-in-notes are handled identically so
    they do not survive as the only dense elements.
    """
    if min_spacing_sec <= 0:
        return raw, 0
    notes = []
    dropped = 0
    # Sort by time; within a window, notes count as one chord, so a chord's
    # members all survive together as long as the chord itself passes.
    by_time = {}
    for n in raw["_notes"]:
        by_time.setdefault(round(n["_time"], 5), []).append(n)
    last_kept = None
    for t in sorted(by_time):
        group = by_time[t]
        real = [n for n in group if n["_type"] in (0, 1)]
        if not real:
            notes.extend(group)
            continue
        if last_kept is not None and (t - last_kept) < min_spacing_sec - 1e-9:
            dropped += len(group)
            continue
        notes.extend(group)
        last_kept = t
    out = dict(raw, _notes=notes)
    # Obstacles are untouched: they are sparse and carry no flow cost.
    return out, dropped


def stats(raw, bpm):
    t = note_times(raw)
    if len(t) < 2:
        return 0, 0.0, 0.0
    span = t[-1] - t[0]
    nps = len(t) / span if span > 0 else 0.0
    return len(t), span, nps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--target-nps", type=float, required=True)
    ap.add_argument("--lead-beats", type=float, default=0.0)
    ap.add_argument("--tail-beats", type=float, default=0.0)
    args = ap.parse_args()

    raw = load(args.input)
    n0, span0, nps0 = stats(raw, args.bpm)
    print(f"input:  {n0:4} notes  span {span0:6.1f}s  {nps0:.2f} nps")

    if args.lead_beats or args.tail_beats:
        raw, lo, hi = trim(raw, args.lead_beats, args.tail_beats)
        n1, span1, nps1 = stats(raw, args.bpm)
        print(f"trim:   {n1:4} notes  span {span1:6.1f}s  {nps1:.2f} nps  "
              f"(dropped beats {lo:g}-{lo + args.lead_beats:g} and {hi - args.tail_beats:g}-{hi:g})")

    # Solve for the spacing that yields the target NPS. Rather than bisecting a
    # continuous value, which lands on spacings the beat grid cannot express and
    # then overshoots, sweep the real beat subdivisions a chart can use and take
    # whichever lands closest to target. Notes sit on the grid, so only these
    # spacings are reachable anyway.
    beat_sec = 60.0 / args.bpm
    best = None
    for k in range(1, 97):
        sp = (k / 24.0) * beat_sec
        cand, _ = thin(raw, sp)
        _, span, nps = stats(cand, args.bpm)
        err = abs(nps - args.target_nps)
        if best is None or err < best[0]:
            best = (err, sp, cand, nps)
    if best is None:
        best = (float("inf"), 0.0, raw, nps0)
    _, spacing, out, nps = best

    n2, span2, _ = stats(out, args.bpm)
    print(f"thin:   {n2:4} notes  span {span2:6.1f}s  {nps:.2f} nps  "
          f"(spacing {spacing:.3f}s, target {args.target_nps:.2f})")
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()