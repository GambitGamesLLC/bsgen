import argparse
import html
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlencode

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

import generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "web"
VIEWER = ROOT / "viewer"
STYLES = {"Auto": None, "Acc": "acc", "Tech": "tech", "Midspeed": "midspeed", "Speed": "speed"}
ARCS = {"Off": None, "Rare": 0.0, "Normal": 0.025, "Lots": 0.08}
VARIETY = {"Low": (0.0, 0.0), "Normal": (1.0, 0.0), "High": (1.0, 0.3)}
DIFFS = {"Easy": (3.5, "Easy"), "Normal": (4.5, "Normal"), "Hard": (5.5, "Hard"), "Expert": (7.5, "Expert"),
         "Expert+": (10.0, "ExpertPlus")}
MAX_LEVELS = 5
START_STARS = (10, 7.5, 5.5, 12.5, 4)
RED, BLUE = "#e8394b", "#2b8fd8"
GITHUB = "https://github.com/cata02222/bsgen"

CSS = f"""
.gradio-container {{ max-width: 760px !important; margin: auto; }}
#title {{ text-align: center; padding: 28px 0 4px; }}
#title h1 {{ font-size: 44px; font-weight: 800; letter-spacing: -1px; margin: 0; }}
#title h1 .r {{ color: {RED}; }}
#title h1 .b {{ color: {BLUE}; }}
#title p {{ opacity: .6; margin: 6px 0 0; }}
#go {{ background: linear-gradient(90deg, {RED}, {BLUE}); border: none; color: white; font-weight: 700; }}
footer {{ display: none !important; }}
#status .bar {{ height: 8px; border-radius: 4px; background: rgba(127,127,127,.2); overflow: hidden; }}
#status .fill {{ height: 100%; background: linear-gradient(90deg, {RED}, {BLUE}); transition: width .4s; }}
#status .busy .fill {{ width: 35% !important; animation: slide 1.2s ease-in-out infinite alternate; }}
@keyframes slide {{ from {{ margin-left: 0; }} to {{ margin-left: 65%; }} }}
#status .head {{ display: flex; justify-content: space-between; font-weight: 600; margin: 4px 0 8px; }}
#status .row {{ display: grid; grid-template-columns: 120px 1fr 48px; gap: 10px; align-items: center;
               font-size: 14px; margin: 6px 0; }}
#status .row span:last-child {{ text-align: right; opacity: .7; font-variant-numeric: tabular-nums; }}
#status .sum {{ font-size: 14px; opacity: .8; margin: 2px 0; }}
#status .err {{ color: {RED}; }}
"""
SCROLL = """
(...args) => {
  document.getElementById('status')?.scrollIntoView({behavior: 'smooth', block: 'center'});
  return args;
}
"""
DARK = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

state = {"model": None}
lock = threading.Lock()


def number(text, name):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise gr.Error(f"{name} should be a number, like 174.")


class Status:

    NAMES = {"ExpertPlus": "Expert+"}
    REPLAY_RATE = 0.06

    def __init__(self, names=None, levels=1, replay=False):
        self.t0 = time.time()
        self.phase, self.beats, self.summary, self.found = "Starting", {}, {}, ""
        for n in names or []:
            self.beats[n] = (0, 1)
        self.started = {}
        self.levels, self.replay, self.song = levels, replay, None

    def update(self, msg):
        if not msg:
            return
        if msg[0] == "phase":
            self.phase = msg[1]
        elif msg[0] == "beat":
            _, name, done, total = msg
            self.beats[name] = (done, total)
            self.phase = "Playing the map" if name.endswith(" replay") else "Making the map"
            self.started.setdefault(name, (time.time(), done / total))
        elif msg[0] == "log":
            line = msg[1]
            if line.startswith("timing:"):
                self.found = line.split(":", 1)[1].split(",")[0].strip()
            elif line.startswith("song:"):
                self.phase = "Making the map"
                self.song = float(line.split()[1])
            elif " replay: " in line:
                name, _, rest = line.partition(": ")
                self.summary[name] = rest
            elif "notes/s" in line:
                name, _, rest = line.partition(": ")
                if not rest:
                    name, rest = "map", line
                self.summary[name] = rest.split(", swings")[0]

    def label(self, name):
        base, _, replay = name.partition(" ")
        return f"{self.NAMES.get(base, base)} {replay}".strip()

    def fraction(self):
        parts = self.levels * (2 if self.replay else 1)
        return sum(d / max(t, 1e-9) for d, t in self.beats.values()) / max(parts, len(self.beats), 1)

    def left(self):
        now, worst, replays = time.time(), 0.0, 0.0
        for name, (d, total) in self.beats.items():
            if name not in self.started:
                return None
            t, f0 = self.started[name]
            f = d / total
            if f >= 1:
                continue
            if f <= f0:
                return None
            if name.endswith(" replay"):
                replays += (now - t) * (1 - f) / (f - f0)
            else:
                worst = max(worst, (now - t) * (1 - f) / (f - f0))
        if self.replay:
            waiting = self.levels - sum(n.endswith(" replay") for n in self.beats)
            if waiting and self.song is None:
                return None
            replays += waiting * (self.song or 0) * self.REPLAY_RATE
        return worst + replays

    def html(self, done=False, error=None):
        took = time.time() - self.t0
        f = 1.0 if done else self.fraction()
        if error:
            head = f'<span class="err">Stopped: {html.escape(error)}</span>'
        elif done:
            head = f"<span>Done in {took:.0f} s</span>"
        else:
            left = self.left()
            left = f" · about {left:.0f} s left" if left is not None else ""
            head = f"<span>{self.phase}…</span><span>{took:.0f} s{left}</span>"
        busy = not done and not error and not any(d for d, _ in self.beats.values())
        out = [f'<div class="head">{head}</div>',
               f'<div class="bar{" busy" if busy else ""}"><div class="fill" style="width:{f:.0%}"></div></div>']
        if self.found:
            out.append(f'<div class="sum">Song: {html.escape(self.found)}</div>')
        if len(self.beats) > 1 or (self.beats and not done):
            for name, (d, t) in self.beats.items():
                g = 1.0 if done else d / max(t, 1)
                out.append(f'<div class="row"><span>{html.escape(self.label(name))}</span>'
                           f'<div class="bar"><div class="fill" style="width:{g:.0%}"></div></div><span>{g:.0%}</span></div>')
        if done:
            for name, text in self.summary.items():
                out.append(f'<div class="sum">{html.escape(self.label(name))}: {html.escape(text)}</div>')
            out.append(f'<div class="sum">Enjoying bsgen? A star on <a href="{GITHUB}" target="_blank">GitHub</a> helps.</div>')
        return "".join(out)


def file_url(path):
    return "/gradio_api/file/" + quote(str(path))


def watch_choices(zip_path):
    from bsgen import beatmap, export
    z = Path(zip_path)
    with beatmap.open_map(z) as f:
        diffs = [d["difficulty"] for d in beatmap.read_info(f).difficulties]
    out = []
    for d in sorted(diffs, key=lambda d: -export.DIFF_RANK.get(d, 0)):
        bsor = z.with_name(f"{z.stem} {d}.bsor")
        label = Status.NAMES.get(d, d)
        out.append((f"{label}, played by the model" if bsor.exists() else f"{label}, map only",
                    "\n".join([str(z), d, str(bsor) if bsor.exists() else ""])))
    return out


def screen(value):
    if not value:
        return ""
    zip_path, difficulty, bsor = value.split("\n")
    q = {"url": file_url(zip_path), "noProxy": "true"}
    if bsor:
        q["replayURL"] = file_url(bsor)
    else:
        q.update(difficulty=difficulty, mode="Standard")
    src = html.escape(file_url(VIEWER / "index.html") + "?" + urlencode(q))
    return (f'<iframe src="{src}" allow="autoplay; fullscreen" allowfullscreen '
            f'style="width:100%;aspect-ratio:16/9;border:0;border-radius:8px;background:#000"></iframe>'
            f'<div class="sum"><a href="{src}" target="_blank">Open in a new tab</a></div>')


def build(ckpt, device):
    def make(song, mode, diff, count, *rest):
        stars, (style, arcs, extras, replay, variety, temp, cfg, njs, bpm, first_ms, seed, mapper, fmt) = \
            rest[:MAX_LEVELS], rest[MAX_LEVELS:]
        if not song:
            raise gr.Error("Add a song first.")
        if mode == "Stars":
            levels = [(x, None) for x in sorted(set(stars[:count]))]
        else:
            levels = [DIFFS[d] for d in DIFFS if d in diff]
            if not levels:
                raise gr.Error("Pick at least one difficulty.")
        folder = OUT / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True)
        argv = ["--audio", song, "--stars", *(f"{x:g}" for x, _ in levels),
                "--temp", f"{temp:g}", "--cfg", f"{cfg:g}", "--format", fmt, "--device", device, "--out", str(folder)]
        if mode != "Stars":
            argv += ["--difficulty", *(name for _, name in levels)]
        if ckpt:
            argv += ["--ckpt", ckpt]
        if (seed := number(seed, "Seed")) is not None:
            argv += ["--seed", str(int(seed))]
        if (njs := number(njs, "Note speed")) is not None:
            argv += ["--njs", f"{njs:g}"]
        if STYLES[style]:
            argv += ["--style", STYLES[style]]
        argv += ["--no-arcs"] if ARCS[arcs] is None else ["--arc-rate", f"{ARCS[arcs]:g}"]
        for part in ("Walls", "Bombs", "Lights"):
            if part not in extras:
                argv.append(f"--no-{part.lower()}")
        dry, xtc = VARIETY[variety]
        argv += ["--dry", f"{dry:g}", "--xtc", f"{xtc:g}"]
        bpm, first_ms = number(bpm, "Song BPM"), number(first_ms, "First beat")
        if bpm:
            argv += ["--bpm", f"{bpm:g}"]
            if first_ms is not None:
                argv += ["--offset-ms", f"{first_ms:g}"]
        if mapper and mapper != "Any":
            argv += ["--mapper", mapper]
        if replay:
            argv.append("--replay")
        args = generate.parser().parse_args(argv)

        q, result = queue.Queue(), {}

        def work():
            try:
                with lock:
                    if state["model"] is None:
                        q.put(("phase", "Loading the model"))
                        from bsgen import hub, sample
                        state["model"] = sample.load_model(ckpt or hub.map_model(), device)
                    if replay and state.get("player") is None:
                        from bsgen import botplay, hub
                        state["player"] = botplay.load_player(hub.player_model(), device)
                    q.put(("phase", "Reading the song"))
                    result["zip"] = generate.run(args, state["model"], log=lambda m: q.put(("log", m)),
                                                 progress=lambda name, done, total: q.put(("beat", name, done, total)),
                                                 player=state.get("player"))
            except Exception as e:
                result["error"] = str(e)
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        view = Status([name for _, name in levels] if mode != "Stars" else None, len(levels), bool(replay))
        no_watch = (gr.update(visible=False), gr.update(choices=[], value=None), "")
        yield gr.update(value=None, visible=False), view.html(), *no_watch
        shown = 0.0
        while True:
            try:
                msg = q.get(timeout=0.5)
            except queue.Empty:
                msg = ()
            if msg is None:
                break
            view.update(msg)
            if time.time() - shown > 0.25:
                shown = time.time()
                yield gr.update(value=None, visible=False), view.html(), *no_watch
        if "error" in result:
            yield gr.update(value=None, visible=False), view.html(error=result["error"]), *no_watch
            raise gr.Error(result["error"])
        z = Path(result["zip"])
        files = [str(z)] + sorted(str(p) for p in z.parent.iterdir()
                                  if p.suffix == ".bsor" and p.name.startswith(z.stem + " "))
        if has_viewer:
            choices = watch_choices(z)
            watch = (gr.update(visible=True), gr.update(choices=choices, value=choices[0][1]), screen(choices[0][1]))
        else:
            watch = no_watch
        yield gr.update(value=files, visible=True), view.html(done=True), *watch

    has_viewer = (VIEWER / "index.html").exists()
    mappers = ["Any"]
    try:
        from bsgen import hub, sample
        mappers += sorted(sample.read_checkpoint(ckpt or hub.map_model()).get("mappers") or [])
    except Exception:
        pass

    with gr.Blocks(title="bsgen", analytics_enabled=False) as demo:
        gr.HTML('<div id="title"><h1><span class="r">bs</span><span class="b">gen</span></h1>'
                "<p>Beat Saber maps from any song</p></div>")
        song = gr.Audio(label="Song", type="filepath", sources=["upload"])
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                mode = gr.Radio(["Difficulty", "Stars"], value="Difficulty", show_label=False, elem_id="mode")
                diff = gr.CheckboxGroup(list(DIFFS), value=["Expert+"], show_label=False,
                                        info="Pick more than one to get them all in one map")
                with gr.Group(visible=False) as star_box:
                    stars = [gr.Slider(1, 15, value=v, step=0.5, label=f"Stars, difficulty {i + 1}", show_label=False,
                                       visible=i == 0) for i, v in enumerate(START_STARS)]
                    with gr.Row():
                        more = gr.Button("+ another difficulty", size="sm")
                        fewer = gr.Button("− remove", size="sm", visible=False)
                count = gr.State(1)
            bpm = gr.Textbox(label="Song BPM", placeholder="e.g. 174", scale=1,
                             info="Leave empty to detect it, tempo changes included; a BPM given holds for the whole song")
        with gr.Row():
            style = gr.Radio(list(STYLES), value="Auto", label="Style")
            arcs = gr.Radio(list(ARCS), value="Normal", label="Arcs")
        extras = gr.CheckboxGroup(["Walls", "Bombs", "Lights"], value=["Walls", "Bombs", "Lights"], show_label=False)
        replay = gr.Checkbox(False, label="Also make replays",
                             info="The player model plays each difficulty: a .bsor replay to watch next to the map")
        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                variety = gr.Radio(list(VARIETY), value="High", label="Variety")
                fmt = gr.Radio(["v2", "v3"], value="v2", label="Format",
                               info="v2 opens everywhere; v3 keeps exact wall shapes")
            with gr.Row():
                temp = gr.Slider(0.6, 1.2, value=0.9, step=0.05, label="Creativity")
                cfg = gr.Slider(0.5, 3, value=1.5, step=0.1, label="Follow the settings")
            with gr.Row():
                njs = gr.Textbox(label="Note speed (NJS)", placeholder="auto",
                                 info="Leave empty for the usual speed of each difficulty")
                seed = gr.Textbox(label="Seed", placeholder="random",
                                  info="The same seed and settings make the same map")
            first_ms = gr.Textbox(label="First beat (ms)", placeholder="found automatically",
                                  info="Song time of the first beat, with a BPM")
            mapper = gr.Dropdown(mappers, value="Any", label="Map like")
        go = gr.Button("Generate map", elem_id="go", size="lg")
        status = gr.HTML('<div class="sum">Add a song and press Generate.</div>', elem_id="status", padding=True)
        out = gr.File(label="Map (.zip) and replays (.bsor)", file_count="multiple", height=140, visible=False)
        with gr.Group(visible=False) as watch_box:
            watch = gr.Dropdown([], label="Watch", interactive=True)
            view = gr.HTML()
        watch.change(screen, watch, view)
        def resize(n):
            n = max(1, min(MAX_LEVELS, n))
            return [n, *(gr.update(visible=i < n) for i in range(MAX_LEVELS)),
                    gr.update(visible=n < MAX_LEVELS), gr.update(visible=n > 1)]

        mode.change(lambda m: (gr.update(visible=m == "Difficulty"), gr.update(visible=m == "Stars")), mode, [diff, star_box])
        more.click(lambda n: resize(n + 1), count, [count, *stars, more, fewer])
        fewer.click(lambda n: resize(n - 1), count, [count, *stars, more, fewer])
        go.click(make, [song, mode, diff, count, *stars, style, arcs, extras, replay, variety, temp, cfg, njs, bpm, first_ms, seed, mapper, fmt],
                 [out, status, watch_box, watch, view], js=SCROLL)
    return demo


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="map model checkpoint (default: the released model, downloaded once)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true", help="also make a public link")
    args = ap.parse_args()
    theme = gr.themes.Base(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.red,
                           neutral_hue=gr.themes.colors.zinc, radius_size=gr.themes.sizes.radius_md,
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    demo = build(args.ckpt, args.device)
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port, share=args.share,
                                                   theme=theme, css=CSS, js=DARK, allowed_paths=[str(OUT), str(VIEWER)])


if __name__ == "__main__":
    main()
