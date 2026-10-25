import argparse
import os
import queue
import threading
import uuid
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

import generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "web"
STYLES = {"Auto": None, "Acc": "acc", "Tech": "tech", "Midspeed": "midspeed", "Speed": "speed"}
ARCS = {"Off": None, "Rare": 0.0, "Normal": 0.025, "Lots": 0.08}
VARIETY = {"Low": (0.0, 0.0), "Normal": (1.0, 0.0), "High": (1.0, 0.3)}
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


def build(ckpt, device):
    def make(song, stars, style, arcs, extras, variety, temp, cfg, njs, bpm, first_ms, seed, mapper, fmt):
        if not song:
            raise gr.Error("Add a song first.")
        folder = OUT / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True)
        argv = ["--audio", song, "--stars", f"{stars:g}", "--seed", str(int(seed)), "--temp", f"{temp:g}",
                "--cfg", f"{cfg:g}", "--njs", f"{njs:g}", "--format", fmt, "--device", device, "--out", str(folder)]
        if ckpt:
            argv += ["--ckpt", ckpt]
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
        args = generate.parser().parse_args(argv)

        lines, q, result = [], queue.Queue(), {}

        def work():
            try:
                with lock:
                    if state["model"] is None:
                        q.put("loading the model…")
                        from bsgen import hub, sample
                        state["model"] = sample.load_model(ckpt or hub.map_model(), device)
                    result["zip"] = generate.run(args, state["model"], log=q.put)
            except Exception as e:
                result["error"] = str(e)
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (line := q.get()) is not None:
            if not line.startswith(("beat ", "wrote ")):
                lines.append(line)
                yield None, "\n".join(lines[-8:])
        if "error" in result:
            raise gr.Error(result["error"])
        yield str(result["zip"]), "\n".join(lines[-8:] + ["done", "", f"enjoying bsgen? a star on GitHub helps: {GITHUB}"])

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
        with gr.Row():
            stars = gr.Slider(1, 15, value=10, step=0.5, label="Stars", scale=3)
            bpm = gr.Textbox(label="Song BPM", placeholder="e.g. 174", scale=1,
                             info="Enter it for the best map, or leave empty to detect it")
        with gr.Row():
            style = gr.Radio(list(STYLES), value="Auto", label="Style")
            arcs = gr.Radio(list(ARCS), value="Normal", label="Arcs")
        extras = gr.CheckboxGroup(["Walls", "Bombs", "Lights"], value=["Walls", "Bombs", "Lights"], show_label=False)
        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                variety = gr.Radio(list(VARIETY), value="High", label="Variety")
                fmt = gr.Radio(["v2", "v3"], value="v2", label="Format",
                               info="v2 opens everywhere; v3 keeps exact wall shapes")
            with gr.Row():
                temp = gr.Slider(0.6, 1.2, value=0.9, step=0.05, label="Creativity")
                cfg = gr.Slider(0.5, 3, value=1.5, step=0.1, label="Follow the settings")
            with gr.Row():
                njs = gr.Slider(10, 24, value=18, step=1, label="Note speed (NJS)")
                seed = gr.Number(0, label="Seed", precision=0)
            first_ms = gr.Textbox(label="First beat (ms)", placeholder="found automatically",
                                  info="Song time of the first beat, with a BPM")
            mapper = gr.Dropdown(mappers, value="Any", label="Map like")
        go = gr.Button("Generate map", elem_id="go", size="lg")
        out = gr.File(label="Map (.zip)", height=90)
        log = gr.Textbox(label="Progress", lines=4, max_lines=8, interactive=False)
        go.click(make, [song, stars, style, arcs, extras, variety, temp, cfg, njs, bpm, first_ms, seed, mapper, fmt],
                 [out, log])
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
                                                   theme=theme, css=CSS, js=DARK, allowed_paths=[str(OUT)])


if __name__ == "__main__":
    main()
