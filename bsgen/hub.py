from pathlib import Path

REPO = "cata121/bsgen-1.0"


def fetch(*names):
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(REPO, n)) for n in names]


def map_model():
    return fetch("map.safetensors", "map.json")[0]


def timing_model():
    return fetch("timing.safetensors", "octave.npz")[0].parent
