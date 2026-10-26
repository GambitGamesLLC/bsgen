from pathlib import Path

REPO = "cata121/bsgen-1.0"


def fetch(*names):
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(REPO, n)) for n in names]


LOCAL_MAP = Path(__file__).resolve().parents[1] / "models" / "map.pt"


def map_model():
    if LOCAL_MAP.exists():
        return LOCAL_MAP
    return fetch("map.safetensors", "map.json")[0]


LOCAL_PLAYER = LOCAL_MAP.with_name("player.pt")


def player_model():
    if LOCAL_PLAYER.exists():
        return LOCAL_PLAYER
    return fetch("player.safetensors", "player.json")[0]


def timing_model():
    return fetch("timing.safetensors", "octave.npz")[0].parent
