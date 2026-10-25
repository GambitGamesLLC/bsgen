<p align="center"><img src="assets/logo.svg" width="280" alt="bsgen"></p>

bsgen is an experimental Beat Saber map generator. It uses the [bsgen-1.0](https://huggingface.co/cata121/bsgen-1.0) model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cata02222/bsgen/blob/main/colab.ipynb)

## Run locally

Needs Python 3.10+ and ffmpeg.

```
git clone https://github.com/cata02222/bsgen
cd bsgen
pip install -e .
python scripts/webui.py
```

Or from the command line: `python scripts/generate.py --audio song.mp3 --stars 10` (`--help` for options).

## License

AGPL-3.0
