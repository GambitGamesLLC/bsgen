<p align="center"><img src="assets/logo.svg" width="280" alt="bsgen"></p>

<p align="center">bsgen is an experimental Beat Saber map generator. It uses the <a href="https://huggingface.co/cata121/bsgen-1.0">bsgen-1.0</a> model.</p>

<p align="center"><a href="https://colab.research.google.com/github/cata02222/bsgen/blob/main/colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></p>

<p align="center">
<img src="assets/clip-a.webp" width="49%"> <img src="assets/clip-b.webp" width="49%">
<img src="assets/clip-c.webp" width="49%"> <img src="assets/clip-d.webp" width="49%">
</p>

## Run it

Easiest: the Colab button above. Run both cells, open the link.

On your own machine (Python 3.10+ and ffmpeg):

```
git clone https://github.com/cata02222/bsgen
cd bsgen
pip install -e .
python scripts/webui.py
```

## Web UI

<img src="assets/webui.png" width="420" alt="the web page">

Drop a song in, pick the stars, done. 65 M parameters: under a minute per song on a GPU, works
on CPU too, just slower.

## Star it

If bsgen made you a map you liked, a star on this repo helps a lot ⭐

## License

AGPL-3.0
