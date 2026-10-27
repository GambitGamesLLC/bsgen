<p align="center"><img src="assets/logo.svg" width="280" alt="bsgen"></p>

<p align="center">Beat Saber maps from any song, and a model that plays them.</p>

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

## Replays

Tick "Also make replays" and the player model plays each difficulty. The `.bsor` replay lands next
to the map; open both in a web replay viewer to watch it. Without the web page:

```
python scripts/generate.py --audio song.mp3 --stars 10 --replay
python scripts/play.py --map "out/song 10star.zip" --diff ExpertPlus
```

## Models

The [bsgen-1.0](https://huggingface.co/cata121/bsgen-1.0) models:
- map model: 65M parameters, ~40 min on one H100, then a 50 min fine-tune on an RTX 3060
- player model: 38M parameters, ~20 min on one H100

## Changelog

**1.1**
- new map model, better at low and mid stars
- better BPM timing, songs that change tempo too
- replays: a player model plays your map
- small fixes: note speed per star level, no bomb floods, a progress bar

**1.0**
- first release

## Star it

If bsgen made you a map you liked, a star on this repo helps a lot ⭐

## License

AGPL-3.0
