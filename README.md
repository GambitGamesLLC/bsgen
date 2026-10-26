<p align="center"><img src="assets/logo.svg" width="280" alt="bsgen"></p>

<p align="center">bsgen is an experimental Beat Saber map generator. It uses the <a href="https://huggingface.co/cata121/bsgen-1.0">bsgen-1.0</a> models.<br>
The map model has 65 million parameters: about 40 minutes on a single H100, then a 50 minute fine-tune on an RTX 3060.<br>
The player model that plays the maps has 38 million, trained in about 20 minutes on a single H100.</p>

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

## Changelog

**1.1**
- new map model, fine-tuned so every star level gets its share: easy maps stop filling every beat,
  mid stars are closer to human density
- songs that change tempo are timed right when the BPM box is left empty
- replays, played by a new player model
- note speed defaults to what ranked maps use at that star level
- no more maps flooded with bombs
- a progress panel with the time left

**1.0**
- first release

## Star it

If bsgen made you a map you liked, a star on this repo helps a lot ⭐

## License

AGPL-3.0
