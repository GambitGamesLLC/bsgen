from pathlib import Path

import numpy as np
import torch
from torch import nn

CHUNK = 2048
BPM_RANGE = (60.0, 400.0)
SUBDIVISIONS = tuple(sorted({k / n for n in range(2, 9) for k in range(1, n)}))


def scale(logmel):
    return logmel.astype(np.float32) / 40.0 + 1.0


class BeatNet(nn.Module):
    def __init__(self, n_mels=80, d=256, layers=4, heads=4, chunk=CHUNK, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_mels, d, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(d, d, 4, stride=2, padding=1), nn.GELU())
        self.pos = nn.Parameter(torch.randn(chunk // 2, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, "gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2)

    def forward(self, mel):
        x = self.stem(mel.transpose(1, 2)).transpose(1, 2)
        x = self.enc(x + self.pos[: x.shape[1]])
        return self.out(self.norm(x)).flatten(1)


@torch.no_grad()
def activation(model, logmel, dev="cuda", batch=16):
    x = scale(logmel)
    n, hop = len(x), CHUNK // 2
    starts = list(range(0, max(n - CHUNK, 0) + 1, hop))
    if starts[-1] + CHUNK < n:
        starts.append(n - CHUNK)
    pad = max(CHUNK - n, 0)
    x = np.pad(x, ((0, pad), (0, 0)), constant_values=-1.0)
    w = np.hanning(CHUNK + 2)[1:-1].astype(np.float32)
    acc, wsum = np.zeros(len(x), np.float32), np.zeros(len(x), np.float32)
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        mel = torch.from_numpy(np.stack([x[a:a + CHUNK] for a in s])).to(dev)
        p = torch.sigmoid(model(mel).float()).cpu().numpy()
        for a, pa in zip(s, p):
            acc[a:a + CHUNK] += pa * w
            wsum[a:a + CHUNK] += w
    return (acc / np.maximum(wsum, 1e-6))[:n]


def _score(act, period, lo, hi, bins=None):
    bins = bins or max(8, int(round(period * 4)))
    t = np.arange(lo, hi)
    k = ((t % period) / period * bins).astype(int) % bins
    prof = np.bincount(k, act[lo:hi], bins) / np.maximum(np.bincount(k, None, bins), 1)
    r = max(1, int(round(bins / period)))
    prof = np.convolve(np.concatenate([prof[-r:], prof, prof[:r]]), np.ones(2 * r + 1) / (2 * r + 1), "valid")
    off = np.max([np.roll(prof, -int(round(f * bins))) for f in SUBDIVISIONS], 0)
    val = prof - off
    j = int(np.argmax(val))
    return float(val[j]), (j + 0.5) / bins * period


def _refine(act, period, phase, lo, hi):
    for _ in range(3):
        k = np.arange(np.ceil((lo - phase) / period), np.floor((hi - 1 - phase) / period) + 1)
        g = phase + k * period
        r = max(1, int(period / 4))
        idx, wts = [], []
        for c in g:
            a, b = max(int(c) - r, 0), min(int(c) + r + 1, len(act))
            seg = act[a:b]
            if len(seg) < 3:
                idx.append(c), wts.append(0.0)
                continue
            j = int(np.argmax(seg))
            if 0 < j < len(seg) - 1:
                d = seg[j - 1] - 2 * seg[j] + seg[j + 1]
                off = 0.5 * (seg[j - 1] - seg[j + 1]) / d if d < 0 else 0.0
            else:
                off = 0.0
            idx.append(a + j + off), wts.append(float(seg[j]))
        idx, wts = np.array(idx), np.array(wts) ** 2
        if wts.sum() <= 0 or len(k) < 4:
            break
        A = np.stack([k, np.ones_like(k)], 1) * np.sqrt(wts)[:, None]
        sol = np.linalg.lstsq(A, idx * np.sqrt(wts), rcond=None)[0]
        period, phase = float(sol[0]), float(sol[1])
    return period, phase


def fit_grid(act, fps, span=None, seg_sec=40.0):
    lo, hi = span or (0, len(act))
    seg = int(seg_sec * fps)
    mid = (lo + hi) // 2
    slo, shi = max(lo, mid - seg // 2), min(hi, mid + seg // 2)
    min_p, max_p = 60 * fps / BPM_RANGE[1], 60 * fps / BPM_RANGE[0]
    x = act[slo:shi] - act[slo:shi].mean()
    f = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[: int(max_p) + 2]
    lags = np.arange(max(int(min_p), 1), int(max_p) + 1)
    peaks = [int(lag) for lag in lags[1:-1] if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]]
    peaks = sorted(peaks, key=lambda lag: -ac[lag])[:6]
    best = (-np.inf, None, None)
    for p0 in peaks:
        for rel in np.linspace(-0.04, 0.04, 401):
            s, ph = _score(act, p0 * (1 + rel), slo, shi)
            if s > best[0]:
                best = (s, p0 * (1 + rel), ph)
    s, period, phase = best
    period, phase = _refine(act, period, phase, slo, shi)
    a, b = slo, shi
    while a > lo or b < hi:
        w = b - a
        a, b = max(lo, a - w // 2), min(hi, b + w // 2)
        period, phase = _refine(act, period, phase, a, b)
    phase = phase % period
    return 60 * fps / period, phase / fps, s


OCTAVES = (0.5, 1.0, 2.0)


def _marks(act, period, phase, lo, hi):
    k = np.arange(np.ceil((lo - phase) / period), np.floor((hi - 2 - phase) / period) + 1)
    out = []
    for f in (0.0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4):
        g = np.round(phase + (k + f) * period).astype(int)
        g = g[(g >= 1) & (g < len(act) - 1)]
        out.append(float(np.max([act[g - 1], act[g], act[g + 1]], 0).mean()) if len(g) else 0.0)
    return out


def octave_features(act, fps, bpm, first_sec, span=None):
    lo, hi = span or (0, len(act))
    period, phase = 60 * fps / bpm, first_sec * fps
    feats, cands = [], []
    for m in OCTAVES:
        p = period / m
        phases = [phase + j * period for j in range(max(1, int(round(1 / m))))]
        best = max(phases, key=lambda ph: _marks(act, p, ph, lo, hi)[0])
        mk = _marks(act, p, best, lo, hi)
        feats += [np.log2(bpm * m)] + mk + [mk[0] - max(mk[1:])]
        cands.append((bpm * m, (best % p) / fps))
    return np.array(feats, np.float32), cands


def beat_grid(bpm, first_sec, duration):
    spb = 60.0 / bpm
    beat_times = np.arange(0, duration / spb + 2) * spb
    return beat_times, (first_sec / spb) % 1.0


@torch.no_grad()
def octave_class(trees, x):
    raw = trees["baseline"].astype(np.float64).copy()
    for k, i in enumerate(trees["roots"]):
        while not trees["leaf"][i]:
            v = x[trees["feature"][i]]
            left = trees["missing_left"][i] if np.isnan(v) else v <= trees["threshold"][i]
            i = trees["left"][i] if left else trees["right"][i]
        raw[trees["cls"][k]] += trees["value"][i]
    return int(trees["classes"][np.argmax(raw)])


def detect(logmel, fps, root, dev="cuda"):
    from safetensors.torch import load_file
    root = Path(root)
    model = BeatNet().to(dev)
    model.load_state_dict(load_file(root / "timing.safetensors", device=str(dev)))
    model.eval()
    act = activation(model, logmel, dev)
    bpm, first, _ = fit_grid(act, fps)
    feats, cands = octave_features(act, fps, bpm, first)
    bpm, first = cands[octave_class(dict(np.load(root / "octave.npz")), feats)]
    return float(bpm), float(first)
