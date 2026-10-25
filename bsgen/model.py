import torch
from torch import nn
from torch.nn import functional as F

from .data import MAX_TOKENS, N_FRAMES, OVERVIEW_BARS


class Generator(nn.Module):
    def __init__(self, vocab: int, d=384, enc_layers=6, dec_layers=6, heads=6, n_mels=80, dropout=0.1,
                 max_tokens=MAX_TOKENS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_mels, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, d, 4, stride=2, padding=1))
        self.enc_pos = nn.Parameter(torch.randn(N_FRAMES // 2, d) * 0.02)
        self.ov_in = nn.Linear(2 * n_mels, d)
        self.ov_rel = nn.Embedding(2 * OVERVIEW_BARS - 1, d)
        nn.init.normal_(self.ov_rel.weight, std=0.02)
        layer = lambda: dict(d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=dropout,
                             activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer()), enc_layers, enable_nested_tensor=False)
        self.tok = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.dec_pos = nn.Parameter(torch.randn(max_tokens, d) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer()), dec_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.out.weight = self.tok.weight

    def encode(self, mel, overview, rel):
        x = self.stem(mel.transpose(1, 2)).transpose(1, 2)
        x = x + self.enc_pos[: x.shape[1]]
        o = self.ov_in(overview) + self.ov_rel(rel)
        return self.enc(torch.cat([x, o], 1))

    def decode(self, memory, tk):
        L = tk.shape[1]
        x = self.tok(tk) + self.dec_pos[:L]
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tk.device)
        return self.out(self.norm(self.dec(x, memory, tgt_mask=causal, tgt_is_causal=True)))

    def decode_step(self, memory, tk, cache):
        past = cache.get("len", 0)
        x = self.tok(tk) + self.dec_pos[past:past + tk.shape[1]]
        for i, layer in enumerate(self.dec.layers):
            c = cache.setdefault(i, {})
            sa, ca = layer.self_attn, layer.multihead_attn
            q, k, v = self._heads(F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias), sa.num_heads, 3)
            if "k" in c:
                k, v = torch.cat([c["k"], k], 2), torch.cat([c["v"], v], 2)
            c["k"], c["v"] = k, v
            x = x + sa.out_proj(self._merge(F.scaled_dot_product_attention(q, k, v, is_causal=past == 0 and tk.shape[1] > 1)))
            d = ca.embed_dim
            if "mk" not in c:
                mk, mv = self._heads(F.linear(memory, ca.in_proj_weight[d:], ca.in_proj_bias[d:]), ca.num_heads, 2)
                c["mk"], c["mv"] = mk.expand(x.shape[0], -1, -1, -1), mv.expand(x.shape[0], -1, -1, -1)
            q, = self._heads(F.linear(layer.norm2(x), ca.in_proj_weight[:d], ca.in_proj_bias[:d]), ca.num_heads, 1)
            x = x + ca.out_proj(self._merge(F.scaled_dot_product_attention(q, c["mk"], c["mv"])))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        cache["len"] = past + tk.shape[1]
        return self.out(self.norm(x[:, -1]))

    @staticmethod
    def _heads(y, h, n):
        B, T, _ = y.shape
        return y.view(B, T, n, h, -1).permute(2, 0, 3, 1, 4).unbind(0)

    @staticmethod
    def _merge(y):
        B, h, T, dh = y.shape
        return y.transpose(1, 2).reshape(B, T, h * dh)

    def forward(self, audio, tk):
        return self.decode(self.encode(*audio), tk)

    def loss(self, audio, tk, mask):
        logits = self(audio, tk[:, :-1])
        m = mask[:, 1:]
        ce = F.cross_entropy(logits.transpose(1, 2), tk[:, 1:], reduction="none")
        return (ce * m).sum() / m.sum().clamp(min=1)
