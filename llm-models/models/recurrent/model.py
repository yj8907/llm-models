import torch
import torch.nn as nn
import torch.nn.functional as F
from span_attn import SpanRefreshedAttention


# ============================================================================
# Synthetic tasks, reimplemented from the Mamba paper (Gu & Dao 2023, sec 4.1.1)
# These generators are NOT in state-spaces/mamba; they originate in the
# S4/H3/Safari line of work. Faithful to the task definitions, not byte-identical.
# ============================================================================

def make_induction_heads(batch, L, V, device="cpu", generator=None):
    """Associative recall. content tokens 0..V-1, special trigger = V.
    A trigger appears once early, followed by a random 'value' token; the
    trigger appears again as the final token, and the model must predict the
    value. Only the last position is scored. Returns x:(B,L), y:(B,) labels."""
    special = V
    x = torch.randint(0, V, (batch, L), device=device, generator=generator)
    x[:, -1] = special                                   # query trigger at the end
    pos = torch.randint(1, L - 2, (batch,), device=device, generator=generator)  # pos+1 <= L-2
    rows = torch.arange(batch, device=device)
    y = x[rows, pos + 1].clone()                         # value to memorize
    x[rows, pos] = special                               # place the early trigger
    return x, y


def make_selective_copy(batch, num_data, data_len, C, device="cpu", generator=None):
    """Copy num_data scattered data tokens (ids 1..C) in order, ignoring blanks
    (id 0), at a trailing block of num_data marker tokens (id C+1).
    The data positions are random per-sample -> requires content selection.
    Returns x:(B,L), y:(B,L) with -100 on non-scored positions. L=data_len+num_data."""
    blank, marker = 0, C + 1
    L = data_len + num_data
    x = torch.zeros(batch, L, dtype=torch.long, device=device)
    y = torch.full((batch, L), -100, dtype=torch.long, device=device)
    x[:, data_len:] = marker                             # trailing marker block
    for b in range(batch):
        positions = torch.randperm(data_len, generator=generator)[:num_data].sort().values
        data = torch.randint(1, C + 1, (num_data,), generator=generator)
        x[b, positions] = data
        y[b, data_len:] = data                           # output in positional order
    return x, y


# ============================================================================
# Tiny model: token+pos embedding -> N x (attn block + MLP block) -> head
# ============================================================================

class StandardAttention(nn.Module):
    """Plain causal multi-head attention, for a baseline comparison."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        sh = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        return self.out(o.transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mixer):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mix = (SpanRefreshedAttention(d_model, n_heads) if mixer == "span"
                    else StandardAttention(d_model, n_heads))
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, max_len, d_model=64, n_heads=2, n_layers=2, mixer="span"):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mixer) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        L = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.lnf(h))
