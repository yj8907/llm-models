import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Reference (single head, no batch) -- this is your turn-1 function, but written
# to operate on already-projected Qo, Qi, Ki, Vi + Wk, Wv so it shares EXACTLY
# the same math as the vectorized path below. Used only to validate correctness.
# ----------------------------------------------------------------------------
def span_refreshed_attention_ref(Qo, Qi, Ki, Vi, Wk, Wv, include_query_token=True):
    L, d = Qo.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros(L, d, dtype=Qo.dtype)
    for j in range(L):
        k_ij = torch.zeros(j + 1, d, dtype=Qo.dtype)
        v_ij = torch.zeros(j + 1, d, dtype=Qo.dtype)
        for i in range(j + 1):
            hi = j if include_query_token else max(i, j - 1)
            sl = slice(i, hi + 1)
            a = (Qi[i] @ Ki[sl].T) * scale
            b = F.softmax(a, dim=-1)
            u = b @ Vi[sl]                 # refreshed repr of i w.r.t. j
            k_ij[i] = u @ Wk
            v_ij[i] = u @ Wv
        s = (Qo[j] @ k_ij.T) * scale
        p = F.softmax(s, dim=-1)
        out[j] = p @ v_ij
    return out


# ----------------------------------------------------------------------------
# Vectorized, batched, multi-head. Loops only over j (the outer query position),
# everything else is batched. Still O(L^3) work overall -- that is intrinsic to
# query-position-dependent keys/values -- but no Python loop over i or m.
# ----------------------------------------------------------------------------
def span_refreshed_attention_batched(Qo, Qi, Ki, Vi, Wk, Wv, include_query_token=True):
    # Qo,Qi,Ki,Vi: (B, H, L, d).  Wk,Wv: (H, d, d).
    B, H, L, d = Qo.shape
    scale = 1.0 / math.sqrt(d)
    device, dtype = Qo.device, Qo.dtype

    # Inner score matrix S[...,i,m] = qi_i . ki_m   -> (B,H,L,L)
    S = torch.einsum("bhid,bhmd->bhim", Qi, Ki) * scale

    idx = torch.arange(L, device=device)
    causal_im = idx[None, :] >= idx[:, None]            # m >= i  -> (L,L)

    out = torch.zeros(B, H, L, d, device=device, dtype=dtype)
    for j in range(L):
        if include_query_token:
            upper = torch.full((L,), j, device=device)          # m <= j for every row
        else:
            upper = torch.clamp(idx, min=j - 1)                 # m <= max(i, j-1) per row
        valid = causal_im & (idx[None, :] <= upper[:, None])    # i<=m<=upper[i] -> (L,L)
        # inner softmax over m for every row i (rows i>j are computed but unused)
        Sj = S.masked_fill(~valid[None, None], float("-inf"))
        Bw = torch.softmax(Sj, dim=-1)                  # (B,H,L,L)
        Bw = torch.nan_to_num(Bw, nan=0.0)              # rows with no valid m (none here, but safe)
        U = torch.einsum("bhim,bhmd->bhid", Bw, Vi)     # refreshed repr u_{i,j} -> (B,H,L,d)

        Kj = torch.einsum("bhid,hde->bhie", U, Wk)      # refreshed outer keys   (B,H,L,d)
        Vj = torch.einsum("bhid,hde->bhie", U, Wv)      # refreshed outer values (B,H,L,d)

        Kj += Ki
        Vj += Vi

        qoj = Qo[:, :, j, :]                            # (B,H,d)
        so = torch.einsum("bhd,bhid->bhi", qoj, Kj) * scale     # (B,H,L)
        outer_valid = (idx <= j)                        # i<=j
        so = so.masked_fill(~outer_valid[None, None], float("-inf"))
        p = torch.softmax(so, dim=-1)                   # (B,H,L)
        out[:, :, j, :] = torch.einsum("bhi,bhid->bhd", p, Vj)
    return out


class SpanRefreshedAttention(nn.Module):
    """Multi-head span-refreshed attention as a drop-in sequence mixer."""
    def __init__(self, d_model, n_heads, include_query_token=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.include_query_token = include_query_token
        self.Wqo = nn.Linear(d_model, d_model, bias=False)   # outer query
        self.Wqi = nn.Linear(d_model, d_model, bias=False)   # inner query (token i attends span)
        self.Wki = nn.Linear(d_model, d_model, bias=False)   # inner key
        self.Wvi = nn.Linear(d_model, d_model, bias=False)   # inner value
        self.Wk = nn.Parameter(torch.empty(self.h, self.dh, self.dh))  # refreshed->outer key
        self.Wv = nn.Parameter(torch.empty(self.h, self.dh, self.dh))  # refreshed->outer value
        self.out = nn.Linear(d_model, d_model, bias=False)
        for p in (self.Wk, self.Wv):
            nn.init.normal_(p, std=0.02)

    def _split(self, t):
        B, L, _ = t.shape
        return t.view(B, L, self.h, self.dh).transpose(1, 2)  # (B,H,L,dh)

    def forward(self, x):                                     # x: (B,L,d_model)
        B, L, _ = x.shape
        Qo, Qi = self._split(self.Wqo(x)), self._split(self.Wqi(x))
        Ki, Vi = self._split(self.Wki(x)), self._split(self.Wvi(x))
        o = span_refreshed_attention_batched(Qo, Qi, Ki, Vi, self.Wk, self.Wv,
                                             self.include_query_token)
        o = o.transpose(1, 2).reshape(B, L, -1)
        return self.out(o)


if __name__ == "__main__":
    torch.manual_seed(0)
    # Equivalence check: batched(B=1,H=1) must equal the reference loop.
    L, d = 7, 5
    Qo, Qi, Ki, Vi = (torch.randn(L, d, dtype=torch.float64) for _ in range(4))
    Wk = torch.randn(d, d, dtype=torch.float64)
    Wv = torch.randn(d, d, dtype=torch.float64)
    for inc in (True, False):
        ref = span_refreshed_attention_ref(Qo, Qi, Ki, Vi, Wk, Wv, inc)
        bat = span_refreshed_attention_batched(
            Qo[None, None], Qi[None, None], Ki[None, None], Vi[None, None],
            Wk[None], Wv[None], inc)[0, 0]
        err = (ref - bat).abs().max().item()
        print(f"include_query_token={inc}: max|ref-batched| = {err:.2e}")
        assert err < 1e-9, "vectorized path disagrees with reference!"
    print("OK: vectorized batched attention matches the reference loop.")
