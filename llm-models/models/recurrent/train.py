import argparse, time, torch
import torch.nn.functional as F
from model import TinyLM, make_induction_heads, make_selective_copy

torch.set_num_threads(1)


def run_induction(mixer, L, V, d_model, n_heads, n_layers, batch, steps, lr, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    model = TinyLM(V + 1, L, d_model, n_heads, n_layers, mixer)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[induction|{mixer}] params={n_params/1e3:.1f}K  L={L} V={V} "
          f"d={d_model} H={n_heads} layers={n_layers} batch={batch}")
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_induction_heads(batch, L, V, generator=g)
        logits = model(x)[:, -1, :]                       # score only last position
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 8) == 0 or step == 1:
            with torch.no_grad():
                xe, ye = make_induction_heads(256, L, V, generator=g)
                acc = (model(xe)[:, -1, :].argmax(-1) == ye).float().mean().item()
            print(f"  step {step:4d}  loss {loss.item():.3f}  eval_acc {acc:.3f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")
    return acc


def run_selcopy(mixer, num_data, data_len, C, d_model, n_heads, n_layers, batch, steps, lr, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    L = data_len + num_data

    model = TinyLM(C + 2, L, d_model, n_heads, n_layers, mixer)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[selcopy|{mixer}] params={n_params/1e3:.1f}K  num_data={num_data} "
          f"data_len={data_len} L={L} C={C} d={d_model} H={n_heads} layers={n_layers} batch={batch}")
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_selective_copy(batch, num_data, data_len, C, generator=g)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, C + 2), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 8) == 0 or step == 1:
            with torch.no_grad():
                xe, ye = make_selective_copy(256, num_data, data_len, C, generator=g)
                pred = model(xe).argmax(-1)
                mask = ye != -100
                tok_acc = (pred[mask] == ye[mask]).float().mean().item()
                seq_ok = ((pred == ye) | ~mask).all(dim=1).float().mean().item()
            print(f"  step {step:4d}  loss {loss.item():.3f}  tok_acc {tok_acc:.3f}  "
                  f"seq_acc {seq_ok:.3f}  ({(time.time()-t0)/step*1000:.0f} ms/step)")
    return tok_acc, seq_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["induction", "selcopy"], required=True)
    ap.add_argument("--mixer", choices=["span", "standard"], default="span")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    # induction
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--V", type=int, default=8)
    # selcopy
    ap.add_argument("--num_data", type=int, default=4)
    ap.add_argument("--data_len", type=int, default=12)
    ap.add_argument("--C", type=int, default=8)
    a = ap.parse_args()
    if a.task == "induction":
        run_induction(a.mixer, a.L, a.V, a.d_model, a.n_heads, a.n_layers,
                      a.batch, a.steps, a.lr, a.seed)
    else:
        run_selcopy(a.mixer, a.num_data, a.data_len, a.C, a.d_model, a.n_heads,
                    a.n_layers, a.batch, a.steps, a.lr, a.seed)
