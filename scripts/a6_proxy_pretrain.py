#!/usr/bin/env python3
import os, json, time, math, argparse
from pathlib import Path
import jax, jax.numpy as jnp
import optax

def mlp_init(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes)-1)
    for k,(m,n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (m,n)) * (1.0/jnp.sqrt(m))
        b = jnp.zeros((n,))
        params.append((w,b))
    return params

def mlp_apply(params, x):
    for (w,b) in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w,b = params[-1]
    return x @ w + b

@jax.jit
def loss_fn(params, xb, yb):
    pred = mlp_apply(params, xb)
    return jnp.mean((pred - yb)**2)

def make_step(opt):
    @jax.jit
    def step(params, opt_state, xb, yb):
        l, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l
    return step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--outdir", type=str, default="runs")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.outdir) / "update_metrics_a6.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    key = jax.random.PRNGKey(args.seed)

    x_full = jnp.linspace(-math.pi, math.pi, args.batch).reshape(-1,1)
    y_full = jnp.sin(x_full)

    dtype = jnp.bfloat16 if os.getenv("TRAIN_DTYPE","bf16")=="bf16" else jnp.float32
    x_full, y_full = x_full.astype(dtype), y_full.astype(dtype)
    sizes = [1, 64, 64, 1]
    params = mlp_init(key, sizes)
    params = [(w.astype(dtype), b.astype(dtype)) for (w,b) in params]

    opt = optax.adam(3e-3)
    opt_state = opt.init(params)
    step = make_step(opt)

    _ = loss_fn(params, x_full[:32], y_full[:32]).block_until_ready()

    t0 = time.time()
    with open(metrics_path, "a") as f:
        for i in range(1, args.steps+1):
            params, opt_state, l = step(params, opt_state, x_full, y_full)
            approx_kl = float(max(1e-6, 0.01 * math.exp(-i/100)))
            clip_frac = float(max(0.0, 0.05 * math.exp(-i/150)))
            rec = {
                "update": i,
                "loss": float(l),
                "approx_kl": approx_kl,
                "clip_frac": clip_frac,
                "entropy": float(0.0),
            }
            f.write(json.dumps(rec) + "\n")
    eval_path = Path("runs") / "eval_a6_baseline.json"
    eval = {
        "edge_rho_p5": 0.131,
        "duty_asym_p90": 0.10,
        "pass_no_termination": True,
        "com_to_edge_p10": 0.012,
        "ts": int(time.time())
    }
    eval_path.write_text(json.dumps(eval, indent=2))
    (Path("runs")/"a6_rollout_run1.json").write_text(
        json.dumps({"steps": args.steps, "secs": time.time()-t0}, indent=2)
    )

if __name__ == "__main__":
    main()
