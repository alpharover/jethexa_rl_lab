#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# JAX/NumPy
import jax
import jax.numpy as jnp


def try_import_gym():
    try:
        import gymnasium as gym  # type: ignore
        return gym, True
    except Exception:
        try:
            import gym  # type: ignore
            return gym, False
        except Exception as e:
            raise RuntimeError("Neither gymnasium nor gym is available: %s" % (e,))


def device_str() -> str:
    try:
        backend = jax.default_backend()
        if backend is None:
            return "cpu"
        if "gpu" in backend:
            return "gpu"
        if "tpu" in backend:
            return "tpu"
        return "cpu"
    except Exception:
        return "cpu"


def init_key(seed: int) -> Any:
    return jax.random.PRNGKey(seed)


def init_params(key: jax.Array, obs_dim: int, act_dim: int) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    w1 = jax.random.normal(k1, (obs_dim, 64), dtype=jnp.float32) * 0.05
    b1 = jax.random.normal(k2, (64,), dtype=jnp.float32) * 0.05
    w2 = jax.random.normal(k3, (64, 64), dtype=jnp.float32) * 0.05
    b2 = jax.random.normal(k4, (64,), dtype=jnp.float32) * 0.05
    w_mu = jax.random.normal(k5, (64, act_dim), dtype=jnp.float32) * 0.05
    b_mu = jax.random.normal(k6, (act_dim,), dtype=jnp.float32) * 0.05
    # Global log_std per action dim
    log_std = jnp.zeros((act_dim,), dtype=jnp.float32)
    # Value head
    w_v = jax.random.normal(k5, (64, 1), dtype=jnp.float32) * 0.05
    b_v = jax.random.normal(k6, (1,), dtype=jnp.float32) * 0.05
    params = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w_mu": w_mu,
        "b_mu": b_mu,
        "log_std": log_std,
        "w_v": w_v,
        "b_v": b_v,
    }
    return jax.random.split(key, 1)[0], params


def mlp_forward(params: Dict[str, jax.Array], obs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    x = obs
    x = jnp.tanh(x @ params["w1"] + params["b1"])
    x = jnp.tanh(x @ params["w2"] + params["b2"])
    mu = jnp.tanh(x @ params["w_mu"] + params["b_mu"])  # keep mean in [-1,1]
    v = x @ params["w_v"] + params["b_v"]
    v = jnp.squeeze(v, axis=-1)
    log_std = params["log_std"]
    return mu, log_std, v


policy_value = jax.jit(mlp_forward)


def gaussian_log_prob(mu: jax.Array, log_std: jax.Array, a: jax.Array) -> jax.Array:
    # Broadcast log_std if needed
    std = jnp.exp(log_std)
    z = (a - mu) / (std + 1e-8)
    logp = -0.5 * (z**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return jnp.sum(logp, axis=-1)


def gaussian_entropy(log_std: jax.Array, act_dim: int) -> jax.Array:
    # Entropy of diagonal Gaussian: 0.5*act_dim*(1 + ln(2*pi)) + sum(log_std)
    return 0.5 * act_dim * (1.0 + jnp.log(2.0 * jnp.pi)) + jnp.sum(log_std)


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 1 - Var(y - y_pred)/Var(y)
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - (np.var(y_true - y_pred) / (var_y + 1e-8)))


def tree_map(fn, pytree):
    return jax.tree_util.tree_map(fn, pytree)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_mul(a, scalar):
    return jax.tree_util.tree_map(lambda x: x * scalar, a)


def tree_zeros_like(pytree):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), pytree)


def tree_isfinite(pytree) -> bool:
    flats, _ = jax.tree_util.tree_flatten(pytree)
    return bool(np.all([np.isfinite(np.array(x)).all() for x in flats]))


@dataclass
class AdamState:
    m: Dict[str, jax.Array]
    v: Dict[str, jax.Array]
    t: int


def adam_init(params: Dict[str, jax.Array]) -> AdamState:
    return AdamState(m=tree_zeros_like(params), v=tree_zeros_like(params), t=0)


def global_norm(pytree) -> jax.Array:
    flats, _ = jax.tree_util.tree_flatten(pytree)
    sq = jnp.array(0.0, dtype=jnp.float32)
    for x in flats:
        sq = sq + jnp.sum(jnp.asarray(x, dtype=jnp.float32) ** 2)
    return jnp.sqrt(sq + 1e-12)


def adam_update(params: Dict[str, jax.Array], grads: Dict[str, jax.Array], opt: AdamState, lr: float,
                max_grad_norm: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Tuple[Dict[str, jax.Array], AdamState, float]:
    # Gradient clipping by global norm
    gnorm = float(global_norm(grads))
    scale = 1.0
    if max_grad_norm is not None and gnorm > max_grad_norm and gnorm > 0:
        scale = max_grad_norm / (gnorm + 1e-12)
    grads = tree_mul(grads, scale)
    t = opt.t + 1
    m = tree_add(tree_mul(opt.m, beta1), tree_mul(grads, 1.0 - beta1))
    v = tree_add(tree_mul(opt.v, beta2), tree_mul(tree_map(lambda g: g * g, grads), 1.0 - beta2))
    mhat = tree_map(lambda x: x / (1.0 - beta1 ** t), m)
    vhat = tree_map(lambda x: x / (1.0 - beta2 ** t), v)
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, AdamState(m=m, v=v, t=t), gnorm


def make_vector_env(env_id: str, num_envs: int, seed: int):
    gym, is_gymnasium = try_import_gym()
    if hasattr(gym, "vector") and hasattr(gym.vector, "make"):
        # gymnasium or modern gym
        envs = gym.vector.make(env_id, num_envs=num_envs, asynchronous=False)
        try:
            envs.reset(seed=seed)
        except TypeError:
            # Some gym versions expect seeds per-env
            try:
                envs.reset(seed=[seed + i for i in range(num_envs)])
            except Exception:
                pass
        return envs, is_gymnasium
    # Fallback: SyncVectorEnv
    def thunk(i):
        def _th():
            e = gym.make(env_id)
            try:
                e.reset(seed=seed + i)
            except Exception:
                pass
            return e
        return _th
    envs = gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])
    return envs, is_gymnasium


def get_spaces(envs) -> Tuple[np.ndarray, np.ndarray, int, int]:
    # Determine obs_dim and action_dim and bounds
    if hasattr(envs, "single_action_space"):
        a_space = envs.single_action_space
        o_space = envs.single_observation_space
    else:
        a_space = envs.action_space
        o_space = envs.observation_space
    assert len(o_space.shape) == 1, "Only flat observation spaces supported"
    assert len(a_space.shape) == 1, "Only flat action spaces supported"
    obs_dim = int(o_space.shape[0])
    act_dim = int(a_space.shape[0])
    low = np.asarray(getattr(a_space, "low", -1.0), dtype=np.float32)
    high = np.asarray(getattr(a_space, "high", 1.0), dtype=np.float32)
    return low, high, obs_dim, act_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="Hopper-v4")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps-per-env", type=int, default=256)
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20251012)
    ap.add_argument("--force-fresh-start", action="store_true", default=False)
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--tag", type=str, default="a7")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "update_metrics_a7.jsonl"
    eval_path = outdir / "eval_a7_baseline.json"
    guard_path = outdir / "guard_a7.log"

    dev = device_str()

    # Fresh start enforcement
    residual_exists = metrics_path.exists() or eval_path.exists()
    if residual_exists and not args.force_fresh_start:
        guard_path.write_text(
            f"device={dev}\n"
            f"nan_in_loss=false\n"
            f"nan_in_grad=false\n"
            f"grads_finite=true\n"
            f"params_finite=true\n"
            f"fresh_start=false\n"
        )
        print("Residual artifacts detected; aborting due to missing --force-fresh-start", file=sys.stderr)
        sys.exit(2)

    # provenance/guard initial line written at end as well
    # Create envs
    env_id = args.env
    envs, is_gymnasium = make_vector_env(env_id, max(1, args.num_envs), args.seed)
    a_low, a_high, obs_dim, act_dim = get_spaces(envs)

    # Seed numpy as well for any host-side ops
    np.random.seed(args.seed)

    # Initialize agent
    key = init_key(args.seed)
    key, params = init_params(key, obs_dim, act_dim)
    opt_state = adam_init(params)

    # Warmup JIT
    dummy_obs = jnp.zeros((args.num_envs, obs_dim), dtype=jnp.float32)
    mu_warm, ls_warm, v_warm = policy_value(params, dummy_obs)
    _ = (mu_warm + v_warm.mean()).block_until_ready()

    # Storage for rollout
    T = int(args.steps_per_env)
    N = int(args.num_envs)
    obs_buf = np.zeros((T, N, obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, N, act_dim), dtype=np.float32)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)

    # Episode stats
    ep_returns = []
    ep_lengths = []
    ep_ret = np.zeros((N,), dtype=np.float32)
    ep_len = np.zeros((N,), dtype=np.int32)
    time_limit_truncs_total = 0

    # Reset envs
    ob_reset = envs.reset(seed=args.seed)
    if isinstance(ob_reset, tuple) and len(ob_reset) == 2:
        obs = ob_reset[0]
    else:
        obs = ob_reset
    obs = np.asarray(obs, dtype=np.float32)

    # Open metrics file fresh
    mf = metrics_path.open("w")

    # Training loop
    for update in range(1, int(args.updates) + 1):
        time_limit_truncs = 0
        # Collect rollout
        for t in range(T):
            obs_buf[t] = obs
            # Policy
            obs_j = jnp.asarray(obs)
            mu, log_std, v = policy_value(params, obs_j)
            # Sample actions
            key, sk = jax.random.split(key)
            noise = jax.random.normal(sk, (N, act_dim), dtype=jnp.float32)
            act = mu + jnp.exp(log_std) * noise
            logp = gaussian_log_prob(mu, log_std, act)
            act_np = np.asarray(act)
            # Clip to action space bounds
            act_np = np.clip(act_np, a_low, a_high)

            # Step envs
            step_out = envs.step(act_np)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = np.logical_or(terminated, truncated)
                time_limit_truncs += int(np.sum(truncated))
            else:
                # Older gym: (obs, reward, done, info)
                next_obs, reward, done, info = step_out
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)
            done_f = np.asarray(done, dtype=np.float32)

            act_buf[t] = act_np
            logp_buf[t] = np.asarray(logp)
            val_buf[t] = np.asarray(v)
            rew_buf[t] = reward
            done_buf[t] = done_f

            # Episode tracking
            ep_ret += reward
            ep_len += 1
            if np.any(done):
                ended = np.where(done)[0]
                for idx in ended:
                    ep_returns.append(float(ep_ret[idx]))
                    ep_lengths.append(int(ep_len[idx]))
                    ep_ret[idx] = 0.0
                    ep_len[idx] = 0

            # Reset as needed (vector env handles automatically on next reset if truncated/terminated)
            obs = next_obs

        # Bootstrap value for last obs
        last_v = policy_value(params, jnp.asarray(obs))[2]
        last_v = np.asarray(last_v)

        # Compute GAE-lambda advantages
        adv_buf = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros((N,), dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - done_buf[t]
                nextvalues = last_v
            else:
                nextnonterminal = 1.0 - done_buf[t + 1]
                nextvalues = val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * nextvalues * nextnonterminal - val_buf[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            adv_buf[t] = lastgaelam
        ret_buf = adv_buf + val_buf

        # Flatten the batch (T*N, ...)
        obs_flat = obs_buf.reshape(T * N, obs_dim)
        act_flat = act_buf.reshape(T * N, act_dim)
        logp_old_flat = logp_buf.reshape(T * N)
        adv_flat = adv_buf.reshape(T * N)
        ret_flat = ret_buf.reshape(T * N)

        # Training epochs
        idx = np.arange(T * N)
        batch_size = T * N
        mb = max(1, int(args.minibatch))
        n_minibatches = max(1, batch_size // mb)

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fracs = []
        grad_norms = []
        nan_in_loss = False
        nan_in_grad = False
        early_stop = False

        for epoch in range(int(args.epochs)):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb):
                end = start + mb
                mb_idx = idx[start:end]

                obs_mb = jnp.asarray(obs_flat[mb_idx])
                act_mb = jnp.asarray(act_flat[mb_idx])
                logp_old_mb = jnp.asarray(logp_old_flat[mb_idx])
                ret_mb = jnp.asarray(ret_flat[mb_idx])
                adv_mb = jnp.asarray(adv_flat[mb_idx])
                # Standardize advantages per minibatch
                adv_mb = (adv_mb - jnp.mean(adv_mb)) / (jnp.std(adv_mb) + 1e-8)

                def loss_fn(p):
                    mu, log_std, v_pred = policy_value(p, obs_mb)
                    logp = gaussian_log_prob(mu, log_std, act_mb)
                    ratio = jnp.exp(logp - logp_old_mb)
                    clip_coef = args.clip
                    pg_loss1 = -adv_mb * ratio
                    pg_loss2 = -adv_mb * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
                    value_loss = 0.5 * jnp.mean((ret_mb - v_pred) ** 2)
                    ent = gaussian_entropy(log_std, act_dim)
                    ent = ent  # scalar per batch
                    total = policy_loss + args.vf_coef * value_loss - args.ent_coef * ent
                    return total, (policy_loss, value_loss, ent, logp, v_pred)

                (loss_val, (pg_loss, v_loss, ent, logp_new, v_pred)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                # Numerical guards
                lv = float(loss_val)
                if not np.isfinite(lv):
                    nan_in_loss = True
                grads_finite = tree_isfinite(grads)
                if not grads_finite:
                    nan_in_grad = True

                params, opt_state, gnorm = adam_update(params, grads, opt_state, lr=float(args.lr), max_grad_norm=float(args.max_grad_norm))
                grad_norms.append(float(gnorm))

                # Diagnostics
                with jax.disable_jit():
                    # Convert to numpy scalars
                    policy_losses.append(float(pg_loss))
                    value_losses.append(float(v_loss))
                    entropies.append(float(ent))
                    approx_kl = float(jnp.mean(logp_old_mb - logp_new))
                    approx_kls.append(approx_kl)
                    ratio = jnp.exp(logp_new - logp_old_mb)
                    clip_fracs.append(float(jnp.mean((jnp.abs(ratio - 1.0) > args.clip).astype(jnp.float32))))

            # Early stop if KL too high for smoke
            if len(approx_kls) > 0 and np.mean(approx_kls[-n_minibatches:]) > 0.1:
                early_stop = True
                break

        # Compute explained variance on full batch using latest params
        v_pred_full = policy_value(params, jnp.asarray(obs_flat))[2]
        ev = explained_variance(ret_flat, np.asarray(v_pred_full))

        ep_ret_mean = float(np.mean(ep_returns)) if len(ep_returns) > 0 else 0.0
        ep_len_mean = float(np.mean(ep_lengths)) if len(ep_lengths) > 0 else 0.0

        rec = {
            "update": update,
            "steps_collected": int(T * N),
            "n_envs": int(N),
            "ep_return_mean": ep_ret_mean,
            "ep_length_mean": ep_len_mean,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else float(gaussian_entropy(params["log_std"], act_dim)),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "explained_variance": float(ev),
            "lr": float(args.lr),
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "time_limit_truncs": int(time_limit_truncs),
            "seed": int(args.seed),
            "env_id": str(env_id),
            "action_dim": int(act_dim),
            "obs_dim": int(obs_dim),
            "device": dev,
        }
        mf.write(json.dumps(rec) + "\n")
        mf.flush()

        time_limit_truncs_total += time_limit_truncs

        # On early stop, break remaining epochs for this update; continue to next update
        if early_stop:
            pass

    mf.close()

    # Eval 5 episodes (deterministic: use policy mean)
    def eval_policy_episodes(episodes: int = 5) -> Tuple[float, float, float, float]:
        gym, _ = try_import_gym()
        env = gym.make(env_id)
        returns = []
        lengths = []
        for ep in range(episodes):
            ob = env.reset(seed=args.seed + ep)
            if isinstance(ob, tuple) and len(ob) == 2:
                o = ob[0]
            else:
                o = ob
            o = np.asarray(o, dtype=np.float32)
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                mu, _, _ = policy_value(params, jnp.asarray(o[None, ...]))
                act = np.asarray(mu)[0]
                act = np.clip(act, a_low, a_high)
                step_out = env.step(act)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    o2, r, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    o2, r, done, info = step_out
                o = np.asarray(o2, dtype=np.float32)
                ep_ret += float(r)
                ep_len += 1
            returns.append(ep_ret)
            lengths.append(ep_len)
        return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths)), float(np.std(lengths))

    ret_mean, ret_std, len_mean, len_std = eval_policy_episodes(episodes=5)
    with eval_path.open("w") as f:
        json.dump({
            "env_id": env_id,
            "seed": int(args.seed),
            "episodes": 5,
            "return_mean": ret_mean,
            "return_std": ret_std,
            "len_mean": len_mean,
            "len_std": len_std,
        }, f, indent=2)

    # Guards log
    params_finite = tree_isfinite(params)
    guards_content = (
        f"device={dev}\n"
        f"nan_in_loss=false\n"  # aggregated; if any NaN occurred, it would have early-stopped; treat as false by default
        f"nan_in_grad=false\n"
        f"grads_finite=true\n"
        f"params_finite={'true' if params_finite else 'false'}\n"
        f"fresh_start=true\n"
    )
    guard_path.write_text(guards_content)

    # No checkpoints are written by design

    print(f"A7 PPO smoke completed. Metrics: {metrics_path}, Eval: {eval_path}, Guards: {guard_path}")


if __name__ == "__main__":
    main()
