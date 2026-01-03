import os
import gc
import jax
import jax.numpy as jnp
import numpy as np
import dill
from tqdm import tqdm
from trainer.model import Model
from trainer.training.forward import forward
import pcx.utils as pxu
import pcx.predictive_coding as pxc


@staticmethod
def _energy_given_xy(x, y, *, model: "Model"):
    model.eval()
    x = jnp.atleast_2d(x)  # (B,D)
    y = jnp.atleast_2d(y)  # (B,C)

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        _ = forward(x, y, model=model)
        e = model.energy()  # (B,)

        l2_x = model.l2_x.get()
        if l2_x > 0:
            e = e + 0.5 * l2_x * jnp.sum(x * x, axis=-1)

        l2_h = model.l2_h.get()
        if l2_h > 0:
            h_pen = 0.0
            for vode in model.vodes[:-1]:
                h = vode.get("h")            # (B,dim)
                h_pen = h_pen + jnp.sum(h*h, axis=-1)
            e = e + 0.5 * l2_h * h_pen

    return jnp.sum(e)  # scalar


def _rand_init(key, shape, kind="normal", scale=1.0, box=None):
    """
    Draw a random initial x_0.
    kind: "normal" or "uniform"
    scale: std (normal) or half-width (uniform)
    box:  tuple (lo, hi) → clip to [lo, hi] after sampling
    """
    if kind == "normal":
        x0 = scale * jax.random.normal(key, shape)
    elif kind == "uniform":
        x0 = jax.random.uniform(key, shape, minval=-scale, maxval=scale)
    else:
        raise ValueError("kind must be 'normal' or 'uniform'")
    if box is not None:
        lo, hi = box
        x0 = jnp.clip(x0, lo, hi)
    return x0


def _invert_once(
    rng_key: "jax.Array",
    y_target: "jax.Array",
    *,
    model: "Model",
    steps: int = 200,
    lr: float = 0.05,
    noise_sigma: float = 0.0,
    noise_type: str = "normal",
    temp: float = 1.0,
    init_kind: str = "normal",
    init_scale: float = 1.0,
    early_stop_energy: float | None = None
):
    """
    Reparameterized inversion: optimize z in R^D with x = tanh(z) in [-1,1]^D.
    """
    y_target = jnp.atleast_1d(y_target)
    input_dim = model.input_dim.get()

    # initialize z (unconstrained latent variable)
    k_init, k_noise = jax.random.split(rng_key)
    z = _rand_init(k_init, (1, input_dim), kind=init_kind, scale=init_scale)

    # energy as a function of z, via x = tanh(z)
    def E_of_z(z_flat):
        x = jnp.tanh(z_flat)
        return _energy_given_xy(x[None, :], y_target[None, :], model=model)

    grad_E = jax.grad(lambda z_flat: E_of_z(z_flat.squeeze(0)))

    def step(carry, i):
        z_cur, key = carry
        g = jax.grad(_energy_given_xy)(
            jnp.tanh(z_cur), y_target[None, :], model=model
        ) * (1.0 - jnp.tanh(z_cur) ** 2)  # chain rule: dE/dz = dE/dx * (1 - tanh^2 z)

        # gradient step in z-space
        z_next = z_cur - lr * g

        # Langevin-like noise in z-space
        def add_noise(kn, zval):
            if noise_sigma <= 0:
                return zval
            if noise_type == "normal":
                z = jax.random.normal(kn, zval.shape)
            elif noise_type == "uniform":
                z = jax.random.uniform(kn, zval.shape, minval=-1.0, maxval=1.0)
            else:
                raise ValueError("noise_type must be 'normal' or 'uniform'")
            return zval + noise_sigma * jnp.sqrt(2.0 * lr * temp) * z

        key, kn = jax.random.split(key)
        z_next = add_noise(kn, z_next)

        x_next = jnp.tanh(z_next)
        e = _energy_given_xy(x_next, y_target[None, :], model=model)
        return (z_next, key), e

    # run gradient descent / Langevin steps
    (z_final, _), e_trace = jax.lax.scan(step, (z, k_noise), jnp.arange(steps))
    x_final = jnp.tanh(z_final)

    # early-stop handling (optional)
    if early_stop_energy is not None:
        idx = jnp.argmax(e_trace <= early_stop_energy)
        use_idx = jnp.where(jnp.any(e_trace <= early_stop_energy), idx, steps - 1)

        def step_noiseless(carry, i):
            z_cur = carry
            g = jax.grad(_energy_given_xy)(
                jnp.tanh(z_cur), y_target[None, :], model=model
            ) * (1.0 - jnp.tanh(z_cur) ** 2)
            z_next = z_cur - lr * g
            return z_next, z_next

        z_star, _ = jax.lax.scan(step_noiseless, z, jnp.arange(use_idx + 1))
        x_star = jnp.tanh(z_star)
        return x_star.squeeze(0), e_trace

    return x_final.squeeze(0), e_trace


def invert_output(
    target_label: int | None = None,
    target_logits: "np.ndarray | jax.Array" = None,
    *,
    num_samples: int = 2048,
    model_ids: list[int] | None = None,
    num_top_models=10,
    steps: int = 100,
    lr: float = 0.005,
    noise_sigma: float = 0,      # set >0 to make it *non-deterministic*, e.g., 0.07
    noise_type: str = "normal",
    temp: float = 1.0,
    init_kind: str = "normal",
    init_scale: float = 0.5,
    batch_size: int = 32,
    early_stop_energy: float | None = None,
    l2_w=0.0,
    l2_x=0.0,
    l2_h=0.0,
    save=False,
    save_dir=None,
    # Trainer-specific parameters
    root=None,
    study_name=None,
    num_models=None,
    input_dim=None,
    output_dim=None,
    hidden_dims=None,
    act_fn=None,
    residual=False,
    model_keys=None
):
    """
    Produce many reconstructions x* such that the PCN believes y(x*) ≈ target.
    - If target_logits is None, we clamp to one-hot for target_label.
    - Non-determinism comes from: random x0 and per-step noise (Langevin).
    Returns a dict: {model_id: np.ndarray [N,2]}
    """
    assert (target_label is not None) ^ (target_logits is not None), \
        "Provide exactly one of target_label or target_logits."
    
    if target_logits is None:
        y = jnp.asarray(jax.nn.one_hot(jnp.asarray(target_label), output_dim))
    else:
        y = jnp.asarray(target_logits).astype(jnp.float32)
        if y.ndim == 1 and y.shape[0] == output_dim:
            pass
        else:
            raise ValueError(f"target_logits must have shape ({output_dim},)")

    if save_dir is None:
        save_dir = f'{root}/{study_name}'
    else:
        save_dir = f'../../{save_dir}'
    save_filepath = f'{save_dir}/reconstructions_s{steps}'

    def get_top_ids():
        lines = []
        with open(f"{root}/{study_name}/accuracies.txt", "r") as f:
            for line in f:
                model_num, acc = line.strip().split(":")
                lines.append([int(model_num), float(acc)])
        sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_lines]

    if model_ids is None:
        model_ids = get_top_ids()[:min(num_models, num_top_models)]
        print(f'Using models: {model_ids}')

    results = {}

    for mid in tqdm(model_ids):
        # Load model weights
        model = Model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            model_key=model_keys[mid],
            residual=residual,
            l2_w=l2_w,
            l2_x=l2_x,
            l2_h=l2_h
        )
        pxu.load_params(model, f'{root}/{study_name}/trained_models/model_{mid}')

        # jit a single-trajectory kernel for speed
        kernel = jax.jit(lambda k: _invert_once(
            k, y, model=model,
            steps=steps, lr=lr,
            noise_sigma=noise_sigma, noise_type=noise_type, temp=temp,
            init_kind=init_kind, init_scale=init_scale,
            early_stop_energy=early_stop_energy
        ))

        # sample many x*
        xs = []
        Es = []

        # batched PRNG
        main_key = jax.random.PRNGKey(int(model_keys[mid][0]))
        keys = jax.random.split(main_key, num_samples)

        # loop in manageable chunks to avoid host <-> device thrash
        for start in range(0, num_samples, batch_size):
            k_chunk = keys[start:start+batch_size]
            # vmap over the kernel to parallelize several samples at once
            x_chunk, e_traces = jax.vmap(kernel)(k_chunk)
            xs.append(np.asarray(x_chunk))
            # final energies are last elements of traces
            Es.append(np.asarray(e_traces[:, -1]))

        X = np.concatenate(xs, axis=0)  # shape (N, 2)
        E = np.concatenate(Es, axis=0)  # shape (N,)

        if save:
            os.makedirs(save_filepath, exist_ok=True)
            with open(f'{save_filepath}/model_{mid}_label_{int(target_label)}.dill', 'wb') as f:
                dill.dump(X, f)

        results[mid] = X

        # cleanup (helps on long runs)
        del model
        gc.collect()
        jax.clear_caches()

    return results

