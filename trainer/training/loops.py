import jax
import jax.numpy as jnp
import numpy as np
import pcx.functional as pxf
import pcx.utils as pxu
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
from trainer.model import Model
from trainer.training.forward import forward, energy


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int,
    x: jax.Array,
    y: jax.Array,
    *,
    model: Model,
    optim_w: pxu.Optim,
    optim_h: pxu.Optim
):
    # This only sets an internal flag to be "train" (instead of "eval")
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    # As it is explained later, we initialise the state optimizer for the current batch.
    # We specify to ignore the `VodeParams` which have the `frozen` attribute set to True.
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),
                has_aux=True
            )(energy)(x, model=model)

        optim_h.step(model, g["model"])

    optim_h.clear()

    # Weight update step
    model.clear_params(pxc.VodeParam.Cache) # Clear cache before weight update
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)

    # Since the energy function returns the sum of the energies over the batch dimension, we need to scale the
    # gradient according to the number of samples in the batch.
    optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])
    

@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_
    

# Standard training loop
def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        x_flat = x.reshape(x.shape[0], -1)
        x_jax = jnp.asarray(x_flat)
        y_jax = jnp.asarray(y)
        train_on_batch(T, x_jax, jax.nn.one_hot(y_jax, model.output_dim.get()), model=model, optim_w=optim_w, optim_h=optim_h)
        

# Standard evaluation loop
def eval(dl, *, model: Model):
    acc = []
    ys_ = []

    for x, y in dl:
        x_flat = x.reshape(x.shape[0], -1)
        x_jax = jnp.asarray(x_flat)
        y_jax = jnp.asarray(y)
        a, y_ = eval_on_batch(x_jax, y_jax, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)

