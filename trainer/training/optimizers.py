import optax
import pcx.utils as pxu
import pcx.nn as pxnn
from trainer.model import Model


def get_opts(model: Model, init_w, init_h, transition_steps, decay_rate, T):
    optim_w = pxu.Optim(
        lambda: optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(model.l2_w.get()),
            optax.adamw(
                learning_rate=optax.exponential_decay(
                    init_value=init_w,
                    transition_steps=transition_steps,
                    decay_rate=decay_rate,
                    staircase=False,
                )
            )
        ),
        pxu.M(pxnn.LayerParam)(model)
    )

    optim_h = pxu.Optim(
        lambda: optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.sgd(
                learning_rate=optax.cosine_decay_schedule(
                    init_value=init_h,
                    decay_steps=T,
                    alpha=0.05,
                ),
                momentum=0.9,
                nesterov=True
            )
        )
    )

    return optim_w, optim_h

