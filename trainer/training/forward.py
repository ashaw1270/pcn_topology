import pcx.functional as pxf
import pcx.utils as pxu
import pcx.predictive_coding as pxc
from trainer.model import Model


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)
    

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    import jax.lax
    
    y_ = model(x, None)

    # base per-sample energy
    e = model.energy()  # scalar for this sample (because we're inside vmap)

    # ---- Orchard & Sun style "activity decay" ----
    l2_x = model.l2_x.get()
    if l2_x > 0:
        e = e + 0.5 * l2_x * jnp.sum(x * x)

    l2_h = model.l2_h.get()
    if l2_h > 0:
        h_pen = 0.0
        # exclude output Vode (often clamped/frozen)
        for vode in model.vodes[:-1]:
            h = vode.get("h")
            h_pen = h_pen + jnp.sum(h * h)
        e = e + 0.5 * l2_h * h_pen
        
    return jax.lax.psum(e, "batch"), y_

