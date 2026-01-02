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
    return jax.lax.psum(model.energy(), "batch"), y_

