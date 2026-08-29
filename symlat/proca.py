from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax

@dataclass
class Field:
    a: float
    v_a: float
    # (x, y, z)
    phi: jax.Array
    # (x, y, z)
    mom_phi: jax.Array
    # (dim, x, y, z)
    vec: jax.Array
    # (dim, x, y, z)
    mom_vec: jax.Array
