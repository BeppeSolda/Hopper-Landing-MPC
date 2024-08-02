import jax.numpy as jnp

vector = jnp.concatenate(([1], -jnp.ones(3)))
T = jnp.diag(vector)
zeros_row = jnp.zeros((1, 3))
I = jnp.eye(3)
H = jnp.vstack((zeros_row, I))
