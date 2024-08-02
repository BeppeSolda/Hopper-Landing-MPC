import jax.numpy as jnp
from jax import jit
from params import J_B, r_T_B, g_I, T, H, h, m

@jit
def skew(v):
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

@jit
def L(q):
    s = q[0]
    v = q[1:4]
    skew_v = skew(v)
    v = v.reshape(3, 1)
    return jnp.block([[s, -jnp.transpose(v)], [v, s * jnp.eye(3) + skew_v]])

@jit
def qtoQ(q, T, H):
    return jnp.transpose(H) @ T @ L(q) @ T @ L(q) @ H

@jit
def G(q, H):
    return L(q) @ H

@jit
def rptoq(phi):
    phi_norm_sq = jnp.dot(phi.T, phi)
    scalar_part = 1 / jnp.sqrt(1 + phi_norm_sq)
    vector_part = scalar_part * phi
    return jnp.concatenate(([scalar_part], vector_part))

@jit
def qtorp(q):
    return q[1:4] / q[0]

@jit
def rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I, h):
    f1 = rocket_dynamics(x, u, m, J_B, r_T_B, g_I,T, H)
    f2 = rocket_dynamics(x + 0.5 * h * f1, u, m, J_B, r_T_B, g_I,T, H)
    f3 = rocket_dynamics(x + 0.5 * h * f2, u, m, J_B, r_T_B, g_I,T, H)
    f4 = rocket_dynamics(x + h * f3, u, m, J_B, r_T_B, g_I,T, H)
    xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    q_norm = jnp.linalg.norm(xn[3:7])
    xn = xn.at[3:7].set(xn[3:7] / q_norm)
    return xn

@jit
def rocket_dynamics(x, u, m, J_B, r_T_B, g_I, T, H):
    n_x = 13
    r = x[0:3]
    q = x[3:7]
    v = x[7:10]
    q_norm = jnp.linalg.norm(q)
    q = q / jnp.where(q_norm == 0, 1, q_norm)
    w = x[10:13]
    f = jnp.zeros(n_x)

    Q = qtoQ(q, T, H)
    Q_t = Q.T

    f = f.at[0:3].set(Q.dot(v))
    f = f.at[3:7].set(0.5 * jnp.dot(L(q), jnp.dot(H, w)))
    f = f.at[7:10].set(((1 / m) * u[0:3]) + Q_t.dot(g_I) - skew(w).dot(v))
    ang_vel_dynamics = jnp.linalg.inv(J_B).dot(skew(r_T_B).dot(u[0:3]) - skew(w).dot(J_B).dot(w))
    #ang_vel_dynamics = ang_vel_dynamics.at[2].add(u[3] / J_B[2, 2])
    f = f.at[10:13].set(ang_vel_dynamics)

    return f
