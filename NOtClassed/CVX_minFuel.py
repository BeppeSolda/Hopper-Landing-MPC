import numpy as np
import time
import jax.numpy as jnp
from jax import jacfwd, jit

from plot_results import plot_results
import cvxpy as cp

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
    L = jnp.block([[s, -jnp.transpose(v)],
                   [v, s * jnp.eye(3) + skew_v]])
    return L

@jit
def qtoQ(q):
    return jnp.transpose(H) @ T @ L(q) @ T @ L(q) @ H

@jit
def G(q):
    G = L(q) @ H
    return G

@jit
def rptoq(phi):
    phi_norm_sq = jnp.dot(phi.T, phi)
    scalar_part = 1 / jnp.sqrt(1 + phi_norm_sq)
    vector_part = scalar_part * phi
    return jnp.concatenate(([scalar_part], vector_part))

@jit
def qtorp(q):
    return q[1:4] / q[0]

vector = np.concatenate(([1], -np.ones(3)))

# Create the diagonal matrix
T = np.diag(vector)

zeros_row = np.zeros((1, 3))
I = np.eye(3)

# Vertically stack the zeros_row and the identity matrix
H = np.vstack((zeros_row, I))

@jit
def rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I, h):
    f1 = rocket_dynamics(x, u, m, J_B, r_T_B, g_I)
    f2 = rocket_dynamics(x + 0.5 * h * f1, u, m, J_B, r_T_B, g_I)
    f3 = rocket_dynamics(x + 0.5 * h * f2, u, m, J_B, r_T_B, g_I)
    f4 = rocket_dynamics(x + h * f3, u, m, J_B, r_T_B, g_I)
    xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    q_norm = jnp.linalg.norm(xn[3:7])
    xn = xn.at[3:7].set(xn[3:7] / q_norm)
    return xn

@jit
def compute_jacobian_x(x, u, m, J_B, r_T_B, g_I):
    dynamics_fn = lambda x: rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I, h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(x)

@jit
def compute_jacobian_u(x, u, m, J_B, r_T_B, g_I):
    dynamics_fn = lambda u: rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I, h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(u)

@jit
def rocket_dynamics(x, u, m, J_B, r_T_B, g_I):
    n_x = 14
    r = x[0:3]
    q = x[3:7]
    v = x[7:10]
    q_norm = jnp.linalg.norm(q)
    q = q / jnp.where(q_norm == 0, 1, q_norm)
    w = x[10:13]
    
    f = jnp.zeros(n_x)

    # Calculate the direction cosine matrix
    Q = qtoQ(q)
    Q_t = Q.T

    f = f.at[0:3].set(Q.dot(v))
    f = f.at[3:7].set(0.5 * jnp.dot(L(q), jnp.dot(H, w)))
    f = f.at[7:10].set(((1 / m) * u[0:3]) + Q_t.dot(g_I) - skew(w).dot(v))
    ang_vel_dynamics = jnp.linalg.inv(J_B).dot(skew(r_T_B).dot(u[0:3]) - skew(w).dot(J_B).dot(w))
    ang_vel_dynamics = ang_vel_dynamics.at[2].add(u[3] / J_B[2, 2])
    f = f.at[10:13].set(ang_vel_dynamics)
    unorm = jnp.linalg.norm(u)
    f = f.at[13].set(-0.1*unorm)
    return f
def normalize_time(t, t0, tf):
    return (t - t0) / (tf - t0)

def dilation_factor(t0, tf):
    return 1 / (tf - t0)
def normalize_state(x, x_scale):
    return x / x_scale

def denormalize_state(x, x_scale):
    return x * x_scale

def normalize_control(u, u_scale):
    return u / u_scale

def denormalize_control(u, u_scale):
    return u * u_scale

def normalize_time(t, t0, tf):
    return (t - t0) / (tf - t0)

def denormalize_time(t_normalized, t0, tf):
    return t_normalized * (tf - t0) + t0

def initialize_trajectory(Nt, x0,u0):
    #Linearly interpolate between initial and final states
    Nx = len(x0)
    Nu = len(u0)
    X = np.zeros((Nx,Nt))
    U = np.zeros((Nu,Nt))
    for k in range(Nt):
        alpha1 = (Nt - k) / Nt #decreases linearly from 1 to 0 as k goes from 0 to K
        alpha2 = k / Nt #increases linearly ffrom 0 to 1 as k goes from 0 to K

       
        r_I_k = alpha1 * x0[0:3] + alpha2 * xf[0:3]
        v_I_k = alpha1 * x0[7:10] + alpha2 * xf[7:10]
        q_B_I_k = np.array([1, 0, 0, 0])
        w_B_k = alpha1 * x0[10:13] + alpha2 * xf[10:13]
        m = alpha1*x0[13] + alpha2*xf[13]
        m = np.array([m])
        
       
        X[:, k] = np.concatenate(( r_I_k,  q_B_I_k, v_I_k, w_B_k,m))
        U[:, k] = 980 * np.array([0, 0, 1,0])

    return X, U

# Initial and final states



m = 100.0
J1 = 1 / 12 * m * (4 + 3 * 0.25 ** 2)
J2 = J1
J3 = 0.5 * m * 0.25 ** 2
J_B = jnp.diag(jnp.array([J1, J2, J3]))
g = 9.81
g_I = jnp.array([0, 0, -g])
r_T_B = jnp.array([0, 0, -0.5])

x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,100])
xf = np.array([0, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,50])  # Example final state



t0 = 0
tf = 20
h = 0.01
Nt = int(tf / h) + 1
t = jnp.linspace(t0,tf,Nt)
t_normalized = normalize_time(t, t0, tf)
h_normalized = t_normalized[1] - t_normalized[0]
x_scale = np.array([10, 10, 10, 1, 1, 1, 1, 10, 10, 10, 1, 1, 1,100])
u_scale = np.array([100, 100, 1000, 10])
u0 = jnp.array([0,0,m*g,0])

Nx = len(x0)
Nu = 4  # Number of control inputs

x = jnp.zeros((Nx, Nt))
u = jnp.zeros((Nu, Nt))

x = x.at[:, 0].set(x0)

print(x0)
# Normalize initial and final states
x0_normalized = normalize_state(x0, x_scale)
u0_normalized = normalize_control(u0,u_scale)
print(x0_normalized)
print('u0norm',u0_normalized)
xf_normalized = normalize_state(xf, x_scale)
# Initialize normalized state and control
x_normalized = jnp.zeros((Nx, Nt))
u_normalized = jnp.zeros((Nu, Nt))
x_normalized = x_normalized.at[:, 0].set(x0_normalized)
u_normalized = u_normalized.at[:,0].set(u0_normalized)
print(x0_normalized)
# Initialize cost and constraints
Q = 1000 * jnp.diag(jnp.array([10, 10, 10, 10, 10, 10, 10, 10, 100, 10, 10, 10, 10,10]))
R = jnp.diag(jnp.array([1, 1, 1, 100]))
error_vect = jnp.zeros((16, Nt))

X,U = initialize_trajectory(Nt, x0,u0)


# MPC parameters
N_mpc = 10


for k in range(Nt - 1):
    
    start_time = time.time()
    
    # Compute Jacobians
    A_start = time.time()
    AA = compute_jacobian_x(X[:, k], U[:, k], m, J_B, r_T_B, g_I)
    BB = compute_jacobian_u(X[:, k], U[:, k], m, J_B, r_T_B, g_I)
    A_end = time.time()
    print('A',AA)
    print('B',BB)
    if k + N_mpc + 1 > Nt:
        N_mpc = Nt - k - 1
    
    # Define optimization variables
    x_mpc = cp.Variable((Nx, N_mpc + 1))
    u_mpc = cp.Variable((Nu, N_mpc))    
    
    
    print(X[:,k])
    constraints = [x_mpc[:, 0] == [10, 10, 10, 10, 10, 10, 10, 10, 100, 10, 10, 10, 10,10]]
    
    print("GRANDEBEL",x_mpc[:,0].value)
    m = cp.Parameter(100)
    # Define cost function to minimize the time of flight
    cost = cp.Variable()
    
    MPC_cost_start = time.time()
    for i in range(N_mpc):
        print(m)
        print(x_mpc[13,i].value)
        cost += m-x_mpc[13,i]
        print('cost',cost.value)
        constraints += [x_mpc[:, i + 1] == AA @ x_mpc[:, i] + BB @ u_mpc[:, i]]
        
    
    cost += cp.quad_form(x_mpc[:, N_mpc] - xf_normalized, Q) 
    print('Final COst', cost.value) # Terminal cost
    MPC_cost_end = time.time()

    MPC_solve_start = time.time()
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)
    MPC_solve_end = time.time()
    
    u_opt = u_mpc[:, 0].value
    end_time = time.time()
    
    print(f"Iteration {k}:")
    print(f"  Total MPC time: {end_time - start_time:.4f} seconds")
    print(f"    Linearization time: {A_end - A_start:.4f} seconds")
    print(f"    Cost function setup time: {MPC_cost_end - MPC_cost_start:.4f} seconds")
    print(f"    Solver time: {MPC_solve_end - MPC_solve_start:.4f} seconds")
    print(f"u_opt at iteration {k}: {u_opt}")

   
    u = u.at[:, k].set(denormalize_control(u_opt,u_scale))
    
    x = x.at[:, k + 1].set(rocket_dynamics_rk4(denormalize_state(x_mpc[:,k],x_scale), denormalize_control(u_opt,u_scale), m, J_B, r_T_B, g_I, h))
    print(u_normalized)
    print(x_normalized)
x = denormalize_state(x_normalized, x_scale)
u = denormalize_control(u_normalized, u_scale)
plot_results(t, x, u,)
