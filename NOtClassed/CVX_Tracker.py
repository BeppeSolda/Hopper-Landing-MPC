import numpy as np
import time


import jax.numpy as jnp
from jax import jacfwd, jit
from get_reference_trajectory import X_ref, U_ref, t

from plot_results_tracker import plot_results_tracker
import cvxpy as cp
import jax.random as jrandom
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
    L=jnp.block([[s, -jnp.transpose(v)],
                [v, s* jnp.eye(3) + skew_v]])
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
     
     return q[1:4]/q[0]

vector = np.concatenate(([1], -np.ones(3)))

# Create the diagonal matrix
T = np.diag(vector)

zeros_row = np.zeros((1, 3))
I = np.eye(3)

# Vertically stack the zeros_row and the identity matrix
H = np.vstack((zeros_row, I))
@jit
def rocket_dynamics_rk4(x,u, m, J_B, r_T_B, g_I,h):
    
    f1 = rocket_dynamics( x, u, m, J_B, r_T_B, g_I)
    assert f1.shape == x.shape, f"Shape mismatch: f1 {f1.shape} != x {x.shape}"
    f2 = rocket_dynamics(x + 0.5*h*f1,u, m, J_B, r_T_B, g_I)
    assert f2.shape == x.shape, f"Shape mismatch: f2 {f2.shape} != x {x.shape}"
    f3 = rocket_dynamics(x + 0.5*h*f2,u, m, J_B, r_T_B, g_I)
    assert f3.shape == x.shape, f"Shape mismatch: f3 {f3.shape} != x {x.shape}"
    f4 = rocket_dynamics(x + h*f3,u, m, J_B, r_T_B, g_I)
    assert f4.shape == x.shape, f"Shape mismatch: f4 {f4.shape} != x {x.shape}"
    xn = x + (h/6.0)*(f1+ 2*f2 + 2*f3 + f4)
    assert xn.shape == x.shape, f"Shape mismatch: xn {xn.shape} != x {x.shape}"
    q_norm = jnp.linalg.norm(xn[3:7])
    xn = xn.at[3:7].set(xn[3:7]/ q_norm)
    
    return xn
@jit
def compute_jacobian_x(x, u, m, J_B, r_T_B, g_I):
    dynamics_fn = lambda x: rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I,h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(x)


@jit
def compute_jacobian_u(x, u, m, J_B, r_T_B, g_I):
    dynamics_fn = lambda u: rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I,h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(u)

@jit
def rocket_dynamics( x, u, m, J_B, r_T_B, g_I):

   
    n_x = 13
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
    # Velocity dynamics
    f = f.at[3:7].set(0.5 * jnp.dot(L(q),jnp.dot(H,w)) )
    
    # Quaternion dynamics
    f = f.at[7:10].set(((1 /m) * u[0:3]) + Q_t.dot(g_I) - skew(w).dot(v))
    
    # Angular velocity dynamics
    ang_vel_dynamics = jnp.linalg.inv(J_B).dot(skew(r_T_B).dot(u[0:3]) - skew(w).dot(J_B).dot(w))
    #ang_vel_dynamics = ang_vel_dynamics.at[2].add(u[3] / J_B[2, 2])
    f = f.at[10:13].set(ang_vel_dynamics)
    
    return f









U_ref = U_ref.T
X_ref = X_ref.T
T_final = t[-1]
h = t[1] - t[0]
Nt = int(T_final / h) + 1
m = 100.0
J1 = 1/12 * m * (4 + 3 * 0.25**2)
J2 = J1
J3 = 0.5 * m * 0.25**2
J_B = jnp.diag(jnp.array([J1, J2, J3]))
g = 9.81
g_I = jnp.array([0, 0, -g])
r_T_B = jnp.array([0, 0, -0.5])

u0 = U_ref[:, 0]

x0 = X_ref[:, 0]

Max_Thrust = 2*m*g 
Min_Thrust = 0.25*Max_Thrust

Nx = len(x0)
Nu = len(u0)

x = jnp.zeros((Nx, Nt))
u = jnp.zeros((Nu, Nt))
x = x.at[:, 0].set(x0)

Q = 1000000 * jnp.diag(jnp.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))
R = 0.1*jnp.diag(jnp.array([1, 1, 1]))
Q = jnp.identity(Nx)
R = jnp.identity(Nu)
error_vect = jnp.zeros((Nx+Nu, Nt))



AA = compute_jacobian_x(x0, u0, m, J_B, r_T_B, g_I)
BB = compute_jacobian_u(x0, u0, m, J_B, r_T_B, g_I)
N_mpc = 10
measurement_noise_std = 0.1  # Standard deviation of measurement noise
process_noise_std = 0.08  # Standard deviation of process noise
key = jrandom.PRNGKey(0) 
for k in range(Nt - 1):
    if k % 1 == 0:
        
        start_time = time.time()
        A_start = time.time()
        # AA = compute_jacobian_x(X_ref[:, k], U_ref[:, k], m, J_B, r_T_B, g_I)
        # BB = compute_jacobian_u(X_ref[:, k], U_ref[:, k], m, J_B, r_T_B, g_I)
        A_end = time.time()
        AA = compute_jacobian_x(x[:, k], u[:, k-1], m, J_B, r_T_B, g_I)
        BB = compute_jacobian_u(x[:, k], u[:, k-1], m, J_B, r_T_B, g_I)
     
        if k + N_mpc + 1 > Nt:
                N_mpc = Nt - k - 1


        x_mpc = cp.Variable((Nx, N_mpc + 1))
        u_mpc = cp.Variable((Nu, N_mpc))    
        cost = 0
        constraints = [x_mpc[:, 0] == x[:, k]]
       
        problem = cp.Problem(cp.Minimize(cost), constraints)
        MPC_cost_start = time.time()
        for i in range(N_mpc):
            cost += cp.quad_form(x_mpc[:, i] - X_ref[:, k + i], Q) + cp.quad_form(u_mpc[:, i] - U_ref[:, k + i], R) 
            
            constraints += [x_mpc[:, i + 1] == AA @ x_mpc[:, i] + BB @ u_mpc[:, i],cp.norm(u_mpc[:,i], axis=0) <= Max_Thrust]
        #,cp.norm(u_mpc[:,i], axis=0) <= Max_Thrust
        cost += cp.quad_form(x_mpc[:, N_mpc] - X_ref[:, k + N_mpc], Q)
        MPC_cost_end = time.time()

        MPC_solve_start = time.time()
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        MPC_solve_end = time.time()
       
        u_opt = u_mpc[:, 0].value
        

        
        end_time = time.time()
        # print(f"Iteration {k}:")
        # print(f"  Total MPC time: {end_time - start_time:.4f} seconds")
        # print(f"    Linearization time: {A_end - A_start:.4f} seconds")
        # print(f"    Cost function setup time: {MPC_cost_end - MPC_cost_start:.4f} seconds")
        # print(f"    Solver time: {MPC_solve_end - MPC_solve_start:.4f} seconds")
        print(k)
    else:
        u_opt = u[:, k - 1]
    # error_vect=error_vect.at[:,0:Nx].set(X_ref[:,k]-x[:,k])
    # error_vect=error_vect.at[:,Nx:Nx+Nu].set(U_ref[:,k]-u_opt)
    key, subkey = jrandom.split(key)
    position_noise = jrandom.normal(subkey, shape=(3,), dtype=jnp.float32) * process_noise_std
    velocity_noise = jrandom.normal(subkey, shape=(3,), dtype=jnp.float32) * process_noise_std
    noisy_x = x[:, k].copy()
    noisy_x = noisy_x.at[0:3].add(position_noise)
    noisy_x = noisy_x.at[7:10].add(velocity_noise)
    vect = jnp.hstack([X_ref[:,k]-noisy_x, U_ref[:,k]-u_opt])
    # print('size vect',np.shape(vect))
    # print('size error vect',np.shape(error_vect.at[:,k]))
    # print('size vect',vect)
    error_vect = error_vect.at[:,k].set(vect)
    #print('dioboia',np.shape(error_vect))
    #print(error_vect.at[:,k])
    u = u.at[:, k].set(u_opt ) 
   
    x = x.at[:, k + 1].set(rocket_dynamics_rk4(x[:,k], u[:, k], m, J_B, r_T_B, g_I, h))
 

       
    
print(jnp.size(error_vect.at[:,100]))
plot_results_tracker(t, x, X_ref, u, U_ref,error_vect)