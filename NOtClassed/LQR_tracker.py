import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.linalg import inv
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd, jit
from scipy.linalg import eig, svd, block_diag
import control as ct
from get_reference_trajectory import X_ref, U_ref, t
import control
#from plot_results_tracker import plot_results
def skew(v):
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def L(q):
    s = q[0]
    v = q[1:4]
    skew_v = skew(v)
    v = v.reshape(3, 1)
    L=jnp.block([[s, -jnp.transpose(v)],
                [v, s* jnp.eye(3) + skew_v]])
    return L

def qtoQ(q):
    return jnp.transpose(H) @ T @ L(q) @ T @ L(q) @ H
def G(q):
      G = L(q) @ H
      return G
def rptoq(phi):
    phi_norm_sq = jnp.dot(phi.T, phi)
    scalar_part = 1 / jnp.sqrt(1 + phi_norm_sq)
    vector_part = scalar_part * phi
    return jnp.concatenate(([scalar_part], vector_part))

def qtorp(q):
     
     return q[1:4]/q[0]

vector = np.concatenate(([1], -np.ones(3)))

# Create the diagonal matrix
T = np.diag(vector)

zeros_row = np.zeros((1, 3))
I = np.eye(3)

# Vertically stack the zeros_row and the identity matrix
H = np.vstack((zeros_row, I))

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

def compute_jacobian_x(x, u, m, J_B, r_T_B, g_I):
    dynamics_fn = lambda x: rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I,h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(x)



def compute_jacobian_u(x, u, m, J_B, r_T_B, g_I):
    dynamics_fn = lambda u: rocket_dynamics_rk4(x, u, m, J_B, r_T_B, g_I,h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(u)


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
    ang_vel_dynamics = ang_vel_dynamics.at[2].add(u[3] / J_B[2, 2])
    f = f.at[10:13].set(ang_vel_dynamics)
    
    return f

def controller(x,K,X_ref,U_ref):
    q0 = X_ref[3:7]
    q = x[3:7]
    L_t = jnp.transpose(L(q0))
    phi = qtorp(jnp.dot(L_t,q))

    delta_x_tilde = jnp.zeros(12)
    delta_x_tilde = delta_x_tilde.at[0:3].set(x[0:3] - X_ref[0:3])
    delta_x_tilde= delta_x_tilde.at[3:6].set(phi)
    delta_x_tilde = delta_x_tilde.at[6:9].set(x[7:10] - X_ref[7:10])
    delta_x_tilde = delta_x_tilde.at[9:12].set(x[10:13] -X_ref[10:13])

    u = U_ref - jnp.dot(K, delta_x_tilde)
   
    delta_x_tilde = delta_x_tilde.reshape(-1, 1)
    error = (U_ref - u).reshape(-1, 1)
    #error_vect = jnp.vstack([delta_x_tilde, error])
    return u
def E(q):
    return block_diag(jnp.eye(3), G(q), jnp.eye(6))
import numpy as np

def compute_jacobian_x_fd(x, u, delta, rocket_dynamics_rk4, m, J_B, r_T_B, g_I,h):
    Nx = len(x)
    jacobian_x = np.zeros((Nx, Nx))
    
    for i in range(Nx):
        x1 = np.copy(x)
        x2 = np.copy(x)
        x1[i] += delta
        x2[i] -= delta
        f1 = rocket_dynamics_rk4(x1, u, m, J_B, r_T_B, g_I,h)
        f2 = rocket_dynamics_rk4(x2, u, m, J_B, r_T_B, g_I,h)
        jacobian_x[:, i] = (f1 - f2) / (2 * delta)
    
    return jacobian_x

def compute_jacobian_u_fd(x, u, delta, rocket_dynamics_rk4, m, J_B, r_T_B, g_I,h):
    Nu = len(u)
    Nx = len(x)
    jacobian_u = np.zeros((Nx, Nu))
    
    for i in range(Nu):
        u1 = np.copy(u)
        u2 = np.copy(u)
        u1[i] += delta
        u2[i] -= delta
        f1 = rocket_dynamics_rk4(x, u1, m, J_B, r_T_B, g_I,h)
        f2 = rocket_dynamics_rk4(x, u2, m, J_B, r_T_B, g_I,h)
        jacobian_u[:, i] = (f1 - f2) / (2 * delta)
    
    return jacobian_u


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
print("u0",u0)
x0 = X_ref[:, 0]
print("x0",x0)

Nx = len(x0)
Nu = len(u0)

x = jnp.zeros((Nx, Nt))
u = jnp.zeros((Nu, Nt))
x = x.at[:, 0].set(x0)

Q = 100000000 * jnp.diag(jnp.array([10, 10, 10, 10, 10, 10, 10, 10, 100, 10, 10, 10]))
R = jnp.diag(jnp.array([1, 1, 1, 100]))
error_vect = jnp.zeros((16, Nt))

Ep1 = E(X_ref[3:7, 1])
Ep = E(X_ref[3:7, 0])

# AA = compute_jacobian_x_fd(x0, u0, 1e-9, rocket_dynamics_rk4, m, J_B, r_T_B, g_I,h)
# BB =compute_jacobian_u_fd(x0, u0, 1e-5, rocket_dynamics_rk4, m, J_B, r_T_B, g_I,h)
AA = compute_jacobian_x(x0, u0, m, J_B, r_T_B, g_I)
BB = compute_jacobian_u(x0, u0, m, J_B, r_T_B, g_I)
#print("BB",BB)
A_tilde = Ep1.T @ AA @ Ep
B_tilde = Ep1.T @ BB
print(A_tilde)
print("rank A_tilde",matrix_rank(A_tilde))
print(B_tilde)
print("rank B_tilde",matrix_rank(B_tilde))

# P = solve_discrete_are(A_tilde, B_tilde, Q, R)
# K = jnp.linalg.inv(R + B_tilde.T @ P @ B_tilde) @ (B_tilde.T @ P @ A_tilde)
C = ct.ctrb(AA,BB)

C_tilde = ct.ctrb(A_tilde,B_tilde)

K,S,EE = control.dlqr(A_tilde, B_tilde, Q, R)
K = np.array(K)  # Convert K to a numpy array
K = jnp.array(K)

for k in range(Nt - 1):
    n_iter = 400
    if k % n_iter == 0 and k > 0:
        print(k)
        AA = compute_jacobian_x(X_ref[:, k], U_ref[:, k], m, J_B, r_T_B, g_I)
        BB = compute_jacobian_u(X_ref[:, k], U_ref[:, k], m, J_B, r_T_B, g_I)
        # AA = compute_jacobian_x_fd(X_ref[:, k], U_ref[:, k], 1e-6, rocket_dynamics_rk4, m, J_B, r_T_B, g_I,h)
        # BB =compute_jacobian_u_fd(X_ref[:, k], U_ref[:, k], 1e-6, rocket_dynamics_rk4, m, J_B, r_T_B, g_I,h)
        Ep1 = E(X_ref[3:7, k + 1])
        Ep = E(X_ref[3:7, k])

        A_tilde = Ep1.T @ AA @ Ep
        B_tilde = Ep1.T @ BB

        # P = solve_discrete_are(A_tilde, B_tilde, Q, R)
        # K = jnp.linalg.inv(R + B_tilde.T @ P @ B_tilde) @ (B_tilde.T @ P @ A_tilde)
        K,S,EE= control.dlqr(A_tilde, B_tilde, Q, R)
        K = np.array(K)  # Convert K to a numpy array
        K = jnp.array(K)
    u = u.at[:, k].set(controller(x[:, k], K,X_ref[:,k],U_ref[:,k]))
    x = x.at[:, k + 1].set(rocket_dynamics_rk4(x[:, k], u[:, k], m, J_B, r_T_B, g_I,h))

#plot_results(t, x, X_ref, u, U_ref,error_vect)