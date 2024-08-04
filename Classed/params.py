import jax.numpy as jnp
import numpy as np

# Constants
g = 9.81
g_I = jnp.array([0, 0, -g])  # Gravity vector in the inertial frame



# Inertia matrix for the rocket body
m = 100.0
J1 = 1 / 12 * m * (4 + 3 * 0.25 ** 2)
J2 = J1
J3 = 0.5 * m * 0.25 ** 2
J_B = jnp.diag(jnp.array([J1, J2, J3]))
r_T_B = jnp.array([0, 0, -0.5])

# Transform matrices
vector = np.concatenate(([1], -np.ones(3)))
h = 0.025 
# Create the diagonal matrix
T = np.diag(vector)

zeros_row = np.zeros((1, 3))
I = np.eye(3)

# Vertically stack the zeros_row and the identity matrix
H = np.vstack((zeros_row, I))
cos_delta_max = np.cos(np.deg2rad(15))
max_gimbal = 15
tan_delta_max = np.tan(np.deg2rad(max_gimbal))
# MPC parameters
N_mpc = 50
Q = jnp.identity(13)
R = jnp.identity(3)
Max_Thrust = 2*m*g  # Maximum thrust value
Min_Thrust = 0.25*Max_Thrust

tr_radius = 5

rho_0 = 0.0
rho_1 = 0.25
rho_2 = 0.9
alpha = 2.0
beta = 3.2
