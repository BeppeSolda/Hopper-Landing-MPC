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

# MPC parameters
N_mpc = 5
Q = jnp.identity(13)
R = jnp.identity(3)
Max_Thrust = 2*m*g  # Maximum thrust value
Min_Thrust = 0.25*Max_Thrust