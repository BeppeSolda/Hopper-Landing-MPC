import jax.numpy as jnp
from trajectory import Trajectory
from mpc import MPC
#from dynamics import rocket_dynamics
from params import J_B, r_T_B, g_I, N_mpc, Q, R, T, H, Max_Thrust,m
from plot import plot_results


# Initialize trajectory and initial conditions
trajectory = Trajectory()
x0, u0 = trajectory.get_initial_conditions()
h = trajectory.h
print(h)
# Create MPC object
mpc = MPC(J_B, r_T_B, g_I,  h, N_mpc, Q, R, T, H,m)

# Solve MPC
x, u = mpc.solve_mpc(x0, trajectory.X_ref, trajectory.U_ref, 13, 3, trajectory.Nt, Max_Thrust)



# Plot results
plot_results(trajectory.t, x, trajectory.X_ref, u, trajectory.U_ref)
