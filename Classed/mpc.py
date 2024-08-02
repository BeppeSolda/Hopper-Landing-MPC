import cvxpy as cp
import jax.numpy as jnp
from jax import jacfwd, jit
from dynamics import rocket_dynamics_rk4
from params import J_B, r_T_B, g_I, T, H, h, m
import time

@jit
def compute_jacobian_x( x, u, m,J_B, r_T_B, g_I, h):
    dynamics_fn = lambda x: rocket_dynamics_rk4(x, u, m,J_B,  r_T_B, g_I, h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(x)

@jit
def compute_jacobian_u( x, u, m,J_B, r_T_B, g_I, h):
    dynamics_fn = lambda u: rocket_dynamics_rk4(x, u, m,J_B,  r_T_B, g_I, h)
    jacobian_fn = jacfwd(jit(dynamics_fn))
    return jacobian_fn(u)

class MPC:
    def __init__(self, J_B, r_T_B, g_I, h, N_mpc, Q, R, T, H,m):
        self.J_B = J_B
        self.r_T_B = r_T_B
        self.g_I = g_I
        self.h = h
        self.N_mpc = N_mpc
        self.Q = Q
        self.R = R
        self.T = T
        self.H = H
        self.m = m

   

    def solve_mpc(self, x0, X_ref, U_ref, Nx, Nu, Nt, Max_Thrust):
        x = jnp.zeros((Nx, Nt))
        u = jnp.zeros((Nu, Nt))
        x = x.at[:, 0].set(x0)

        for k in range(Nt - 1):
            if k % 1 == 0:
                start_time = time.time()
                A_start = time.time()
                AA = compute_jacobian_x(X_ref[:, k], U_ref[:, k],self.m,self.J_B,  self.r_T_B, self.g_I, self.h)
                BB = compute_jacobian_u(X_ref[:, k], U_ref[:, k],self.m,self.J_B, self.r_T_B, self.g_I, self.h)
                A_end = time.time()
                if k + self.N_mpc + 1 > Nt:
                    self.N_mpc = Nt - k - 1

                x_mpc = cp.Variable((Nx, self.N_mpc + 1))
                u_mpc = cp.Variable((Nu, self.N_mpc))
                cost = 0
                constraints = [x_mpc[:, 0] == x[:, k]]
                MPC_cost_start = time.time()
                for i in range(self.N_mpc):
                    cost += cp.quad_form(x_mpc[:, i] - X_ref[:, k + i], self.Q)
                    constraints += [x_mpc[:, i + 1] == AA @ x_mpc[:, i] + BB @ u_mpc[:, i]]

                #cost += cp.quad_form(x_mpc[:, self.N_mpc] - X_ref[:, k + self.N_mpc], self.Q)
                MPC_cost_end = time.time()
                MPC_solve_start = time.time()
                problem = cp.Problem(cp.Minimize(cost), constraints)
                problem.solve(solver=cp.SCS, verbose=False)
                MPC_solve_end = time.time()
                u_opt = u_mpc[:, 0].value

                end_time = time.time()
                print(f"Iteration {k}:")
                print(f"  Total MPC time: {end_time - start_time:.4f} seconds")
                print(f"    Linearization time: {A_end - A_start:.4f} seconds")
                print(f"    Cost function setup time: {MPC_cost_end - MPC_cost_start:.4f} seconds")
                print(f"    Solver time: {MPC_solve_end - MPC_solve_start:.4f} seconds")

            else:
                u_opt = u[:, k - 1]
            u = u.at[:, k].set(u_opt)
            x = x.at[:, k + 1].set(rocket_dynamics_rk4(x[:, k], u[:, k], self.m,self.J_B, self.r_T_B, self.g_I, self.h))

        return x, u
