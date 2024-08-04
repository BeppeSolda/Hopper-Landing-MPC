import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit
from dynamics import rocket_dynamics_rk4
from params import J_B, r_T_B, g_I, T, H, h, m, Min_Thrust, Max_Thrust, cos_delta_max, tan_delta_max, tr_radius,rho_0,rho_1,rho_2,alpha,beta

import time
import jax.random as jrandom

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

def get_linear_cost(self):
        cost = np.sum(self.s_prime.value)
        return cost
def get_nonlinear_cost(self, X=None, U=None):
        magnitude = np.linalg.norm(U, 2, axis=0)
        is_violated = magnitude < self.Min_Thrust
        violation = self.Min_Thrust - magnitude
        cost = np.sum(is_violated * violation)
        return cost


class MPC:
    def __init__(self, J_B, r_T_B, g_I, h, N_mpc, Q, R, T, H,m,Nt,rho_0,rho_1,rho_2,alpha,beta):
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
        self.Min_Thrust = Min_Thrust
        self.Max_Thrust = Max_Thrust
        self.cos_delta_max = cos_delta_max
        self.tan_delta_max = tan_delta_max
        #self.s_prime = cp.Variable((Nt, 1), nonneg=True)
        
        self.Nt = Nt
        self.rho_0 = rho_0
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.alpha = alpha
        self.beta = beta

   

    def solve_mpc(self, x0, X_ref, U_ref, Nx, Nu, Nt, Max_Thrust,tr_radius):
        x = jnp.zeros((Nx, Nt))
        u = jnp.zeros((Nu, Nt))
        error_vect = jnp.zeros((Nx+Nu, Nt))
        x = x.at[:, 0].set(x0)

        measurement_noise_std = 0.1  # Standard deviation of measurement noise
        process_noise_std = 0.001  # Standard deviation of process noise
        key = jrandom.PRNGKey(0) 
        
        for k in range(Nt - 1):
            
                start_time = time.time()
                A_start = time.time()
                AA = compute_jacobian_x(x[:, k], u[:, k-1],self.m,self.J_B,  self.r_T_B, self.g_I, self.h)
                BB = compute_jacobian_u(x[:, k], u[:, k-1],self.m,self.J_B, self.r_T_B, self.g_I, self.h)
                A_end = time.time()
                if k + self.N_mpc + 1 > Nt:
                    self.N_mpc = Nt - k - 1

                x_mpc = cp.Variable((Nx, self.N_mpc + 1))
                u_mpc = cp.Variable((Nu, self.N_mpc))
                s_prime = cp.Variable((Nt, 1), nonneg=True)
                nu = cp.Variable((Nx,self.N_mpc))
                
                x_mpc_sub = cp.Variable((Nx, 10 + 1))
                u_mpc_sub = cp.Variable((Nu, 10))
                
                u_lin_next_iter = cp.Parameter((Nu,self.N_mpc))
                x_lin_next_iter = cp.Parameter((Nx,self.N_mpc))
                x_lin_next_iter.value = X_ref[:,0:self.N_mpc]
                u_lin_next_iter.value = U_ref[:,0:self.N_mpc]
                
                tr_radius = cp.Parameter(nonneg = True)
                tr_radius.value = 5

                
                
                cost = 10*cp.sum(s_prime)
                
                cost_sub = 0
                constraints = [x_mpc[:, 0] == x[:, k]]
                constraints_sub = [x_mpc_sub[:, 0] == x[:, k]]
                # if k==0:
                #     u_opt = jnp.array([0,0,981])
                MPC_cost_start = time.time()
                iter = 0
                while True:
                    
                    for i in range(self.N_mpc):
                        cost += cp.quad_form(x_mpc[:, i] - X_ref[:, k + i], self.Q) + 1e3*cp.norm(nu[:,i],1)
                        #cost_sub+= cp.quad_form(x_mpc_sub[:, i] - X_ref[:, k + i], self.Q)

                
                        du = u_mpc[:,i] - u_lin_next_iter[:,i]
                        dx = x_mpc[:,i] - x_lin_next_iter[:,i]
                        constraints += [x_mpc[:, i + 1] == AA @ x_mpc[:, i] + BB @ u_mpc[:, i] + nu[:,i], 
                                        self.Min_Thrust -u_lin_next_iter[:, i]/cp.norm(u_lin_next_iter[:,i]).T @ u_mpc[:,i] <= s_prime,
                                        cp.norm(u_mpc[:,i]) <= self.Max_Thrust,
                                        cp.norm(u_mpc[0:2,i], axis = 0) <= self.tan_delta_max * u_mpc[2,:],
                                        cp.norm(dx,1) + cp.norm(du,1) <= tr_radius
                                        ]
                        
                    cost += cp.quad_form(x_mpc[:, self.N_mpc] - X_ref[:, k + self.N_mpc], self.Q)
                    MPC_cost_end = time.time()
                    MPC_solve_start = time.time()
                    problem = cp.Problem(cp.Minimize(cost), constraints)
                    problem.solve(solver=cp.SCS, verbose=False)
                    MPC_solve_end = time.time()
                    nonlinear_cost_dynamics = 0
                    linear_cost_constraints = 0
                    nonlinear_cost_constraints = 0
                    linear_cost_dynamics = 0
                    linear_cost = 0
                    nonlinear_cost = 0
                    u_calc = u_mpc.value
                    x_calc = x_mpc.value
                    nu_calc = nu.value
                    s_prime_val = s_prime.value
                    
                    x_nl = jnp.zeros((Nx,self.N_mpc+1))
                    x_nl = x_nl.at[:,0].set(x[:,k])
                    for ii in range (self.N_mpc):
                        x_nl= x_nl.at[:,ii].set(rocket_dynamics_rk4(x[:,k],u_mpc[:,ii].value,self.m, self.J_B, self.r_T_B, self.g_I, self.h))
                        linear_cost_constraints += s_prime_val[ii]

                    nonlinear_cost_dynamics += np.linalg.norm(x_calc - x_nl, 1)
                    linear_cost_dynamics += np.linalg.norm(nu_calc)
                    nonlinear_cost_constraints = get_nonlinear_cost(self, X=x_calc, U=u_calc)
                    

                    
                    linear_cost = linear_cost_constraints + linear_cost_dynamics
                    nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints
                    if iter == 0:
                         last_nonlinear_cost = nonlinear_cost
                         print('dioboia')
                    # if last_nonlinear_cost is None:
                    #      last_nonlinear_cost = nonlinear_cost

                    actual_change = last_nonlinear_cost - nonlinear_cost  # delta_J
                    predicted_change = last_nonlinear_cost - linear_cost  # delta_L
                    print('')
                    print('Virtual Control Cost', linear_cost_dynamics)
                    print('Constraint Cost', linear_cost_constraints)
                    print('')
                    print('Actual change', actual_change)
                    print('Predicted change', predicted_change)
                    print('')
                    if np.abs(predicted_change) < 1e-6:
                        u_opt = u_mpc[:, 0].value
                            
                        u_lin_next_iter.value = u_mpc.value
                        x_lin_next_iter = x_mpc.value

                        end_time = time.time()
                        if k%20 == 0:
                            print(f"Iteration {k}:")
                            print(f"  Total MPC time: {end_time - start_time:.4f} seconds")
                            print(f"    Linearization time: {A_end - A_start:.4f} seconds")
                            print(f"    Cost function setup time: {MPC_cost_end - MPC_cost_start:.4f} seconds")
                            print(f"    Solver time: {MPC_solve_end - MPC_solve_start:.4f} seconds")
                            print(np.linalg.norm(x[3:7,k]))
                            #print(f"  Total MPC subiteration time: {Subiter_end - SubIter_start:.4f} seconds")
                            
                        break
                    else:
                        rho = actual_change / predicted_change
                        print('rho',rho)
                        

                        if rho < self.rho_0:
                            # reject solution
                            tr_radius /= self.alpha
                            print(f'Trust region too large. Solving again with radius=',tr_radius.value)
                            iter +=1
                        else:
                            # accept solution
                            

                            print('Solution accepted.')

                            if rho < self.rho_1:
                                print('Decreasing radius.')
                                tr_radius /= self.alpha
                            elif rho >= self.rho_2:
                                print('Increasing radius.')
                                tr_radius *= self.beta

                            last_nonlinear_cost = nonlinear_cost
                            

                            u_opt = u_mpc[:, 0].value
                            
                            u_lin_next_iter.value = u_mpc.value
                            x_lin_next_iter = x_mpc.value

                            end_time = time.time()
                            if k%1 == 0:
                                print(f"Iteration {k}:")
                                print(f"  Total MPC time: {end_time - start_time:.4f} seconds")
                                print(f"    Linearization time: {A_end - A_start:.4f} seconds")
                                print(f"    Cost function setup time: {MPC_cost_end - MPC_cost_start:.4f} seconds")
                                print(f"    Solver time: {MPC_solve_end - MPC_solve_start:.4f} seconds")
                                
                                #print(f"  Total MPC subiteration time: {Subiter_end - SubIter_start:.4f} seconds")
                                
                                break

            

                key, subkey = jrandom.split(key)
                position_noise = jrandom.normal(subkey, shape=(3,), dtype=jnp.float32) * process_noise_std
                velocity_noise = jrandom.normal(subkey, shape=(3,), dtype=jnp.float32) * process_noise_std
                noisy_x = x[:, k].copy()
                noisy_x = noisy_x.at[0:3].add(position_noise)
                noisy_x = noisy_x.at[7:10].add(velocity_noise)
                # print("Noise",noisy_x)
                # print("x",x[:,k])
                vect = jnp.hstack([X_ref[:,k]-noisy_x, U_ref[:,k]-u_opt])
                error_vect = error_vect.at[:,k].set(vect)
                u = u.at[:, k].set(u_opt)
                x = x.at[:, k + 1].set(rocket_dynamics_rk4(x[:,k], u[:, k], self.m,self.J_B, self.r_T_B, self.g_I, self.h))

        return x, u, error_vect
