import matplotlib.pyplot as plt
import numpy as np

def plot_results_tracker(t_new, x, X_ref_new, u, U_ref_new, error_vect):
    X_ref = X_ref_new
    U_ref = U_ref_new
    zeros_row = np.zeros((1, 3))
    I = np.eye(3)
    H = np.vstack((zeros_row, I))
    vector = np.array([1, -1, -1, -1])
    T = np.diag(vector)

    # Plot position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_ref[0, :], X_ref[1, :], X_ref[2, :], 'r', label='Reference')
    ax.plot(x[0, :], x[1, :], x[2, :], 'b', label='Actual')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title('3D Position Trajectory')
    ax.grid(True)

    # for k in range(0, len(t_new), 100):
    #     Q_ref = qtoQ(X_ref[4:7, k], H, T)
    #     plotframe(Q_ref, X_ref[0:3, k], 8, LabelBasis=True)
        
    #     Q_actual = qtoQ(x[4:7, k], H, T)
    #     plotframe(Q_actual, x[0:3, k], 8, BasisColors=['k', 'm', 'c'], LabelBasis=True)

    # Plot x, y, z positions and velocities
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].plot(t_new, x[0, :], label='x')
    axs[0, 0].plot(t_new, X_ref[0, :], 'r--', label='x ref')
    axs[0, 0].set_ylabel('x')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[1, 0].plot(t_new, x[1, :], label='y')
    axs[1, 0].plot(t_new, X_ref[1, :], 'g--', label='y ref')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[2, 0].plot(t_new, x[2, :], label='z')
    axs[2, 0].plot(t_new, X_ref[2, :], 'b--', label='z ref')
    axs[2, 0].set_ylabel('z')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    axs[0, 2].plot(t_new, x[7, :], label='vx')
    axs[0, 2].plot(t_new, X_ref[7, :], 'r--', label='vx ref')
    axs[0, 2].set_ylabel('v_x')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    axs[1, 2].plot(t_new, x[8, :], label='vy')
    axs[1, 2].plot(t_new, X_ref[8, :], 'g--', label='vy ref')
    axs[1, 2].set_ylabel('v_y')
    axs[1, 2].legend()
    axs[1, 2].grid(True)
    
    axs[2, 2].plot(t_new, x[9, :], label='vz')
    axs[2, 2].plot(t_new, X_ref[9, :], 'b--', label='vz ref')
    axs[2, 2].set_ylabel('v_z')
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    # Plot angular rates
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t_new, x[10, :], label='p')
    axs[0].plot(t_new, X_ref[10, :], 'r--', label='p ref')
    axs[0].set_ylabel('p')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(t_new, x[11, :], label='q')
    axs[1].plot(t_new, X_ref[11, :], 'g--', label='q ref')
    axs[1].set_ylabel('q')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(t_new, x[12, :], label='r')
    axs[2].plot(t_new, X_ref[12, :], 'b--', label='r ref')
    axs[2].set_ylabel('r')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot control inputs
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t_new, u[0, :], label='Fx')
    axs[0].plot(t_new, U_ref[0, :], label='Fx ref')
    axs[0].set_ylabel('u_1')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(t_new, u[1, :], label='Fy')
    axs[1].plot(t_new, U_ref[1, :], label='Fy ref')
    axs[1].set_ylabel('u_2')
    axs[1].legend()
    axs[1].grid(True)
    h = t_new[1] - t_new[0]
    T_final = t_new[-1]
    Nt = int(T_final / h) + 1
    Fz_up_lim = np.ones((Nt, 1)) * 2*100*9.81
    Fz_low_lim = np.ones((Nt, 1)) * 2*100*9.81 * 0.25
    axs[2].plot(t_new, u[2, :], label='Fz')
    axs[2].plot(t_new, U_ref[2, :], label='Fz ref')
    axs[2].plot(t_new, Fz_up_lim, label='Fz upper limit')
    axs[2].plot(t_new, Fz_low_lim, label='Fz lower limit')
    axs[2].set_ylabel('u_3')
    axs[2].legend()
    axs[2].grid(True)
    
   
    
    # Plot quaternions
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t_new, x[3, :], label='q0')
    axs[0].plot(t_new, X_ref[3, :], 'r--', label='q0 ref')
    axs[0].set_ylabel('q0')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(t_new, x[4, :], label='q1')
    axs[1].plot(t_new, X_ref[4, :], 'g--', label='q1 ref')
    axs[1].set_ylabel('q1')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(t_new, x[5, :], label='q2')
    axs[2].plot(t_new, X_ref[5, :], 'b--', label='q2 ref')
    axs[2].set_ylabel('r')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(t_new, x[6, :], label='q3')
    axs[3].plot(t_new, X_ref[6, :], 'b--', label='q3 ref')
    axs[3].set_ylabel('r')
    axs[3].legend()
    axs[3].grid(True)


    fig, axs = plt.subplots(16, 1)
    axs[0].plot(t_new, error_vect[0, :], label='x')
    axs[0].grid(True)
    
    axs[1].plot(t_new, error_vect[1, :], label='y')
    axs[1].grid(True)
    
    axs[2].plot(t_new, error_vect[2, :], label='z')
    axs[2].grid(True)

  
    axs[3].plot(t_new, error_vect[3, :], label='q0')
    axs[3].grid(True)
    
    axs[4].plot(t_new, error_vect[4, :], label='q1')
    axs[4].grid(True)
    
    axs[5].plot(t_new, error_vect[5, :], label='q2')
    axs[5].grid(True)

    axs[6].plot(t_new, error_vect[6, :], label='q3')
    axs[6].grid(True)

    axs[7].plot(t_new, error_vect[7, :], label='vx')
    axs[7].grid(True)
    
    axs[8].plot(t_new, error_vect[8, :], label='vy')
    axs[8].grid(True)
    
    axs[9].plot(t_new, error_vect[9, :], label='vz')
    axs[9].grid(True)

    axs[10].plot(t_new, error_vect[10, :], label='p')
    axs[10].grid(True)
    
    axs[11].plot(t_new, error_vect[11, :], label='q')
    axs[11].grid(True)
    
    axs[12].plot(t_new, error_vect[12, :], label='r')
    axs[12].grid(True)

    axs[13].plot(t_new, error_vect[13, :], label='ux')
    axs[13].grid(True)
    
    axs[14].plot(t_new, error_vect[14, :], label='uy')
    axs[14].grid(True)
    
    axs[15].plot(t_new, error_vect[15, :], label='uz')
    axs[15].grid(True)

    plt.show()