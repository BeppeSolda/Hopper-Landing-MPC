import matplotlib.pyplot as plt
import numpy as np

def plot_results(t_new, x, u ):
   
    zeros_row = np.zeros((1, 3))
    I = np.eye(3)
    H = np.vstack((zeros_row, I))
    vector = np.array([1, -1, -1, -1])
    T = np.diag(vector)

    # Plot position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x[0, :], x[1, :], x[2, :], 'b', label='Actual')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title('3D Position Trajectory')
    ax.grid(True)

 
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].plot(t_new, x[0, :], label='x')
    
    axs[0, 0].set_ylabel('x')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[1, 0].plot(t_new, x[1, :], label='y')
   
    axs[1, 0].set_ylabel('y')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[2, 0].plot(t_new, x[2, :], label='z')
   
    axs[2, 0].set_ylabel('z')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    axs[0, 2].plot(t_new, x[7, :], label='vx')
    
    axs[0, 2].set_ylabel('v_x')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    axs[1, 2].plot(t_new, x[8, :], label='vy')
    
    axs[1, 2].set_ylabel('v_y')
    axs[1, 2].legend()
    axs[1, 2].grid(True)
    
    axs[2, 2].plot(t_new, x[9, :], label='vz')
    
    axs[2, 2].set_ylabel('v_z')
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    # Plot angular rates
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t_new, x[10, :], label='p')
    
    axs[0].set_ylabel('p')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(t_new, x[11, :], label='q')
    
    axs[1].set_ylabel('q')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(t_new, x[12, :], label='r')
    
    axs[2].set_ylabel('r')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot control inputs
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t_new, u[0, :], label='Fx')
    
    axs[0].set_ylabel('u_1')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(t_new, u[1, :], label='Fy')
    
    axs[1].set_ylabel('u_2')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(t_new, u[2, :], label='Fz')
    axs[2].plot(t_new, 2*100*9.81, label='Fz upper limit')
    axs[2].set_ylabel('u_3')
    axs[2].legend()
    axs[2].grid(True)
    
    axs[3].plot(t_new, u[3, :], label='u4')

    axs[3].set_ylabel('u4')
    axs[3].grid(True)
    
    # Plot quaternions
    fig, axs = plt.subplots(4, 1)
    axs[0]
    plt.show()