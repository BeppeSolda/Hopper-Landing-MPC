import numpy as np
import json

def parse_lqr_traj_parameters(filename):
    """
    Parse the lqr_traj_parameters.txt file to extract all data.

    Parameters:
    filename (str): Path to the lqr_traj_parameters.txt file

    Returns:
    tuple: U_ref, X_ref, t
    """
    with open(filename, 'r') as file:
        data = json.load(file)  # Use json to parse the dictionary from file

    # Extract necessary arrays
    e1bx = data['e1bx']
    e1by = data['e1by']
    e1bz = data['e1bz']
    e2bx = data['e2bx']
    e2by = data['e2by']
    e2bz = data['e2bz']
    e3bx = data['e3bx']
    e3by = data['e3by']
    e3bz = data['e3bz']
    x = data['x']
    y = data['y']
    z = data['z']
    t = data['t']
    omega = data['omega']
    p = omega[0]
    q = omega[1]
    r = omega[2]
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']

    Fx = data['f1']
    Fy = data['f2']
    Fz = data['f3']

    # Number of samples
    num_samples = len(e1bx)

    # Initialize quaternions
    quaternions = np.zeros((4, num_samples))

    for i in range(num_samples):
        R = np.array([
            [e1bx[i], e2bx[i], e3bx[i]],
            [e1by[i], e2by[i], e3by[i]],
            [e1bz[i], e2bz[i], e3bz[i]]
        ])

        quaternions[:, i] = rotation_matrix_to_quaternion(R)

    # Convert data to arrays and reshape
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))
    z = np.reshape(z, (-1, 1))
    vx = np.reshape(vx, (-1, 1))
    vy = np.reshape(vy, (-1, 1))
    vz = np.reshape(vz, (-1, 1))
    p = np.reshape(p, (-1, 1))
    q = np.reshape(q, (-1, 1))
    r = np.reshape(r, (-1, 1))

    #ur = np.zeros((1, num_samples))

    U_ref = np.vstack((Fx, Fy, Fz)).T

    X_ref = np.hstack((x, y, z, quaternions.T, vx, vy, vz, p, q, r))

    return U_ref, X_ref, t

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters:
    R (numpy.ndarray): 3x3 rotation matrix

    Returns:
    numpy.ndarray: Quaternion [q0, q1, q2, q3]
    """
    q0_mag = np.sqrt(abs((1 + R[0, 0] + R[1, 1] + R[2, 2]) / 4))
    q1_mag = np.sqrt(abs((1 + R[0, 0] - R[1, 1] - R[2, 2]) / 4))
    q2_mag = np.sqrt(abs((1 - R[0, 0] + R[1, 1] - R[2, 2]) / 4))
    q3_mag = np.sqrt(abs((1 - R[0, 0] - R[1, 1] + R[2, 2]) / 4))

    if q0_mag > q1_mag and q0_mag > q2_mag and q0_mag > q3_mag:
        q0 = q0_mag
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    elif q1_mag > q0_mag and q1_mag > q2_mag and q1_mag > q3_mag:
        q1 = q1_mag
        q0 = (R[2, 1] - R[1, 2]) / (4 * q1)
        q2 = (R[0, 1] + R[1, 0]) / (4 * q1)
        q3 = (R[0, 2] + R[2, 0]) / (4 * q1)
    elif q2_mag > q0_mag and q2_mag > q1_mag and q2_mag > q3_mag:
        q2 = q2_mag
        q0 = (R[0, 2] - R[2, 0]) / (4 * q2)
        q1 = (R[0, 1] + R[1, 0]) / (4 * q2)
        q3 = (R[1, 2] + R[2, 1]) / (4 * q2)
    else:
        q3 = q3_mag
        q0 = (R[1, 0] - R[0, 1]) / (4 * q3)
        q1 = (R[0, 2] + R[2, 0]) / (4 * q3)
        q2 = (R[1, 2] + R[2, 1]) / (4 * q3)

    return np.array([q0, q1, q2, q3])

# Example usage
filename = 'lqr_traj_parameters_vertTilted.txt'
#filename = 'infinity_traj_parameters.txt'
U_ref, X_ref, t = parse_lqr_traj_parameters(filename)

# Now U_ref and X_ref are ready for use in further computations or visualizations
