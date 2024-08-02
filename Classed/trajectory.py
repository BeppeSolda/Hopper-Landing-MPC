from get_reference_trajectoryClass import X_ref, U_ref, t

class Trajectory:
    def __init__(self):
        self.X_ref = X_ref.T
        self.U_ref = U_ref.T
        self.t = t
        self.T_final = t[-1]
        self.h = t[1] - t[0]
        self.Nt = int(self.T_final / self.h) + 1

    def get_initial_conditions(self):
        return self.X_ref[:, 0], self.U_ref[:, 0]
