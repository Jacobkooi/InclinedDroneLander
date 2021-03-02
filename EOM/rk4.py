

def runge_kutta4(x, u, eom, param, t_s):  # Runge Kutta taking the EOM as an argument

    k1 = eom(x, u, param)
    k2 = eom(x + 0.5 * t_s * k1, u, param)
    k3 = eom(x + 0.5 * t_s * k2, u, param)
    k4 = eom(x + t_s * k3, u, param)
    return (t_s / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
