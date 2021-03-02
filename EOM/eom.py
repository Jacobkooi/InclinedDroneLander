from numba import jit
import numpy as np
from math import cos, sin, pi, sqrt

# Equations of motion used for the 2d and 3d Crazyflie 2.1 quadrotor
# Use jit decorator to greatly increase function speed

@jit
def pwm_to_force(pwm):

    force = 4*(2.130295e-11*(pwm**2) + 1.032633e-6*pwm+5.485e-4)
    return force

@jit
def force_to_pwm(force):  # Identified by Julian Forster
    a = 4*2.130295e-11
    b = 4*1.032633e-6
    c = 5.485e-4 - force
    d = b**2 - 4*a*c
    pwm = (-b + sqrt(d))/(2*a)
    return pwm

@jit
def eom2d_crazyflie_closedloop(x, u, param):

    # States are: [x, z, x_dot. z_dot, Theta, thrust_state]
    # u = [PWM_c, Theta_c] = [10000 to 60000, -1 to 1]
    # param = [mass, gain constant, time constant]

    pwm_commanded = u[0]
    a_ss = -15.4666                                                  # State space A
    b_ss = 1                                                         # State space B
    c_ss = 3.5616e-5                                                 # State space C
    d_ss = 7.2345e-8                                                 # State space AD
    force = 4*(c_ss*x[5] + d_ss*pwm_commanded)                        # Thrust force
    pwm_drag = force_to_pwm(force)                                   # Symbolic PWM to approximate rotor drag
    dragx = 9.1785e-7*4*(0.04076521*pwm_drag + 380.8359)             # Fa,x
    dragz = 10.311e-7*4*(0.04076521*pwm_drag + 380.8359)             # Fa,z
    theta_commanded = u[1]*pi/6                                      # Commanded theta in radians
    dx = np.array([x[2],                                                 # x_dot
                   x[3],                                                 # z_dot
                   (sin(x[4])*(force-dragx*x[2]))/param[0],              # x_ddot
                   (cos(x[4])*(force-dragz*x[3]))/param[0] - 9.81,       # z_ddot
                   (param[1]*theta_commanded - x[4])/param[2],           # Theta_dot
                   a_ss*x[5] + b_ss*pwm_commanded],                      # Thrust_state dot
                   dtype =np.float32)
    return dx

@jit
def eom3d_crazyflie_closedloop(x, u, param):

    # States are: [x, y, z, x_dot, y_dot, z_dot, phi, theta, thrust_state]
    # u = [PWM_c, Phi_c, Theta_c] = [10000 to 60000, -1 to 1, -1 to 1]
    # param = [mass, gain constant, time constant]
    # Note that we use the rotation matrice convention according to the paper, with yaw = 0.

    pwm_commanded = u[0]
    a_ss = -15.4666                                                  # State space A
    b_ss = 1                                                         # State space B
    c_ss = 3.5616e-5                                                 # State space C
    d_ss = 7.2345e-8                                                 # State space AD
    force = 4*(c_ss*x[5] +d_ss*pwm_commanded)                        # Thrust force
    pwm_drag = force_to_pwm(force)                                   # Symbolic PWM to approximate rotor drag
    dragxy = 9.1785e-7*4*(0.04076521*pwm_drag + 380.8359)            # Fa,xy
    dragz = 10.311e-7*4*(0.04076521*pwm_drag + 380.8359)             # Fa,z
    phi_commanded = u[1]*pi/6                                        # Commanded phi in radians
    theta_commanded = u[2]*pi/6                                      # Commanded theta in radians
    dx = np.array([x[3],                                                          # x_dot
                   x[4],                                                          # y_dot
                   x[5],                                                          # z_dot
                   (sin(x[7]))*(force - dragxy*x[3])/param[0],                     # x_ddot
                   (sin(x[6])*cos(x[7])) * (force - dragxy*x[4])/param[0],        # y_ddot
                   (cos(x[6])*cos(x[7])) * (force - dragz*x[5])/param[0] - 9.81,  # z_ddot
                   (param[1]*phi_commanded - x[6])/param[2],                      # Phi_dot
                   (param[1]*theta_commanded - x[7]) / param[2],                # Theta_dot
                   a_ss*x[8] + b_ss*pwm_commanded],                               # Thrust_state dot
                  dtype=np.float32)
    return dx