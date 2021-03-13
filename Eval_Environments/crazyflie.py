
# Custom gym-like Environment classes for the Crazyflie quadrotor

from EOM.eom import pwm_to_force, force_to_pwm, eom2d_crazyflie_closedloop, eom3d_crazyflie_closedloop
from EOM.rk4 import runge_kutta4
import random as r
import numpy as np
import gym
from os import path
from gym.utils import seeding
from gym import spaces
from math import pi, cos, sin, tan
from Reward.rewardfuncs import sparse_reward2d, euclidean_reward3d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Crazyflie_2d_inclined_eval(gym.Env):
    metadata = {'render.modes': ['human']}

    # state = [x, z, xdot, zdot, theta], action = [Thrust, Theta_commanded]

    def __init__(self, t_s, goal_state=np.array([0, 1.25, 0, 0, -pi/7], dtype=float),
                 episode_steps=300, rewardfunc=sparse_reward2d, eom=eom2d_crazyflie_closedloop,
                 max_pwm_from_hover=15000, param=np.array([0.03303, 1.1094, 0.183806]), rk4=runge_kutta4):
        super(Crazyflie_2d_inclined_eval, self).__init__()

        # Construct the landing polygon
        self.landing_angle = -pi/7
        self.platform_center = goal_state[0]
        self.platform_center_height = 1.15
        self.platform_width = 0.8
        self.landing_polygon = Polygon([(self.platform_center-0.5*self.platform_width, self.platform_center_height-
                                     tan(abs(self.landing_angle))*0.5*self.platform_width),
                                     (self.platform_center+0.5*self.platform_width,
                                         self.platform_center_height +
                                         tan(abs(self.landing_angle))*0.5*self.platform_width),
                                     (self.platform_center+0.5*self.platform_width, 0),
                                     (self.platform_center-0.5*self.platform_width, 0)])

        self.quad_arms = 0.2
        self.obstacle_range = self.quad_arms-0.15

        self.viewer = None
        self.episode_steps = episode_steps
        self.param = param
        self.rewardfunc = rewardfunc
        self.EOM = eom
        self.RK4 = rk4
        self.T_s = t_s
        self.Timesteps = 0
        self.goal_state = goal_state

        # Used for simulations
        self.thrust_state = np.array([0])
        self.real_action = np.array([0, 0])
        self.hover_pwm = force_to_pwm(self.param[0]*9.81)   # Calculate the theoretical hover thrust
        self.max_pwm_from_hover = max_pwm_from_hover

        self.episode_counter = 0
        self.goal_range = 0.10
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]), dtype=np.float)
        # States are: [x, z, x_dot. z_dot, Theta, Theta_dot]
        self.observation_space = spaces.Box(low=np.array([-3.4, 0,  -10, -10, -pi/3]),
                                            high=np.array([3.4, 2.4, 10, 10, pi/3]), dtype=np.float)
        self.reward_range = (-float("inf"), float("inf"))
        self.agent_pos = []
        self.reset()
        self.seed()
        self.counter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Fix PWM increments from its theoretical Hover PWM
        pwm_commanded = self.hover_pwm + action[0]*self.max_pwm_from_hover
        # Compute the action vector that goes into the Equations of Motion
        self.real_action = np.array([pwm_commanded, action[1]], dtype=float)
        # Add the physically unobservable thrust state for simulation purposes
        extended_state = np.concatenate((self.agent_pos, self.thrust_state))
        # Simulate the agent with 1 time step using Runge Kutta 4 and the EOM
        extended_state = extended_state + self.RK4(extended_state, self.real_action, self.EOM, self.param, self.T_s)
        # Subtract the thrust state to form the actual states we want to use in training
        self.agent_pos = extended_state[0:-1]
        self.thrust_state = np.array([extended_state[-1]])
        # Clip the agent's position so it doesn't leave the simulation bounds
        self.agent_pos = np.clip(self.agent_pos, self.observation_space.low, self.observation_space.high)
        observation = self.agent_pos
        reward, done = self.rewardfunc(observation, self.goal_state, self.observation_space, self.goal_range,
                                       self.landing_polygon)
        # Check if the quadrotor touches the landing polygon and diminish horizontal velocity if true
        point = Point(self.agent_pos[0], self.agent_pos[1])
        if self.landing_polygon.contains(point):
            self.agent_pos[2] = 0

        self.counter += 1
        self.Timesteps += 1

        if self.counter == self.episode_steps:
            done = True

        info = {}

        return observation, reward, done, info

    def reset(self):

        # We use the reset function to achieve a form of continuous curriculum learning

        self.episode_counter += 1

        # Start episodes within a box around the goal state
        self.agent_pos = np.array([r.uniform(self.goal_state[0]-3,
                                             self.goal_state[0]+3),
                                   r.uniform(self.goal_state[1]-0.3,
                                             self.goal_state[1]+1), 0, 0, 0], dtype=np.float32)

        while self.landing_polygon.contains(Point(self.agent_pos[0], self.agent_pos[1])):
            self.agent_pos = np.array([np.clip(r.uniform(self.goal_state[0] - 3, self.goal_state[0] + 3),
                                       self.observation_space.low[0], self.observation_space.high[0]),
                                       np.clip(r.uniform(self.goal_state[1] - 0.3, self.goal_state[1] + 1),
                                       self.observation_space.low[1], self.observation_space.high[1]), 0, 0, 0],
                                       dtype=np.float32)

        # Clip position to be in the bounds of the Optitrack Stadium
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.observation_space.low[0] + 0.1,
                                        self.observation_space.high[0] - 0.1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.observation_space.low[1] + 1,
                                        self.observation_space.high[1] - 0.1)

        self.counter = 0

        return self.agent_pos

    def render(self, mode='human'):

        # Rendering function

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(740, 288)
            self.viewer.set_bounds(-3.5, 3.5, -0.5, 2.5)

            rod_quad1 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_quad2 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_goalquad1 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_goalquad2 = rendering.make_capsule(self.quad_arms, 0.05)

            obstacle_1_top = rendering.Line((self.landing_polygon.exterior.coords.xy[0][0],
                                               self.landing_polygon.exterior.coords.xy[1][0]),
                                              (self.landing_polygon.exterior.coords.xy[0][1],
                                               self.landing_polygon.exterior.coords.xy[1][1]))
            obstacle_1_right = rendering.Line((self.landing_polygon.exterior.coords.xy[0][1],
                                               self.landing_polygon.exterior.coords.xy[1][1]),
                                              (self.landing_polygon.exterior.coords.xy[0][2],
                                               self.landing_polygon.exterior.coords.xy[1][2]))
            obstacle_1_bottom = rendering.Line((self.landing_polygon.exterior.coords.xy[0][2],
                                               self.landing_polygon.exterior.coords.xy[1][2]),
                                              (self.landing_polygon.exterior.coords.xy[0][3],
                                               self.landing_polygon.exterior.coords.xy[1][3]))
            obstacle_1_left = rendering.Line((self.landing_polygon.exterior.coords.xy[0][3],
                                               self.landing_polygon.exterior.coords.xy[1][3]),
                                              (self.landing_polygon.exterior.coords.xy[0][4],
                                               self.landing_polygon.exterior.coords.xy[1][4]))

            xaxis = rendering.Line((-3.2, 0), (3.2, 0))
            xmetermin3 = rendering.Line((-3, 0), (-3, 0.05))
            xmetermin2 = rendering.Line((-2, 0), (-2, 0.05))
            xmetermin1 = rendering.Line((-1, 0), (-1, 0.05))
            xmeter0 = rendering.Line((0, -0.10), (0, 0.10))
            xmeter1 = rendering.Line((1, 0), (1, 0.05))
            xmeter2 = rendering.Line((2, 0), (2, 0.05))
            xmeter3 = rendering.Line((3, 0), (3, 0.05))

            zaxis = rendering.Line((-3.2, 0), (-3.2, 2.4))
            zmeter1 = rendering.Line((-3.2, 1), (-3.15, 1))
            zmeter2 = rendering.Line((-3.2, 2), (-3.15, 2))

            rod_quad1.set_color(.8, .3, .3)
            rod_quad2.set_color(.8, .3, .3)
            rod_goalquad1.set_color(.4, .3, .3)
            rod_goalquad2.set_color(.4, .3, .3)

            quadcirc1 = rendering.make_capsule(.03, 0.05)
            quadcirc2 = rendering.make_capsule(.03, 0.05)
            quadcirc3 = rendering.make_capsule(.03, 0.05)
            goalquadcirc1 = rendering.make_capsule(.03, 0.05)
            goalquadcirc2 = rendering.make_capsule(.03, 0.05)
            goalquadcirc3 = rendering.make_capsule(.03, 0.05)

            quadcirc1.set_color(.3, .6, .3)
            quadcirc2.set_color(.3, .6, .3)
            quadcirc3.set_color(.3, .6, .3)
            goalquadcirc1.set_color(.3, .3, .3)
            goalquadcirc2.set_color(.3, .3, .3)
            goalquadcirc3.set_color(.3, .3, .3)

            self.pole_transform = rendering.Transform()
            self.pole2_transform = rendering.Transform()
            self.goalpole_transform = rendering.Transform()
            self.goalpole2_transform = rendering.Transform()

            self.circ1_transform = rendering.Transform()
            self.circ2_transform = rendering.Transform()
            self.circ3_transform = rendering.Transform()
            self.goalcirc1_transform = rendering.Transform()
            self.goalcirc2_transform = rendering.Transform()
            self.goalcirc3_transform = rendering.Transform()

            rod_quad1.add_attr(self.pole_transform)
            rod_quad2.add_attr(self.pole2_transform)
            rod_goalquad1.add_attr(self.goalpole_transform)
            rod_goalquad2.add_attr(self.goalpole2_transform)

            quadcirc1.add_attr(self.circ1_transform)
            quadcirc2.add_attr(self.circ2_transform)
            quadcirc3.add_attr(self.circ3_transform)
            goalquadcirc1.add_attr(self.goalcirc1_transform)
            goalquadcirc2.add_attr(self.goalcirc2_transform)
            goalquadcirc3.add_attr(self.goalcirc3_transform)

            self.viewer.add_geom(rod_quad1)
            self.viewer.add_geom(rod_quad2)
            self.viewer.add_geom(rod_goalquad1)
            self.viewer.add_geom(rod_goalquad2)
            self.viewer.add_geom(xaxis)
            self.viewer.add_geom(xmetermin3)
            self.viewer.add_geom(xmetermin2)
            self.viewer.add_geom(xmetermin1)
            self.viewer.add_geom(xmeter0)
            self.viewer.add_geom(xmeter1)
            self.viewer.add_geom(xmeter2)
            self.viewer.add_geom(xmeter3)
            self.viewer.add_geom(zaxis)
            self.viewer.add_geom(zmeter1)
            self.viewer.add_geom(zmeter2)

            self.viewer.add_geom(quadcirc1)
            self.viewer.add_geom(quadcirc2)
            self.viewer.add_geom(quadcirc3)
            self.viewer.add_geom(goalquadcirc1)
            self.viewer.add_geom(goalquadcirc2)
            self.viewer.add_geom(goalquadcirc3)

            self.viewer.add_geom(obstacle_1_right)
            self.viewer.add_geom(obstacle_1_bottom)
            self.viewer.add_geom(obstacle_1_left)
            self.viewer.add_geom(obstacle_1_top)


        self.pole_transform.set_rotation(-self.agent_pos[4])
        self.pole_transform.set_translation(self.agent_pos[0], self.agent_pos[1])
        self.pole2_transform.set_rotation(-self.agent_pos[4] + pi)
        self.pole2_transform.set_translation(self.agent_pos[0], self.agent_pos[1])

        self.goalpole_transform.set_rotation(-self.goal_state[4])
        self.goalpole_transform.set_translation(self.goal_state[0], self.goal_state[1])
        self.goalpole2_transform.set_rotation(-self.goal_state[4] + pi)
        self.goalpole2_transform.set_translation(self.goal_state[0], self.goal_state[1])

        self.circ1_transform.set_translation(self.agent_pos[0] + self.quad_arms*cos(self.agent_pos[4]),
                                             self.agent_pos[1] - self.quad_arms*sin(self.agent_pos[4]))
        self.circ2_transform.set_translation(self.agent_pos[0],
                                             self.agent_pos[1])
        self.circ3_transform.set_translation(self.agent_pos[0] - self.quad_arms*cos(self.agent_pos[4]),
                                             self.agent_pos[1] + self.quad_arms*sin(self.agent_pos[4]))
        self.circ1_transform.set_rotation(-self.agent_pos[4] -pi/2)
        self.circ2_transform.set_rotation(-self.agent_pos[4] -pi/2)
        self.circ3_transform.set_rotation(-self.agent_pos[4] -pi/2)

        self.goalcirc1_transform.set_translation(self.goal_state[0] + self.quad_arms * cos(self.goal_state[4]),
                                             self.goal_state[1] - self.quad_arms * sin(self.goal_state[4]))
        self.goalcirc2_transform.set_translation(self.goal_state[0],
                                             self.goal_state[1])
        self.goalcirc3_transform.set_translation(self.goal_state[0] - self.quad_arms * cos(self.goal_state[4]),
                                             self.goal_state[1] + self.quad_arms * sin(self.goal_state[4]))
        self.goalcirc1_transform.set_rotation(-self.goal_state[4] - pi / 2)
        self.goalcirc2_transform.set_rotation(-self.goal_state[4] - pi / 2)
        self.goalcirc3_transform.set_rotation(-self.goal_state[4] - pi / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class Crazyflie_3d_setpoint_eval(gym.Env):
    metadata = {'render.modes': ['human']}

    # state = [x, y, z, xdot, ydot, zdot, phi, theta], action = [Thrust, Phi_commanded, Theta_commanded]
    def __init__(self, t_s, goal_state=np.array([0, 0, 1.2, 0, 0, 0, 0, 0]), episode_steps=300,
                 rewardfunc=euclidean_reward3d, eom=eom3d_crazyflie_closedloop, max_pwm_from_hover=13000,
                 param=np.array([0.03303, 1.1094, 0.183806]), rk4=runge_kutta4):
        super(Crazyflie_3d_setpoint_eval, self).__init__()

        self.quad_arms = 0.2
        self.obstacle_range = self.quad_arms - 0.15
        self.viewer = None

        self.episode_steps = episode_steps
        self.counter = 0
        self.param = param
        self.rewardfunc = rewardfunc
        self.EOM = eom
        self.RK4 = rk4
        self.real_action = np.array([0, 0])
        self.T_s = t_s
        self.goal_state = goal_state
        self.episode_counter = 0
        self.goal_range = 0.25

        self.x_spawn_radius = 0.2
        self.y_spawn_radius = 0.2
        self.z_spawn_radius = 0.2
        self.spawn_increment = 1/6000

        self.thrust_state = np.array([0])
        self.hover_pwm = force_to_pwm(self.param[0]*9.81)
        self.max_pwm_from_hover = max_pwm_from_hover

        self.action_space = spaces.Box(low=np.array([-1, -1, -1]),
                                       high=np.array([1, 1, 1]), dtype=np.float32)
        # States are: [x, z, x_dot. z_dot, Theta, Theta_dot]
        self.observation_space = spaces.Box(low=np.array([-3.4, -1.4, 0,  -10, -10, -10, -pi/3, -pi/3]),
                                            high=np.array([3.4, 1.4, 2.4, 10, 10, 10, pi/3, pi/3]), dtype=np.float32)
        self.reward_range = (-float("inf"), float("inf"))
        self.agent_pos = []
        self.reset()
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # Fix PWM increments from its theoretical Hover PWM
        PWM_commanded_new = self.hover_pwm + action[0]*self.max_pwm_from_hover
        # Compute the action vector that goes into the Equations of Motion [PWM, clipped, clipped]
        self.real_action = np.array([PWM_commanded_new, action[1], action[2]], dtype=float)
        # Add the physically unobservable thrust state for simulation purposes
        extended_state = np.concatenate((self.agent_pos, self.thrust_state))
        # Simulate the agent for 1 time step using Runge Kutta 4 and the EOM
        extended_state = extended_state + self.RK4(extended_state, self.real_action, self.EOM, self.param, self.T_s)
        # Subtract the thrust state to form the actual states we want to use in training
        self.agent_pos = extended_state[0:-1]
        self.thrust_state = np.array([extended_state[-1]])
        # Clip the agent's position so it doesn't leave the simulation bounds
        self.agent_pos = np.clip(self.agent_pos, self.observation_space.low, self.observation_space.high)

        observation = self.agent_pos
        reward, done = self.rewardfunc(observation, self.goal_state, self.observation_space, self.real_action)

        self.counter += 1

        if self.counter == self.episode_steps:
            done = True

        info = {}

        return observation, reward, done, info

    def reset(self):
        # Use the reset function to achieve a form of continuous curriculum learning

        # Pick the box of starting positions around the euclidean goal
        self.agent_pos = np.array([r.uniform(self.goal_state[0]-2.5,
                                             self.goal_state[0]+2.5),
                                   r.uniform(self.goal_state[1] - 1.3,
                                             self.goal_state[1] + 1.3),
                                   r.uniform(self.goal_state[2]-1,
                                             self.goal_state[2]+1), 0, 0, 0, 0, 0], dtype=np.float32)

        # Clip starting position to be in the bounds of the Optitrack Stadium
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.observation_space.low[0] + 0.1,
                                        self.observation_space.high[0] - 0.1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.observation_space.low[1] + 0.1,
                                        self.observation_space.high[1] - 0.1)
        self.agent_pos[2] = np.clip(self.agent_pos[2], self.observation_space.low[2] + 0.1,
                                        self.observation_space.high[2] - 0.1)
        self.counter = 0

        return self.agent_pos

    def render(self, mode='human'):

        # Rendering for 3-D performance. Since pyglet did not support 3D we made two 2D views.
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(720, 288)
            self.viewer.set_bounds(-3.5*2, 3.6, -0.5*2, 2.5)

            # Define the rods for making quadrotor bodies
            rod_quad1 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_quad2 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_goalquad1 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_goalquad2 = rendering.make_capsule(self.quad_arms, 0.05)
            # Roll is for the y-z plane
            rod_quadroll1 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_quadroll2 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_goalquadroll1 = rendering.make_capsule(self.quad_arms, 0.05)
            rod_goalquadroll2 = rendering.make_capsule(self.quad_arms, 0.05)

            # Draw the axis
            xaxis = rendering.Line((-6.8, 0), (0, 0))
            xmetermin3 = rendering.Line((-6.4, 0), (-6.4, 0.05))
            xmetermin2 = rendering.Line((-5.4, 0), (-5.4, 0.05))
            xmetermin1 = rendering.Line((-4.4, 0), (-4.4, 0.05))
            xmeter0 = rendering.Line((-3.4, -0.10), (-3.4, 0.10))
            xmeter1 = rendering.Line((-2.4, 0), (-2.4, 0.05))
            xmeter2 = rendering.Line((-1.4, 0), (-1.4, 0.05))
            xmeter3 = rendering.Line((-0.4, 0), (-0.4, 0.05))

            xaxisroll = rendering.Line((0, 0), (6.8, 0))
            xmetermin1roll = rendering.Line((1, 0), (1, 0.05))
            xmeter0roll = rendering.Line((2, -0.10), (2, 0.10))
            xmeter1roll = rendering.Line((3, 0), (3, 0.05))

            zaxis = rendering.Line((-6.8, 0), (-6.8, 2.4))
            zmeter1 = rendering.Line((-6.8, 1), (-6.75, 1))
            zmeter2 = rendering.Line((-6.8, 2), (-6.75, 2))

            zaxisroll = rendering.Line((0.6, 0), (0.6, 2.4))
            zmeter1roll = rendering.Line((0.6, 1), (0.65, 1))
            zmeter2roll = rendering.Line((0.6, 2), (0.65, 2))

            # Define the color for the quadrotor bodies
            rod_quad1.set_color(.8, .3, .3)
            rod_quad2.set_color(.8, .3, .3)
            rod_goalquad1.set_color(.4, .3, .3)
            rod_goalquad2.set_color(.4, .3, .3)

            rod_quadroll1.set_color(.4, .3, .3)
            rod_quadroll2.set_color(.4, .3, .3)
            rod_goalquadroll1.set_color(.2, .3, .3)
            rod_goalquadroll2.set_color(.2, .3, .3)

            # Define the circles on top of the rods
            quadcirc1 = rendering.make_capsule(.03, 0.05)
            quadcirc2 = rendering.make_capsule(.03, 0.05)
            quadcirc3 = rendering.make_capsule(.03, 0.05)
            goalquadcirc1 = rendering.make_capsule(.03, 0.05)
            goalquadcirc2 = rendering.make_capsule(.03, 0.05)
            goalquadcirc3 = rendering.make_capsule(.03, 0.05)

            quadcirc1roll = rendering.make_capsule(.03, 0.05)
            quadcirc2roll = rendering.make_capsule(.03, 0.05)
            quadcirc3roll = rendering.make_capsule(.03, 0.05)
            goalquadcirc1roll = rendering.make_capsule(.03, 0.05)
            goalquadcirc2roll = rendering.make_capsule(.03, 0.05)
            goalquadcirc3roll = rendering.make_capsule(.03, 0.05)

            # Define the circle colors on top of the rod
            quadcirc1.set_color(.3, .6, .3)
            quadcirc2.set_color(.3, .6, .3)
            quadcirc3.set_color(.3, .6, .3)
            goalquadcirc1.set_color(.3, .3, .3)
            goalquadcirc2.set_color(.3, .3, .3)
            goalquadcirc3.set_color(.3, .3, .3)

            quadcirc1roll.set_color(.3, .6, .3)
            quadcirc2roll.set_color(.3, .6, .3)
            quadcirc3roll.set_color(.3, .6, .3)
            goalquadcirc1roll.set_color(.3, .3, .3)
            goalquadcirc2roll.set_color(.3, .3, .3)
            goalquadcirc3roll.set_color(.3, .3, .3)

            # Define the transformations that need to happen at every timestep
            self.pole_transform = rendering.Transform()
            self.pole2_transform = rendering.Transform()
            self.goalpole_transform = rendering.Transform()
            self.goalpole2_transform = rendering.Transform()

            self.poleroll_transform = rendering.Transform()
            self.pole2roll_transform = rendering.Transform()
            self.goalpoleroll_transform = rendering.Transform()
            self.goalpole2roll_transform = rendering.Transform()

            self.circ1_transform = rendering.Transform()
            self.circ2_transform = rendering.Transform()
            self.circ3_transform = rendering.Transform()
            self.goalcirc1_transform = rendering.Transform()
            self.goalcirc2_transform = rendering.Transform()
            self.goalcirc3_transform = rendering.Transform()

            self.circ1roll_transform = rendering.Transform()
            self.circ2roll_transform = rendering.Transform()
            self.circ3roll_transform = rendering.Transform()
            self.goalcirc1roll_transform = rendering.Transform()
            self.goalcirc2roll_transform = rendering.Transform()
            self.goalcirc3roll_transform = rendering.Transform()

            # Assign the transformations to the defined rods and circles
            rod_quad1.add_attr(self.pole_transform)
            rod_quad2.add_attr(self.pole2_transform)
            rod_goalquad1.add_attr(self.goalpole_transform)
            rod_goalquad2.add_attr(self.goalpole2_transform)

            rod_quadroll1.add_attr(self.poleroll_transform)
            rod_quadroll2.add_attr(self.pole2roll_transform)
            rod_goalquadroll1.add_attr(self.goalpoleroll_transform)
            rod_goalquadroll2.add_attr(self.goalpole2roll_transform)

            quadcirc1.add_attr(self.circ1_transform)
            quadcirc2.add_attr(self.circ2_transform)
            quadcirc3.add_attr(self.circ3_transform)
            goalquadcirc1.add_attr(self.goalcirc1_transform)
            goalquadcirc2.add_attr(self.goalcirc2_transform)
            goalquadcirc3.add_attr(self.goalcirc3_transform)

            quadcirc1roll.add_attr(self.circ1roll_transform)
            quadcirc2roll.add_attr(self.circ2roll_transform)
            quadcirc3roll.add_attr(self.circ3roll_transform)
            goalquadcirc1roll.add_attr(self.goalcirc1roll_transform)
            goalquadcirc2roll.add_attr(self.goalcirc2roll_transform)
            goalquadcirc3roll.add_attr(self.goalcirc3roll_transform)

            # Draw the rods, circles and axis in the window defined at the beginning
            self.viewer.add_geom(rod_quad1)
            self.viewer.add_geom(rod_quad2)
            self.viewer.add_geom(rod_goalquad1)
            self.viewer.add_geom(rod_goalquad2)
            self.viewer.add_geom(xaxis)
            self.viewer.add_geom(xmetermin3)
            self.viewer.add_geom(xmetermin2)
            self.viewer.add_geom(xmetermin1)
            self.viewer.add_geom(xmeter0)
            self.viewer.add_geom(xmeter1)
            self.viewer.add_geom(xmeter2)
            self.viewer.add_geom(xmeter3)
            self.viewer.add_geom(zaxis)
            self.viewer.add_geom(zmeter1)
            self.viewer.add_geom(zmeter2)

            self.viewer.add_geom(rod_quadroll1)
            self.viewer.add_geom(rod_quadroll2)
            self.viewer.add_geom(rod_goalquadroll1)
            self.viewer.add_geom(rod_goalquadroll2)
            self.viewer.add_geom(xaxisroll)
            self.viewer.add_geom(xmetermin1roll)
            self.viewer.add_geom(xmeter0roll)
            self.viewer.add_geom(xmeter1roll)
            self.viewer.add_geom(zaxisroll)
            self.viewer.add_geom(zmeter1roll)
            self.viewer.add_geom(zmeter2roll)

            self.viewer.add_geom(quadcirc1)
            self.viewer.add_geom(quadcirc2)
            self.viewer.add_geom(quadcirc3)
            self.viewer.add_geom(goalquadcirc1)
            self.viewer.add_geom(goalquadcirc2)
            self.viewer.add_geom(goalquadcirc3)

            self.viewer.add_geom(quadcirc1roll)
            self.viewer.add_geom(quadcirc2roll)
            self.viewer.add_geom(quadcirc3roll)
            self.viewer.add_geom(goalquadcirc1roll)
            self.viewer.add_geom(goalquadcirc2roll)
            self.viewer.add_geom(goalquadcirc3roll)

        # Define how the rods and circles are transformed at every timestep
        self.pole_transform.set_rotation(-self.agent_pos[7])
        self.pole_transform.set_translation(self.agent_pos[0]-3.4, self.agent_pos[2])
        self.pole2_transform.set_rotation(-self.agent_pos[7] + pi)
        self.pole2_transform.set_translation(self.agent_pos[0]-3.4, self.agent_pos[2])

        self.goalpole_transform.set_rotation(-self.goal_state[7])
        self.goalpole_transform.set_translation(self.goal_state[0]-3.4, self.goal_state[2])
        self.goalpole2_transform.set_rotation(-self.goal_state[7] + pi)
        self.goalpole2_transform.set_translation(self.goal_state[0]-3.4, self.goal_state[2])

        self.poleroll_transform.set_rotation(-self.agent_pos[6])
        self.poleroll_transform.set_translation(self.agent_pos[1]+2, self.agent_pos[2])
        self.pole2roll_transform.set_rotation(-self.agent_pos[6] + pi)
        self.pole2roll_transform.set_translation(self.agent_pos[1]+2, self.agent_pos[2])

        self.goalpoleroll_transform.set_rotation(-self.goal_state[6])
        self.goalpoleroll_transform.set_translation(self.goal_state[1]+2, self.goal_state[2])
        self.goalpole2roll_transform.set_rotation(-self.goal_state[6] + pi)
        self.goalpole2roll_transform.set_translation(self.goal_state[1]+2, self.goal_state[2])

        self.circ1_transform.set_translation(self.agent_pos[0]-3.4 + self.quad_arms*cos(self.agent_pos[7]),
                                             self.agent_pos[2] - self.quad_arms*sin(self.agent_pos[7]))
        self.circ2_transform.set_translation(self.agent_pos[0]-3.4,
                                             self.agent_pos[2])
        self.circ3_transform.set_translation(self.agent_pos[0]-3.4 - self.quad_arms*cos(self.agent_pos[7]),
                                             self.agent_pos[2] + self.quad_arms*sin(self.agent_pos[7]))
        self.circ1_transform.set_rotation(-self.agent_pos[7] - pi/2)
        self.circ2_transform.set_rotation(-self.agent_pos[7] - pi/2)
        self.circ3_transform.set_rotation(-self.agent_pos[7] - pi/2)

        self.goalcirc1_transform.set_translation(self.goal_state[0]-3.4 + self.quad_arms * cos(self.goal_state[7]),
                                                 self.goal_state[2] - self.quad_arms * sin(self.goal_state[7]))
        self.goalcirc2_transform.set_translation(self.goal_state[0]-3.4,
                                                 self.goal_state[2])
        self.goalcirc3_transform.set_translation(self.goal_state[0]-3.4 - self.quad_arms * cos(self.goal_state[7]),
                                                 self.goal_state[2] + self.quad_arms * sin(self.goal_state[7]))
        self.goalcirc1_transform.set_rotation(-self.goal_state[7] - pi / 2)
        self.goalcirc2_transform.set_rotation(-self.goal_state[7] - pi / 2)
        self.goalcirc3_transform.set_rotation(-self.goal_state[7] - pi / 2)

        self.circ1roll_transform.set_translation(self.agent_pos[1]+2 + self.quad_arms * cos(self.agent_pos[6]),
                                                 self.agent_pos[2] - self.quad_arms * sin(self.agent_pos[6]))
        self.circ2roll_transform.set_translation(self.agent_pos[1] + 2,
                                                 self.agent_pos[2])
        self.circ3roll_transform.set_translation(self.agent_pos[1] + 2 - self.quad_arms * cos(self.agent_pos[6]),
                                                 self.agent_pos[2] + self.quad_arms * sin(self.agent_pos[6]))
        self.circ1roll_transform.set_rotation(-self.agent_pos[6] - pi / 2)
        self.circ2roll_transform.set_rotation(-self.agent_pos[6] - pi / 2)
        self.circ3roll_transform.set_rotation(-self.agent_pos[6] - pi / 2)

        self.goalcirc1roll_transform.set_translation(self.goal_state[1] + 2 + self.quad_arms * cos(self.goal_state[6]),
                                                     self.goal_state[2] - self.quad_arms * sin(self.goal_state[6]))
        self.goalcirc2roll_transform.set_translation(self.goal_state[1] + 2,
                                                     self.goal_state[2])
        self.goalcirc3roll_transform.set_translation(self.goal_state[1]+2 - self.quad_arms * cos(self.goal_state[6]),
                                                     self.goal_state[2] + self.quad_arms * sin(self.goal_state[6]))
        self.goalcirc1roll_transform.set_rotation(-self.goal_state[6] - pi / 2)
        self.goalcirc2roll_transform.set_rotation(-self.goal_state[6] - pi / 2)
        self.goalcirc3roll_transform.set_rotation(-self.goal_state[6] - pi / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None







