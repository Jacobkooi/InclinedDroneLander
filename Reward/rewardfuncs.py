from math import sqrt, pi
from shapely.geometry import Point
from math import cos, sin

# Sparse reward Function used for Inclined landing and Euclidean reward Function used for set-point tracking


def sparse_reward2d(next_state, goal_state, observation_space, goal_range, polygon):

    # goal_range is the vector of goal threshold ranges
    # polygon represents the polygon of the landing platform

    quad_arm_length = 0.10

    goal_reward = -1 * int((abs(next_state[0] - goal_state[0]) > goal_range or
                            abs(next_state[1] - goal_state[1]) > goal_range or
                            abs(next_state[2] - goal_state[2]) > min(10*goal_range, 1.5) or
                            abs(next_state[3] - goal_state[3]) > min(10*goal_range, 1.5) or
                            abs(next_state[4] - goal_state[4]) > goal_range/4))

    bounds_reward = -1 * int((abs(next_state[0] - observation_space.high[0]) < 0.05 or
                        abs(next_state[0] - observation_space.low[0]) < 0.05 or
                        abs(next_state[1] - observation_space.high[1]) < 0.05 or
                        abs(next_state[1] - observation_space.low[1]) < 0.05))

    # Check if it hits the landing platform
    right_point = Point(next_state[0]+cos(next_state[4])*quad_arm_length,
                        next_state[1] - sin(next_state[4])*quad_arm_length)
    left_point = Point(next_state[0]-cos(next_state[4])*quad_arm_length,
                       next_state[1] + sin(next_state[4])*quad_arm_length)
    middle_point = Point(next_state[0], next_state[1])
    if polygon.contains(middle_point) or polygon.contains(right_point) or polygon.contains(left_point):
        obstacle_reward = -6
    else:
        obstacle_reward = 0

    done = (goal_reward == 0)

    total_reward = goal_reward + bounds_reward + obstacle_reward

    return total_reward, done


def euclidean_reward3d(next_state, goal_state, observation_space, action):

    # Position cost
    error_position = sqrt((next_state[0] - goal_state[0])**2 + (next_state[1] - goal_state[1])**2
                          + (next_state[2] - goal_state[2])**2)
    # Velocity cost
    error_velocity = sqrt((next_state[3] - goal_state[3])**2 + (next_state[4] - goal_state[4])**2
                          + (next_state[5] - goal_state[5])**2)
    # Orientation cost
    error_angle = sqrt((next_state[6] - goal_state[6])**2 + (next_state[7] - goal_state[7])**2)
    # Action cost
    error_action = 0.1*(action[1]**2+action[2]**2)*1/(max(error_position, 0.001))
    # Bounds cost
    bounds_reward = -1 * int((abs(next_state[0] - observation_space.high[0]) < 0.05 or
                              abs(next_state[0] - observation_space.low[0]) < 0.05 or
                              abs(next_state[1] - observation_space.high[1]) < 0.05 or
                              abs(next_state[1] - observation_space.low[1]) < 0.05 or
                              abs(next_state[2] - observation_space.high[2]) < 0.05 or
                              abs(next_state[2] - observation_space.low[2]) < 0.05))
    # Total cost
    total_reward = -1 * error_position + -0.2 * error_velocity - 0.1 * error_angle - error_action + bounds_reward

    done = False
    reward = total_reward

    return reward, done
