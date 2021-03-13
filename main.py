from Environments.crazyflie import Crazyflie_3d_setpoint, Crazyflie_2d_inclined
import numpy as np
import torch
from Reward.rewardfuncs import sparse_reward2d, euclidean_reward3d
from math import pi
from Save_Gif.save_gif import save_frames_as_gif
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, A2C, TD3
import random
import os

# Use the GPU for training
if not torch.cuda.is_available():
    raise NotImplementedError("No GPU")
else:
    device = torch.device('cuda')

if __name__ == '__main__':

    environment = 'CF_2d_inclined'
    algorithm = 'PPO'               # PPO is fast, robust, and on-policy for curriculum learning
    training_timesteps = 3000000    # Total amount of Timesteps to train for
    pwm = 16000                     # PWM from theoretical hover Pwm, in the minus or plus direction.
    t_s = 1/50                      # seconds

    if environment == 'CF_2d_inclined':
        env = Crazyflie_2d_inclined(t_s, rewardfunc=sparse_reward2d, max_pwm_from_hover=pwm)

    elif environment == 'CF_3d_setpoint':
        env = Crazyflie_3d_setpoint(t_s, rewardfunc=euclidean_reward3d, max_pwm_from_hover=pwm)

    # Check if the environment is working right
    check_env(env)

    # Set seeds to be able to reproduce results
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select the algorithm from Stable Baselines 3
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed) # Try 0.995 for inclined landing
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, seed=seed)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)

    model.learn(training_timesteps)

    # Name the pytorch model
    run_name = environment+"_"+algorithm+"_"+str(training_timesteps)+"Timesteps"

    # Save the pytorch model
    torch.save(model.policy.state_dict(), run_name + 'state_dict.pt')

    # Render and save the gif afterwards
    obs = env.reset()
    frames = []

    # Render
    for i in range(1800):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames.append(env.render(mode='rgb_array'))
        if done:
            obs = env.reset()
    # Save Gif
    save_frames_as_gif(frames, filename=run_name+'.gif')
    env.close()







