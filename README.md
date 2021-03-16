Welcome to the repository for the Python code used for the paper:

"Inclined Quadrotor Landing using Deep Reinforcement Learning"

By Jacob Kooi and Robert Babuska.

More documentation will be added soon.


# InclinedDroneLander

Model-free training of 2d inclined landing and 3d setpoint-tracking for the Crazyflie 2.1 Nano-UAV. The resulting policies are transferable to use on a real crazyflie. (See https://youtu.be/53YaqfwUIFU)

## Usage

- In main.py, select either 2d inclined landing or 3d setpoint training.
- When choosing setpoint tracking, one can use SAC,TD3,A2C, PPO etc.
- When choosing inclined landing, use the on-policy PPO algorithm.
- Select the thrust action space, taken as a PWM signal from hover (around 42000 for m = 0.3303). Also select the timesteps to train for and the sampling time (1/50s recommended).
- run main.py



## Dependencies
Trained and tested on:
```
Python 3.7
PyTorch 1.5.0
Numpy 1.18.4
StableBaselines3 0.6.0
gym 0.17.3
```

## Results

PPO Sparse Inclined Landing (3000000 timesteps)           |  PPO Euclidean Setpoint Tracking (1000000 timesteps) (Left is xz-plane, right is yz-plane)
:-------------------------:|:-------------------------:
![](https://github.com/Jacobkooi/InclinedDroneLander/blob/master/Gifs/Gif_landing.gif) |  ![](https://github.com/Jacobkooi/InclinedDroneLander/blob/master/Gifs/Gif_setpoint.gif)

## Questions

For any questions, errors or suggestions contact me at jacobkooi92@gmail.com.

## Citing

When using this code or referring to our work please cite us at:

Arxiv upload coming soon.
