Has multiple custom and prebuilt gym environments used for this project

# Custom Gymnasium Environment

This is a custom environment for Gymnasium. 

## Installation

```bash
pip install -e .

## Example Usage

# Run as a new python script

import gymnasium as gym
import my_gym_environments

env = gym.make('LinearFirstOrderEnv-v0')

obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()

