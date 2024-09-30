import gymnasium as gym
import my_gym_environments

env = gym.make('LinearFirstOrder-v0')

obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    print("env.step(action)", env.step(action))
    obs, reward, done, info, done_ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()

