import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from agents.TD3_agent import Agent
from noise import *

class SimpleRunner:
    def __init__(self, num_episodes=1, eval_every=20, learn_every=1, noise_variance=0.1, viz=False, wandb_on=True, with_dth=False):
        self.env = gym.make('gym_mypendulum:linearsys-v0')
        self.num_episodes = num_episodes
        self.max_action = float(self.env.max_action)
        self.K = self.env.K
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0], max_action=self.max_action, action_scale=self.env.action_scale, action_add=self.env.action_add)

        self.learn_every = learn_every
        self.noise_variance = noise_variance
        self.viz = viz
        self.eval_every = eval_every
        self.wandb_on = wandb_on
        self.avg_window = self.num_episodes // 20

        self.init_state = self.env.init_state
        self.targ_state = self.env.targ_state
        self.test_init_state = self.env.test_init_state
        self.test_targ_state = self.env.test_targ_state
        self.new_reward_flag = 1
        self.with_dth = with_dth

        if self.wandb_on:
            wandb.config = {
                "num_episodes": num_episodes,
                "learn_every": learn_every,
                "noise_variance": noise_variance,
                "eval_every": eval_every,
            }
            wandb.init(project="TD3-errorcontrol-firstorderlinear", entity="shivanichepuri", config=wandb.config)   
    
    def run(self):
        steps = 0
        score_history = []  # score is the sum of rewards of an episode
        best_score = self.env.reward_range[0]
        for episode in range(self.num_episodes):
            observation, info = self.env.reset(targ_state=self.targ_state)
            print('info_after_ep_reset', info)

            done = False
            while not done:
                action = self.agent.act(observation)
                next_observation, reward, done, info = self.env.step(action)
                not_done = 1 - int(done)
                self.agent.memorize(observation, action, reward, next_observation, not_done)
                observation = next_observation
                steps += 1

                if steps % self.learn_every == 0:
                    self.agent.learn()

            score_history.append(reward)
            avg_score = np.mean(score_history[-self.avg_window:])

            if avg_score > best_score:
                best_score = avg_score

            if self.wandb_on:
                wandb.log({
                    "episode": episode,
                    "reward": reward,
                    "avg_reward": avg_score,
                    "best_reward": best_score,
                })

        if self.viz:
            plt.plot(score_history)
            plt.ylabel('Score')
            plt.xlabel('Episode')
            plt.show()

