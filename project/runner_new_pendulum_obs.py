# from Agent import Agent
# import gym
from agents.TD3_Agent import Agent

import gymnasium as gym
import my_gym_environments

import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from noise import *

class SimpleRunner:
    def __init__(self, num_episodes=100, learn_every=1, noise_variance=0.1, viz=False, eval_every=20, wandb_on=True, with_dth=False) -> None:
        # self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='rgb_array')
        self.env = gym.make('MyPendulum-v0')
        self.num_episodes = num_episodes
        self.max_action = float(self.env.max_action)
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0], max_action = self.max_action, action_scale = self.env.action_scale, action_add = self.env.action_add)
        self.learn_every = learn_every
        self.noise_variance = noise_variance
        self.viz = viz
        self.eval_every = eval_every
        self.wandb_on = wandb_on
        self.avg_window = self.num_episodes//20
        self.init_state = self.env.init_state
        self.targ_state = self.env.targ_state
        self.test_init_state = self.env.test_init_state
        self.test_targ_state = self.env.test_targ_state
        self.K = self.env.K
        self.new_reward_flag = 1
        self.with_dth = with_dth
        self.lam = self.env.lam
        self.lam2 = self.env.lam2
        self.w1 = self.env.w1
        self.w2 = self.env.w2

        if self.wandb_on:
            wandb.config = {
                "num_episodes": num_episodes,
                "learn_every": learn_every,
                "noise_variance": noise_variance,
                "eval_every": eval_every,
                "K":self.K,
                "lambda":self.lam,
                "lambda2":self.lam2,
                "w1":self.w1,
                "w2":self.w2,
                "max_action":self.max_action
            }
            wandb.init(project="TD3-errorcontrol", entity="shivanichepuri", config = wandb.config)   
    
    def run(self):
        steps=0
        # actor_noise = OUNoise(mu=np.zeros(self.env.action_space.shape[0]))
        score_history = [] # score is the sum of rewards of an episode
        best_score = self.env.reward_range[0] 
        for episode in range(self.num_episodes):
            observation, info = self.env.reset(targ_state=self.targ_state)
            # print('info*******************************', info)

            done = False
            done_ = False
            score = 0
            episode_len = 0
            episode_action = 0
            episode_error = 0.0
            err_actor_episode = 0
            err_critic_episode = 0
            # score_rewards = []

            while not (done or done_):
            # for i in range(1):
                actor_noise = np.random.normal(0, self.noise_variance, self.env.action_space.shape)    
                action = self.agent.act(observation) + actor_noise
                observation_, reward, done, done_, vars = self.env.step(action)
                sys_error = vars['error']
                sys_error_dot = vars['error_dot']
                sys_theta = vars['theta']
                sys_theta_dot = vars['theta_dot']
                # print('action, reward, next_sys_theta, next_sys_theta_dot,\n', action, reward, sys_theta, sys_theta_dot)
                # print('sys_error, sys_error_dot', sys_error, sys_error_dot)

                # done = done or done_
                self.agent.memorize(observation, action, reward, observation_, not (done or done_))

                observation = observation_
                if steps % self.learn_every == 0:
                     err_actor, err_critic = self.agent.learn()
                steps = (steps + 1) % self.learn_every
                if self.viz:
                    self.env.render()
                score += reward
                episode_len += 1
                episode_action += action
                episode_error += abs(sys_error)
                # score_rewards.append(reward)
                err_actor_episode += err_actor
                err_critic_episode += err_critic

            score_history.append(score)
            avg_score = np.mean(score_history[-self.avg_window:])
            if avg_score > best_score:
                best_score = avg_score
                # best_score_rewards = score_rewards
                self.best_model_episode = episode
                
                if self.new_reward_flag:
                    self.agent.TD.save(f"best_model_"+ str(self.K) + "_targ" + str(int(np.rad2deg(self.targ_state[0]))))
                else:
                    self.agent.TD.save(f"best_model_"+ "targ" + str(int(np.rad2deg(self.targ_state[0]))))

            train_avg_reward_of_episode = score/episode_len
            train_avg_action_of_episode = episode_action/episode_len
            train_avg_error_of_episode = episode_error/episode_len
            # print(f"train_episode: {episode}, train_avg_reward_of_episode:{train_avg_reward_of_episode}")
            print(f"{episode}, {episode_len}, avg_reward:{train_avg_reward_of_episode}, last_st:{sys_theta}, last_e:{sys_error}")

            if self.wandb_on:
                wandb.log({"train_episode":episode, "train_score":score, "train_episode_len":episode_len, "train_episode_actor_loss": err_actor_episode, "train_episode_critic_loss":err_critic_episode, "train_avg_reward_of_episode": train_avg_reward_of_episode, "train_avg_action_of_episode": train_avg_action_of_episode, "train_avg_error_of_episode": train_avg_error_of_episode})

            if episode % self.learn_every == 0:
                # self.agent.TD.save("checkpoints_obs_dth" + str(int(self.with_dth)) + "/" + str(episode))
                self.agent.TD.save("checkpoints/checkpoints_my_pendulum" + "/" + str(episode))

            if episode % self.eval_every == 0:
                self.eval(episode)

        if self.wandb_on:
            wandb.log({"train_best_score": best_score, "train_best_model_episode": self.best_model_episode})

    def eval(self, episode):
        observation, _ = self.env.reset(init_state=self.test_init_state, targ_state=self.test_targ_state)
        done = False
        done_ = False
        score = 0
        episode_error = 0.0
        step = 0

        # while not (done or done_):
        for i in range(2000):
            action = self.agent.act(observation)
            observation_, reward, done, done_, vars = self.env.step(action)
            sys_error = vars['error']
            sys_error_dot = vars['error_dot']
            sys_theta = vars['theta']
            sys_theta_dot = vars['theta_dot']

            # done = done or done_
            observation = observation_
            # if self.viz:
                # self.env.render()
            # self.env.render()
            score += reward
            episode_error += abs(sys_error)
            step+=1

            if self.wandb_on:
                wandb.log({"eval_theta_ep_" + str(episode): sys_theta, "eval_theta_dot_ep_" + str(episode): sys_theta_dot, "eval_action_ep_" + str(episode): action, "eval_step" :step, "eval_reward_ep_" + str(episode): reward, "eval_error_ep_" + str(episode): abs(sys_error), "eval_error_dot_ep_" + str(episode): sys_error_dot})

        print(f"eval_episode: {episode}, eval_score: {score}")
        print(f"eval_avg_reward: {score/step}, eval_episode_len: {step}, eval_final_sys_state: {sys_theta}, eval_final_sys_error: {sys_error}")

        if self.wandb_on:
            wandb.log({"eval_episode": episode, "eval_episode_len": step, "eval_score_ep_" + str(episode): score, "eval_episode_error_ep_" + str(episode): episode_error}) 

    def test_after_training(self): # we test for one episode
        # self.env = gym.make('gym_mypendulum:mypendulum-v0', render_mode='human')
        self.env = gym.make('MyPendulum-v0')

        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0], max_action = self.env.max_action, action_scale = self.env.action_scale, action_add = self.env.action_add)
        if self.new_reward_flag:
            self.agent.TD.load(f"best_model_" + str(self.K) + "_targ" + str(int(np.rad2deg(self.targ_state[0]))))
        else:
            self.agent.TD.load(f"best_model_"+ "targ" + str(int(np.rad2deg(self.targ_state[0]))))
        observation, _ = self.env.reset(targ_state=self.test_targ_state, init_state=self.test_init_state)
        done = False
        done_ = False
        score = 0
        step = 0
        episode_error = 0.0

        # while not (done or done_):
        for i in range(2000):
            action = self.agent.act(observation)
            observation_, reward, done, done_, vars = self.env.step(action)
            sys_error = vars['error']
            sys_error_dot = vars['error_dot']
            sys_theta = vars['theta']
            sys_theta_dot = vars['theta_dot']
            
            # done = done or done_
            observation = observation_
            if self.viz:
                self.env.render()
            self.env.render()
            score += reward
            step += 1
            episode_error += abs(sys_error)  
            if self.wandb_on:
                wandb.log({"test_reward": reward, "test_theta": sys_theta, "test_dtheta": sys_theta_dot, "test_action":action, "test_step":step, "test_error": abs(sys_error), "test_error_dot": sys_error_dot })
        
        # print(f"test_score: {score}, test_episode_len: {step}")
        print(f"test_avg_reward: {score/step}, test_episode_len: {step}, test_final_sys_state: {sys_theta}, test_final_sys_error: {sys_error}")
        
        if self.wandb_on:
            wandb.log({"test_score": score, "test_episode_len": step, "test_error_total": episode_error})

if __name__ == "__main__":
    runner = SimpleRunner(viz = False)
    runner.run()
    runner.test_after_training()