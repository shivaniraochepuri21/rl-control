# from Agent import Agent
from agents.TD3_Agent import Agent

# import gym
import gymnasium as gym
import my_gym_environments

import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from noise import *
import matplotlib.pyplot as plt

class SimpleRunner:
    def __init__(self, num_episodes=10, eval_every=40, learn_every=1, noise_variance=0.1, viz=False, wandb_on=True, with_dth=False) -> None:
        # self.env = gym.make('gym_mypendulum:linearsys-v1')
        self.env = gym.make('LinearSecondOrder-v0')

        self.num_episodes = num_episodes
        self.max_action = float(self.env.max_action)
        self.K = self.env.K
        self.lam = self.env.lam
        self.lam2 = self.env.lam2
        self.w1 = self.env.w1
        self.w2 = self.env.w2 
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0], max_action=self.max_action, action_scale = self.env.action_scale, action_add = self.env.action_add)

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
        self.new_reward_flag = 1
        self.with_dth = with_dth

        if self.wandb_on:
            wandb.config = {
                "num_episodes": num_episodes,
                "learn_every": learn_every,
                "noise_variance": noise_variance,
                "eval_every": eval_every,
                "K": self.K,
                "Lambda": self.lam,
                "Lambda2": self.lam2,
                "w1": self.w1,
                "w2": self.w2,
                "max_action": self.max_action,
                "sys": "wn = 1, zeta = 0.5"
            }
            wandb.init(project="TD3-errorcontrol-secondorderlinear", entity="shivanichepuri", config = wandb.config)   
    
    def run(self):
        steps=0
        # actor_noise = OUNoise(mu=np.zeros(self.env.action_space.shape[0]))
        score_history = [] # score is the sum of rewards of an episode
        best_score = self.env.reward_range[0] 
        for episode in range(self.num_episodes):
            observation, info = self.env.reset(init_state=self.init_state, targ_state=self.targ_state)
            # print('info_after_ep_reset', info)

            done = False
            done_ = False
            score = 0
            episode_len = 0
            episode_action = 0
            episode_error = 0.0
            err_actor_episode = 0
            err_critic_episode = 0
            # score_rewards = []
            # while not (done or done_):
            for i in range(int(1000)):
                actor_noise = np.random.normal(0, self.noise_variance, self.env.action_space.shape)    
                action = self.agent.act(observation) + actor_noise
                # print('action', action)
                observation_, reward, done, done_, vars = self.env.step(action)
                # print(observation, observation_)

                sys_error = vars['e'] #old state and e calculated wrt the old state
                sys_error_dot = vars['e_dot'] #old state and e calculated wrt the old state
                ref_model = vars['ref_model']

                self.agent.memorize(observation, action, reward, observation_, not (done or done_))
                observation = observation_

                if steps % self.learn_every == 0:
                     err_actor, err_critic = self.agent.learn()
                steps = (steps + 1) % self.learn_every
                score += reward
                episode_len += 1
                episode_action += action
                episode_error += abs(sys_error)
                # score_rewards.append(reward)
                err_actor_episode += err_actor
                err_critic_episode += err_critic
                # print('observation, step', observation, episode_len)
                # print('sys_error, sys_error_dot, action, reward', sys_error, sys_error_dot, action, reward)
                # print('\n')
                if self.wandb_on:
                    # wandb.log({'sys_state': sys_state, 'action': action, "step": i, 'reward': reward, 'sys_error': sys_error})
                    wandb.log({'sys_state': observation[0], 'action': action, "step": episode_len, 'reward': reward, 'sys_error': sys_error, 'sys_error_dot': sys_error_dot, 'model': ref_model})
                # if (done or done_):
                    # print("episode done------------------------------")
                    # break

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

            print(f"{episode}, {episode_len}, avg_reward:{train_avg_reward_of_episode}, final_obs_ep:{observation}, final_e_ep:{sys_error}")
            # print("episode done------------------------------")
            
            if self.wandb_on:
                wandb.log({"train_episode":episode, "train_score":score, "train_episode_len":episode_len, "train_episode_actor_loss": err_actor_episode, "train_episode_critic_loss":err_critic_episode, "train_avg_reward_of_episode": train_avg_reward_of_episode, "train_avg_action_of_episode": train_avg_action_of_episode, "train_avg_error_of_episode": train_avg_error_of_episode})

            if episode % self.learn_every == 0:
                # self.agent.TD.save("checkpoints_obs_dth" + str(int(self.with_dth)) + "/" + str(episode))
                self.agent.TD.save("checkpoints/checkpoints_linearsecondorder" + "/" + str(episode))

            if (episode % self.eval_every == 0) and (episode != 0):
                self.eval(episode)

            # if abs(sys_state - self.targ_state) <= 0.001:
                # break

        if self.wandb_on:
            wandb.log({"train_best_score": best_score, "train_best_model_episode": self.best_model_episode})

    def eval(self, episode):
        observation, _ = self.env.reset(init_state=self.test_init_state, targ_state=self.test_targ_state)
        done = False
        done_ = False
        score = 0
        episode_error = 0.0
        step = 0

        while not (done or done_):
            action = self.agent.act(observation)
            observation_, reward, done, done_, vars = self.env.step(action)
            sys_error = vars['e']
            sys_error_dot = vars['e_dot']
            ref_model = vars['ref_model']

            observation = observation_
            score += reward
            episode_error += sys_error
            step+=1

            if self.wandb_on:
                wandb.log({"eval_theta_ep_" + str(episode): observation[0], "eval_action_ep_" + str(episode): action, "eval_step" :step, "eval_reward_ep_" + str(episode): reward, "eval_error_ep_" + str(episode): sys_error, "eval_edot_ep_" + str(episode): sys_error_dot, "eval_errormodel_ep_" + str(episode): ref_model})
        print('--------------------------')
        print(f"eval_episode: {episode}, eval_score: {score}")
        print(f"eval_avg_reward: {score/step}, eval_episode_len: {step}, eval_final_obs: {observation}, eval_final_sys_error: {sys_error}, eval_e_dot: {sys_error_dot}")
        print('---------------------------')
        
        # if self.wandb_on:
            # wandb.log({"eval_episode": episode, "eval_episode_len": step, "eval_score_ep_" + str(episode): score, "eval_episode_error_ep_" + str(episode): episode_error, "eval_avg_reward": score/step}) 

    def test_after_training(self): # we test for one episode
        # self.env = gym.make('gym_mypendulum:linearsys-v1')
        
        self.agent = Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0], max_action=self.max_action)
        if self.new_reward_flag:
            self.agent.TD.load(f"best_model_" + str(self.K) + "_targ" + str(int(np.rad2deg(self.targ_state[0]))))
            # self.agent.TD.load(f"best_model_" + str(9) + "_targ" + str(int(np.rad2deg(0.0))))
        # else:
            # self.agent.TD.load(f"best_model_"+ "targ" + str(int(np.rad2deg(self.targ_state[0]))))
        # self.agent.TD.load("best_model_5.0_targ57")
        observation, _ = self.env.reset(targ_state=self.test_targ_state, init_state=self.test_init_state)
        done = False
        done_ = False
        score = 0
        step = 0
        episode_error = 0.0
        step_arr = []
        sys_error_arr = []
        # while not (done or done_):
        for i in range(2000):
            action = self.agent.act(observation)
            observation_, reward, done, done_, vars = self.env.step(action)
            sys_error = vars['e']
            sys_error_dot = vars['e_dot']
            ref_model = vars['ref_model']
            # print('sys_state, sys_error', sys_state, sys_error)
            observation = observation_
            score += reward
            step += 1
            episode_error += sys_error
            step_arr.append(step)
            sys_error_arr.append(sys_error)
            if self.wandb_on:
                wandb.log({"test_reward": reward, "test_sys_state": observation[0], "test_action":action, "test_step":step, "test_sys_error": sys_error, "test_sys_error_dot": sys_error_dot, "test_sys_refmodel": ref_model })
        print('\n')
        print(f"test_avg_reward: {score/step}, test_episode_len: {step}, test_final_obs: {observation}, test_final_sys_error: {sys_error}")
        print('\n')
        # plt.plot(step_arr, sys_error_arr)
        # plt.show()
        if self.wandb_on:
            wandb.log({"test_score": score, "test_episode_len": step, "test_error_total": episode_error, "test_avg_reward": score/step})

if __name__ == "__main__":
    runner = SimpleRunner(viz = False)
    runner.run()
    runner.test_after_training()