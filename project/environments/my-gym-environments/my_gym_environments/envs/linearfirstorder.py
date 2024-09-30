__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional
import numpy as np
import scipy as sp
import gymnasium as gym
from gymnasium import spaces
# import gym
# from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

class LinearFirstOrderEnv(gym.Env):

    def __init__(self):
        
        # linear first order
        self.A = -2.0
        self.B = 3.0
        self.C = 1.0
        self.D = 0.0

        self.K = 4
        self.lam = 1.005
        self.lam2 = 1.005
        self.w1 = 0.505
        self.w2 = 0.001

        self.dt = 0.05
        self.action_add = [0.0]
        self.action_scale = [1.0]
        self.min_action = -1.0
        self.max_action = 1.0 #step input
        self.min_state = -10.0
        self.max_state = 10.0
        self.min_K  = 0.0        
        self.max_K  = 1000.0
        self.min_e = self.min_state - self.max_state
        self.max_e = self.max_state - self.min_state
        self.min_edot = (self.min_e - self.max_e)/self.dt 
        self.max_edot = (self.max_e - self.min_e)/self.dt 

        self.ref_model_min = self.min_edot + self.K*self.min_e # K >= 0
        self.ref_model_max = self.max_edot + self.K*self.max_e
        
        #obs =  [state, targ_state, model, targ_model, K]
        self.min_obs = np.array([self.min_state, self.min_state, self.ref_model_min, self.ref_model_min, self.min_K])
        self.max_obs = np.array([self.max_state, self.max_state, self.ref_model_max, self.ref_model_max, self.max_K])

        self.targ_state = 0.0 #desired step output
        self.init_state = 1.0
        self.test_targ_state = 0.0
        self.test_init_state = 1.0
        self.targ_m = 0.0

        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        
    def step(self, u):
        # e = y_r - y (y=x, y_r=1), step output
        # e_dot = y_r_dot - y_dot
        u = u[0]
        done = False

        x = self.state
        dx = np.dot(self.A,x) + np.dot(self.B,u)
        new_x = x + self.dt*(dx) 
        self.e = self.targ_state - x
        e_new = self.targ_state - new_x 
        self.e_dot = (e_new - self.e)/self.dt
        self.m = self.e_dot + self.K * self.e
        self.ie += self.e*self.dt

        # e_dot = 0 - (self.A*x + self.B*u) 
        # e = np.linalg.norm(e)
        # e_dot = np.linalg.norm(e_dot)
        # e = float(e)
        # e_dot = float(e_dot)
        # rewards = - (self.K**2 * abs(self.e)) - (self.K * abs(self.e_dot)) - 0.001*u**2 - 5*(abs(self.e * self.e_dot)) 
        # rewards = - (49 * abs(self.e)**2) - (7 * abs(self.e_dot)**2) - 0.5*(abs(self.e_dot + self.K * self.e)**2) - 0.001*u**2
        # IE_term  = self.IE + (abs(self.e))
                # rewards = abs(e_dot + 5*e)**2 + self.K*e_dot**2
        # rewards = -np.exp(-5*(self.dt*(self.i+1)) + 1000)
        # rewards = - (self.alpha **2) * (self.e_dot + self.K * self.e) **2  # -  self.alpha * (self.e)**2
        # rewards = self.alpha*(-(self.K**2 * abs(self.e)**2) - (self.alpha) * abs(self.e_dot)**2 - (2 * self.K/4)*(abs(self.e * self.e_dot)) - self.K * (self.ref_model)) - 0.001*u**2

        # rewards = - self.alpha * (self.K**2 * abs(self.e)**2) - (self.beta) * abs(self.e_dot)**2 - (2 * self.gamma * self.K * abs(self.e * self.e_dot)) - self.omega * (self.ref_model) - self.tau * u**2
        # rewards =  - self.K**2*abs(self.e)**2 - (0.5)*abs(self.e_dot)**2 - 2*self.K*abs(self.e * self.e_dot) - 0.001*u**2
        # rewards = - 20*self.K**2*abs(self.e)**2 - 5*(self.K/20)*abs(self.e_dot)**2 - 4*2*(self.K/10)*abs(self.e * self.e_dot) - 0.001*u**2
        
        # rewards = - (self.K**2 * self.e**2) - self.e_dot**2 - 0.001*u**2 - 2*abs(self.K * self.e * self.e_dot)
  
        # rewards = -25*abs(0.9*self.e_dot + 9*self.K * self.e)**2
        
        # if abs(self.e) <= 0.01:
            # done = True
        
        # if self.state <= (98/100)*self.targ_state:
            # rewards = -abs(self.m - self.targ_m)
        # else:
            # rewards = -abs(self.targ_state - self.state)
            # rewards = -(abs(self.m - self.targ_m) + 0.01*abs(self.targ_state - self.state))

        rewards = -(self.w1*abs(self.m - self.targ_m) + self.lam*abs(e_new)**2 + self.lam2*abs(self.e_dot)**2 + self.w2*u**2)
            
        self.state = new_x
        obs = self.get_obs()
        info = {'e': self.e, 'e_dot': self.e_dot, 'ref_model': self.m}
        return obs, rewards, done, False, info
    
    def reset(self, init_state = None, targ_state = None, K = None):
        if init_state is None:
            self.state = self.np_random.uniform(self.min_state, self.max_state)
        else:
            self.state = init_state

        if targ_state is None:
            # self.targ_state = self.np_random.uniform(self.min_state, self.max_state)
            self.targ_state = self.targ_state
        else:
            self.targ_state = targ_state

        # if K is not None:
            # self.K = K
            # self.K = self.np_random.uniform(self.min_K, self.max_K)
        # else:
            # self.K = self.K

        # self.state = float(self.state)
        # self.targ_state = float(self.targ_state)
        self.e = self.targ_state - self.state
        # self.e_dot = 0 - (self.A*self.state + self.B*0.0)
        self.e_dot = (self.e - self.e)/self.dt
        self.m = self.e_dot + self.K * self.e
        self.ie = self.e
        obs = self.get_obs()
        info = {'init_state':self.state, 'targ_state':self.targ_state, 'e': self.e, 'e_dot': self.e_dot, 'K': self.K, 'model': self.m}
        return obs, info

    def get_obs(self):
        # self.obs = np.array([self.state, ref_model, self.K])
        # self.obs = np.array([self.state, self.targ_state, self.e, self.e_dot, self.K])
        # self.obs = np.array([self.state])
        self.obs = np.array([self.state, self.targ_state, self.m, self.targ_m, self.K])
        return self.obs
    