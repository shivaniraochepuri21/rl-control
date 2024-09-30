__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional
import numpy as np
import scipy as sp
# import gym
import gymnasium as gym
#from gym import spaces
from gymnasium import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

class LinearSecondOrderEnv(gym.Env):

    def __init__(self):
        
        #linear second order
        self.A = np.array([[0.0, 1.0],[-1.0, -1.0]])
        self.B = np.transpose(np.array([0.0, 1.0]))
        self.C = np.array([1.0, 0.0])
        self.D = np.array([0.0])

        self.K = 0.01
        self.lam = 3.0
        self.lam2 = 0.1
        self.w1 = 1.0
        self.w2 = 0.0001

        self.dt = 0.05
        self.action_add = [0.0]
        self.action_scale = [1.0]

        # self.max_obs = np.array([10.0, 10.0])
        # self.min_obs = np.array([-10.0, -10.0])
        # self.max_action = np.array([1.0]) #step input
        # self.min_action = np.array([-1.0])

        self.min_action = -1.0
        self.max_action = 1.0 #step input
        self.min_state0 = -10.0
        self.max_state0 = 10.0
        self.min_state1 = (self.min_state0 - self.max_state0)/self.dt
        self.max_state1 = (self.max_state0 - self.min_state0)/self.dt
        self.min_state = np.array([self.min_state0, self.min_state0])
        self.max_state = np.array([self.max_state1, self.max_state1])
        self.min_K  = 0.0        
        self.max_K  = 1000.0
        self.min_e = self.min_state - self.max_state
        self.max_e = self.max_state - self.min_state
        self.min_edot = (self.min_e - self.max_e)/self.dt 
        self.max_edot = (self.max_e - self.min_e)/self.dt 

        self.ref_model_min = np.linalg.norm(self.min_edot + self.K*self.min_e) # K >= 0
        self.ref_model_max = np.linalg.norm(self.max_edot + self.K*self.max_e)
        
        #obs =  [state, targ_state, model, targ_model, K]
        self.min_obs = np.array([self.min_state0, self.min_state1, self.min_state0, self.min_state1, self.ref_model_min, self.ref_model_min, self.min_K])
        self.max_obs = np.array([self.max_state0, self.max_state1, self.max_state0, self.max_state1, self.ref_model_max, self.ref_model_max, self.max_K])

        self.init_state = np.array([1.0, 0.0])
        self.targ_state = np.array([0.0, 0.0]) #desired is step output
        self.test_init_state = np.array([1.0, 0.0])
        self.test_targ_state = np.array([0.0, 0.0])
        self.targ_m = 0.0

        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
    
    def step(self, u):
        # e = y_r - y (y=x, y_r=1), step output
        # e_dot = y_r_dot - y_dot
        u = u[0]
        done = False
        x = self.state
        # e0 = self.targ_state[0] - x
        # e1 = self.targ_state[1] - xdot
        # e0_dot = 0 - (xdot)
        # e1_dot = 0 - (-3*xdot -2*x + 2*u)
                
        # e = np.array([e0,e1])
        # e_dot = np.array([e0_dot,e1_dot])
        # e = np.linalg.norm(e)
        # e_dot = np.linalg.norm(e_dot)
        # new_xdot = xdot + self.dt*(-3*xdot -2*x + 2*u)
        # new_x = x + new_xdot * self.dt
        dx = np.dot(self.A,x) + np.dot(self.B,u)
        new_x = x + self.dt*(dx) 
        self.e = self.targ_state[0] - x[0]
        e_new = self.targ_state[0] - new_x[0]
        self.e_dot = (e_new - self.e)/self.dt
        self.m = np.linalg.norm(self.e_dot + self.K * self.e)

        # rewards = - (self.K**2 * e**2) - e_dot**2 - 0.001*u**2 - 2*abs(self.K * e * e_dot)       

        # if self.state[0] <= (98/100)*self.targ_state[0]:
            # rewards = -abs(self.m - self.targ_m)
        # else:
            # rewards = -abs(self.targ_state[0] - self.state[0])
        
        rewards = -(self.w1*abs(self.m - self.targ_m) + self.lam*abs(e_new)**2 + self.lam2*abs(self.e_dot)**2 + self.w2*u**2)

        # if abs(e) <= 0.001:
            # done = True
        
        info = {'e': self.e, 'e_dot': self.e_dot, 'ref_model': self.m}
        
        # self.state = np.array([new_x, new_xdot])
        self.state = new_x
        obs = self.get_obs()

        # self.state = np.reshape(self.state, (2,))

        return obs, rewards, done, False, info
    
    def reset(self, init_state = None, targ_state = None):

        if init_state is None:
            self.state = self.np_random.uniform(self.min_state, self.max_state)
        else:
            self.state = init_state

        if targ_state is None:
            # self.targ_state = self.np_random.uniform(self.min_state, self.max_state)
            self.targ_state = self.targ_state
        else:
            self.targ_state = targ_state
        
        self.e = self.targ_state - self.state
        self.e_dot = (self.e - self.e)/self.dt
        self.m = np.linalg.norm(self.e_dot + self.K * self.e)
        obs = self.get_obs()

        info = {'init_state':self.state, 'targ_state':self.targ_state, 'e': self.e[0], 'e_dot': self.e_dot[0], 'K': self.K, 'model': self.m}
        return obs, info

    def get_obs(self):
        # self.obs = np.array([self.state, ref_model, self.K])
        # self.obs = np.array([self.state, self.targ_state, self.e, self.e_dot, self.K])
        # self.obs = np.array([self.state])
        self.obs = np.array([self.state[0], self.targ_state[0], self.state[1], self.targ_state[1], self.m, self.targ_m, self.K])
        return self.obs
