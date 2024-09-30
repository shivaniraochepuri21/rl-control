from gymnasium import Env, spaces
import numpy as np

class LinearSecondOrderEnv(Env):
    def __init__(self):
        self.A = np.array([[0.0, 1.0], [-1.0, -1.0]])
        self.B = np.array([[0.0], [1.0]])
        self.C = np.array([[1.0, 0.0]])
        self.D = np.array([[0.0]])

        self.dt = 0.05
        self.K = 8
        self.action_add = [0.0]
        self.action_scale = [1.0]
        
        self.min_K = 0.01        
        self.max_K = 100.0
        self.ref_model_min = -100.0
        self.ref_model_max = 100.0
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_state = -10.0
        self.max_state = 10.0
        
        self.min_obs = self.min_state
        self.max_obs = self.max_state

        self.targ_state = np.array([0.0, 0.0])
        self.init_state = np.array([1.0, 0.0])
        self.test_targ_state = np.array([0.0, 0.0])
        self.test_init_state = np.array([1.0, 0.0])

        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        
    def step(self, u):
        u = u[0]
        done = False

        x = self.state
        new_x = x + self.dt * (self.A @ x + self.B * u).flatten()
        self.state = new_x
        reward = -((self.C @ new_x) ** 2 + (self.D * u ** 2)).flatten()[0]
        
        return np.array(self.state, dtype=np.float32), reward, done, {}, {}
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=self.min_state, high=self.max_state, size=(2,))
        return np.array(self.state, dtype=np.float32), {}
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        pass

