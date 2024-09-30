from gymnasium import Env, spaces
import numpy as np

class MyPendulumEnv(Env):
    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.viewer = None
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (
            -3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u
        ) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        self.state = np.array([newth, newthdot])

        return self._get_obs(), -costs, False, {}, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs(), {}
    
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode="human"):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

