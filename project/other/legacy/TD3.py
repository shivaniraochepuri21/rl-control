import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, action_scale, action_add):
        super(Actor, self).__init__()
        self.action_scale = torch.FloatTensor(action_scale).to(device)
        self.action_add = torch.FloatTensor(action_add).to(device)
        
        self.l1 = nn.Linear(state_dim, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, action_dim)        
        self.max_action = max_action        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = self.l4(a)
        a = self.action_scale * (torch.tanh(a) + self.action_add)
        return self.max_action * a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 1)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 16)
        self.l6 = nn.Linear(16, 32)
        self.l7 = nn.Linear(32, 16)
        self.l8 = nn.Linear(16, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, action_scale, action_add):
        self.actor = Actor(state_dim, action_dim, max_action, action_scale, action_add).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, memory, batch_size):
        self.total_it += 1

        # Sample replay buffer
        batch = random.sample(memory, batch_size)
        state, action, reward, next_state, not_done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).to(device).reshape(-1, 1)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        not_done = torch.FloatTensor(np.array(not_done)).to(device).reshape(-1, 1)

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

