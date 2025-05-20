import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.nn import functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO_Agent(nn.Module):
    def __init__(self):
        super(PPO_Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(67, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(67, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 21), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 21))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x).unsqueeze(0)
        action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.model=PPO_Agent()
        output_index = 1
        self.model.load_state_dict(torch.load(f'./checkpoint/PPO_v{output_index}.pth', map_location=torch.device('cpu')))
        self.mean = np.zeros((67,), dtype=np.float64)
        self.var = np.ones((67,), dtype=np.float64)
        self.count = 1e-4

    def act(self, observation):
        observation = np.array(observation, dtype=np.float64)
        observation = np.expand_dims(observation, axis=0)
        batch_mean = np.mean(observation, axis=0)
        batch_var = np.var(observation, axis=0)
        batch_count = observation.shape[0]
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
        observation = np.float32((observation - self.mean) / np.sqrt(self.var + 1e-8))
        observation = np.clip(observation, -10.0, 10.0)
        torch_observation = torch.from_numpy(observation).float()
        action,_,_,_=self.model.get_action_and_value(torch_observation)
        action = action.detach().numpy()
        action=np.clip(action, -1.0, 1.0).squeeze(0)
        return action
    

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count