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
            layer_init(nn.Linear(3, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(3, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 1))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
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
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.model=PPO_Agent()
        output_index = 1
        self.model.load_state_dict(torch.load(f'./checkpoint/PPO_v{output_index}.pth'))

    def act(self, observation):
        torch_observation = torch.from_numpy(observation).float()
        action,_,_,_=self.model.get_action_and_value(torch_observation)
        action = action.detach().numpy()
        action=np.clip(action, -2.0, 2.0).squeeze(-1)
        return action