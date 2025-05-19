import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import imageio
import numpy as np
from dmc import make_dmc_env
import os
os.environ["MUJOCO_GL"] = "egl"

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

output_index = 2    
video_path = f"random_agent_{output_index}.mp4"
writer = imageio.get_writer(video_path, fps=30)


global_mean = np.zeros((67,), dtype=np.float64)
global_var = np.ones((67,), dtype=np.float64)
global_count = 1e-4

def act(in_agent,observation):
    global global_mean, global_var, global_count
    batch_mean = np.mean(observation, axis=0)
    batch_var = np.var(observation, axis=0)
    batch_count = observation.shape[0]
    global_mean, global_var, global_count = update_mean_var_count_from_moments(
        global_mean, global_var, global_count, batch_mean, batch_var, batch_count
    )
    observation = np.float32((observation - global_mean) / np.sqrt(global_var + 1e-8))
    observation = np.clip(observation, -10.0, 10.0)
    torch_observation = torch.from_numpy(observation).float()
    action,_,_,_=in_agent.get_action_and_value(torch_observation)
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

def make_env(gym_id, seed=1, idx=0, capture_video=False, run_name="nye"):
        env = make_dmc_env(gym_id, seed , flatten=True, use_pixels=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),None)
        env = gym.wrappers.NormalizeReward(env,0.95)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

if __name__=="__main__":        # Test the PPO_Agent class
    agent = PPO_Agent()
    agent.load_state_dict(torch.load(f'./checkpoint/PPO_v{output_index}.pth', map_location=torch.device('cpu')))
    env = make_env('humanoid-walk')  # 設置渲染模式為 rgb_array
    obs,_=env.reset()
    done = False
    total_reward = 0
    while not done:
        obs=np.expand_dims(obs, axis=0)
        action = act(agent,obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        obs=next_obs
        total_reward += reward

        frame = env.render()
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)
        writer.append_data(frame)

        if done:
            print(f"Total reward: {total_reward}")
            print(f"Episode reward: {info['episode']['r']}")
            break
    env.close()