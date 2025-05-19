import argparse
import os
import random
from distutils.util import strtobool
import time
#tools
from torch.utils.tensorboard import SummaryWriter
# torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn import functional as F
import numpy as np
#PPO env
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from dmc import make_dmc_env
import os
os.environ["MUJOCO_GL"] = "egl"


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp-name', type= str, default=os.path.basename(__file__).rstrip(".py"),
            help="the name of this experiment")
    parser.add_argument('--gym-id', type= str, default='SuperMarioBrosRandomStages-v0',
            help="the id of the gym environment")    
    parser.add_argument('--learning-rate', type= float, default=1e-3,
            help="lr of the optimizer")  
    parser.add_argument('--seed', type= int, default=1,
            help="seed of rand num")        
    parser.add_argument('--total-timesteps', type= int, default=1000000,
            help="total time steps of the experiment")      
    # GPU args
    parser.add_argument('--torch-deterministic', type= lambda x: bool(strtobool(x)), default=True,
            nargs='?', const=True, help="if toggled, torch.backend.cudnn.deterministic=False")      
    parser.add_argument('--cuda', type= lambda x: bool(strtobool(x)), default=True,
            nargs='?', const=True, help="if toggled, cuda will not be enabled by default") 
    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--from-pretrained", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the model will be loaded from a pretrained model")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

#global variables
args=parse_args()

# env wrapper
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = make_dmc_env(gym_id, seed , flatten=True, use_pixels=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),None)
        env = gym.wrappers.NormalizeReward(env,args.gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

#Agent setting
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


#PPO　Agent
class ppo_buffer():
    def __init__(self, envs, device):
        self.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)


class PPO_agent(ppo_buffer):   
    def __init__(self,envs,device,lr,writer):
        super(PPO_agent, self).__init__(envs, device)
        self.model=Agent(envs).to(device)
        self.writer=writer
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr,eps=1e-5)
        self.lr=lr
        self.lr_frac= 1.0
        # print(f"debug:{type(envs.reset())}")
        self.next_obs = torch.tensor(envs.reset()[0],dtype=torch.float32).to(device)
        self.next_done = torch.zeros(args.num_envs).to(device)
        #background setting
            #static variables
        self.global_step = 0
        self.start_time = time.time()
        self.num_updates = args.total_timesteps // args.batch_size
        
    def GAE(self):
        with torch.no_grad():
            # mean_reward = self.rewards.mean()
            # std_reward = self.rewards.std() + 1e-8  # 避免除以0
            # self.rewards = (self.rewards - mean_reward) / std_reward
            next_value = self.model.get_value(self.next_obs).reshape(1, -1)
            if args.gae: # GAE
                advantages = torch.zeros_like(self.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else: # normal advantage function
                returns = torch.zeros_like(self.rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done.float()
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - self.values
        return returns, advantages
    def train(self):
        # collect data
        for update in range(1,self.num_updates+1):
            if args.anneal_lr:
                self.lr_frac = 1.0 - (update-1.0) / self.num_updates
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * self.lr_frac
            for step in range(args.num_steps):
                self.global_step += 1*args.num_envs 
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done

                with torch.no_grad():
                    action, logprob, _, value = self.model.get_action_and_value(self.next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                #print("action",action.shape)
                next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                done = terminated | truncated
                self.rewards[step] = torch.tensor(reward).to(device).view(-1)
                self.next_obs, self.next_done = torch.tensor(next_obs,dtype=torch.float32).to(device), torch.tensor(done).to(device)
                if "episode" in info.keys():
                    # print(f"global_step={self.global_step}, episodic_return={info['episode']['r'].mean()}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"].mean(), self.global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"].mean(), self.global_step)
                    break
            #print("rewards",self.rewards)
            returns,advantages=self.GAE()
            # flatten the batch
            b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)
            #print(f"b_returns:{b_returns},b_values:{b_values}, b_advantages:{b_advantages}")
            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = self.model.get_action_and_value(b_obs[mb_inds], b_actions.float()[mb_inds])
                    newlogprob = newlogprob.flatten()
                    # print(f"newlogprob:{newlogprob}, b_logprobs[mb_inds]:{b_logprobs[mb_inds]}")
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    # print(f"ratio:{ratio}")
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    # Policy loss
                    # print(f"ratio:{ratio}, b_advantages[mb_inds]:{b_advantages[mb_inds]}")
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:                        
                        v_loss_unclipped = (newvalue-b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped-b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # print(f"epoch:{epoch}, step:{step}, pg_loss:{pg_loss.item()}, v_loss:{v_loss.item()}, entropy_loss:{entropy_loss.item()}, approx_kl:{approx_kl.item()}")
                    loss = pg_loss - args.ent_coef * entropy_loss +  args.vf_coef * v_loss 

                    self.optimizer.zero_grad()
                    loss.backward()
                    #nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    self.optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break 
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            writer.add_scalar("charts/value", b_values.mean().item(), self.global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)          
            

if __name__=="__main__":
    output_index = 1
    print(args)
    run_name=f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    # whether build wandb 
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # Tensorboard construction
    writer = SummaryWriter(f"./runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )        
    #test
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #build envs
    envs = SyncVectorEnv([make_env(args.gym_id, args.seed+i,i,
                                   args.capture_video,run_name)
                          for i in range(args.num_envs)])

    print('envs.single_observation_space.shape',envs.single_observation_space.shape)
    print('envs.single_action_space.shape',envs.single_action_space.shape)

    agent=PPO_agent(envs,device,lr=args.learning_rate,writer=writer)
    if args.from_pretrained:
        agent.model.load_state_dict(torch.load(f'./checkpoint/PPO_v4.pth', map_location=device))
    agent.train()
    torch.save(agent.model.state_dict(), f'./checkpoint/PPO_v{output_index}.pth')
    
    
    
    """
    output=1
    python -m train --gym-id humanoid-walk \
                    --total-timesteps 150000000 \
                    --learning-rate 1e-4 \
                    --num-envs 4 \
                    --num-minibatches 64 \
                    --num-steps 4096 \
                    --update-epochs 6 \
                    --max-grad-norm 0.5 \
                    --clip-coef 0.15 \
                    --vf-coef 0.2 \
                    --ent-coef 0.0003 \
                    --gamma 0.99 \
                    --gae-lambda 0.95 \
                    --capture-video False \
                    --from-pretrained False \
                         
    output=2
    python -m train --gym-id humanoid-walk \
                    --total-timesteps 150000000 \
                    --learning-rate 1.5e-4 \
                    --num-envs 4 \
                    --num-minibatches 64 \
                    --num-steps 4096 \
                    --update-epochs 4 \
                    --max-grad-norm 0.5 \
                    --clip-coef 0.2 \
                    --vf-coef 0.22 \
                    --ent-coef 0.0003 \
                    --gamma 0.99 \
                    --gae-lambda 0.95 \
                    --capture-video False \
                    --from-pretrained False \
                        
    output=4
python -m train --gym-id humanoid-walk \
                    --total-timesteps 150000000 \
                    --learning-rate 8e-5 \
                    --num-envs 4 \
                    --num-minibatches 64 \
                    --num-steps 4096 \
                    --update-epochs 10 \
                    --max-grad-norm 0.5 \
                    --clip-coef 0.2 \
                    --vf-coef 0.2 \
                    --ent-coef 0.0003 \
                    --gamma 0.99 \
                    --gae-lambda 0.95 \
                    --capture-video False \
    output=5
    python -m train --gym-id humanoid-walk \
                    --total-timesteps 150000000 \
                    --learning-rate 1.5e-4 \
                    --num-envs 4 \
                    --num-minibatches 32 \
                    --num-steps 2048 \
                    --update-epochs 4 \
                    --max-grad-norm 0.5 \
                    --clip-coef 0.2 \
                    --vf-coef 0.22 \
                    --ent-coef 0.0003 \
                    --gamma 0.99 \
                    --gae-lambda 0.95 \
                    --capture-video False \
                    --from-pretrained False \
    
    """