import numbers

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

EPSILON = 1e-8

class EnvInterface:
    def reset(self, seed=None):
        """ Returns:
            observation
        """
        raise NotImplementedError()
    def step(self, action):
        """ Returns:
            observation
            terminated
            reward
        """
        raise NotImplementedError()
    
class GridWorldPointTargetEnv:
    def __init__(self,
              grid_size=10,
              episode_length=20,
              dimensions=2,
              set_target = None,
              set_state = None,):
        self.grid_size      = grid_size
        self.dimensions     = dimensions
        self.episode_length = episode_length
        self.set_target     = set_target
        self.set_state      = set_state
    
    def __get_obs(self):
        observation = (torch.tensor(np.concatenate((self.target, self.state)), dtype=torch.float32)[None, ...] / self.grid_size) * 2 - 1
        # observation = torch.cat([observation, torch.tensor([[self.remaining_actions / self.episode_length * 2 - 1]])], dim=1)
        return observation

    def reset(self,
              seed=None):
        np.random.seed(seed)
        if self.set_target is not None:
            if self.dimensions == 1 and isinstance(self.set_target, numbers.Number):
                self.set_target = [self.set_target]
            self.target = np.array(self.set_target)
        else:
            self.target = (np.random.random(size=self.dimensions) * self.grid_size).astype(int)
        if self.set_state is not None:
            if self.dimensions == 1 and isinstance(self.set_state, numbers.Number):
                self.set_state = [self.set_state]
            self.state = np.array(self.set_state)
        else:
            self.state = (np.random.random(size=self.dimensions) * self.grid_size).astype(int)
        self.remaining_actions = self.episode_length
        return self.__get_obs()

    def step(self, action):
        if self.dimensions == 1:
            action_choices = np.array([
                [-1], [0], [1]
            ])
        elif self.dimensions == 2:            
            action_choices = np.array([
                [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]
            ])

        self.state += action_choices[action]
        self.state = np.clip(self.state, 0, self.grid_size)
        self.remaining_actions -= 1

        terminated = self.remaining_actions == 0
        reward = -np.linalg.norm(self.state - self.target) / self.grid_size
        
        return self.__get_obs(), terminated, reward


    
class SimpleReLuNetwork(nn.Module):
    def __init__(self, input_dims, output_dims,
                 out_logsoftmax=False,
                 fixed_init=None,
                 hidden_dims=[16]):
        super().__init__()
        layers = [nn.Linear(input_dims, hidden_dims[0]), nn.ReLU()]
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dims))
        if out_logsoftmax:
            layers.append(nn.LogSoftmax(dim=-1))
        self.net = nn.Sequential(*layers)

        if fixed_init is not None:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    m.weight.data.fill_(fixed_init[0])
                    m.bias.data.fill_(fixed_init[1])
            self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.old_logits = []
        self.rewards = []
        self.terminateds = []
        self.final_size = None
    
    def add(self, action, logits, observation, terminated, reward):
        self.actions.append(action)
        self.old_logits.append(logits)
        self.observations.append(observation)
        self.terminateds.append(terminated)
        self.rewards.append(reward)

    def finalize(self):
        if self.final_size is not None:
            raise RuntimeError("RolloutBuffer cannot be finalized twice")
        self.actions =      torch.concatenate(self.actions)
        self.observations = torch.concatenate(self.observations)
        self.old_logits =   torch.concatenate(self.old_logits)
        self.rewards =      torch.tensor(self.rewards)
        self.terminateds =  torch.tensor(self.terminateds)
        self.final_size = len(self.actions)
    
    def load(self, data):
        self.actions =      torch.tensor(data['actions'])
        self.old_logits =   torch.tensor(data['old_logits'])
        self.observations = torch.tensor(data['observations'])
        self.terminateds =  torch.tensor(data['terminateds'])
        self.rewards =      torch.tensor(data['rewards'])
        self.final_size = len(self.actions)
        return self
    
    def dump(self):
        return dict(
            actions=        self.actions.tolist(),
            old_logits=     self.old_logits.tolist(),
            observations=   self.observations.tolist(),
            terminateds=    self.terminateds.tolist(),
            rewards=        self.rewards.tolist(),
        )

def distribution_sample(logits, seed=None):
    if seed is not None:
        torch.manual_seed(seed[0])
        seed[:] = np.roll(seed, 1)
    distr = torch.distributions.categorical.Categorical(logits=logits)

    # return np.argmax(logits)[None, None]
    return distr.sample([1])













def show_lineworld_rollouts(env: GridWorldPointTargetEnv, rollout_buffer: RolloutBuffer):
    fig = plt.figure()
    i = 0
    curr_rollout = []
    success = []
    for goal, obs, term in zip(rollout_buffer.observations[:, 0], rollout_buffer.observations[:, 1], rollout_buffer.terminateds):
        curr_rollout.append((obs - goal + 1) / 2 * env.grid_size)
        if term:
            plt.plot(curr_rollout, color='blue', alpha=0.1, linewidth=4.0)
            curr_rollout = []
            success.append(torch.all((torch.abs(obs - goal) <= 1e-10)).item())
    plt.xlabel("Timestep")
    plt.ylabel("Difference from target")
    return np.sum(success) / len(success)

def get_success_rate(env: GridWorldPointTargetEnv, rollout_buffer: RolloutBuffer):
    fig = plt.figure()
    i = 0
    rollouts = []
    curr_rollout = []
    success = []
    for goal, obs, term in zip(rollout_buffer.observations[:, :2], rollout_buffer.observations[:, 2:], rollout_buffer.terminateds):
        curr_rollout.append((obs - goal + 1) / 2 * env.grid_size)
        if term:
            rollouts.append(curr_rollout)
            success.append(torch.all((torch.abs(obs - goal) <= 1e-10)).item())
            curr_rollout = []
    return np.sum(success) / len(success)









class OpenAIGymEnv:
    def __init__(self, vis=False):
        import gymnasium as gym
        if vis:
            self.gym = gym.make('CartPole-v1', render_mode="human")
        else:
            self.gym = gym.make('CartPole-v1')
        self.gym = gym.wrappers.NormalizeObservation(self.gym)
    def reset(self,seed=None):
        obs, _ = self.gym.reset()
        return torch.tensor(obs, dtype=torch.float32)[None,...]

    def step(self, action):            
        obs, reward, terminated, truncated, _ = self.gym.step(action.item())
        return torch.tensor(obs, dtype=torch.float32)[None,...], terminated or truncated, reward