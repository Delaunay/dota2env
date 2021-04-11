from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from luafun.model.actor_critic import ActorCritic


class TrainEngine:
    def __init__(self, train, model, args):
        self.engine = LocalTrainEngine(*args)

    @property
    def weights(self):
        """Returns new weights if ready, returns none if not ready"""
        return None

    def push(self, uid, state, reward, done, info, action, logprob, filter):
        self.engine.push(uid, state, reward, done, info, action, logprob, dilter)


@dataclass
class Observation:
    action: torch.Tensor
    state: torch.Tensor
    logprob: torch.Tensor
    reward: torch.Tensor
    is_terminal: torch.Tensor
    filter: torch.Tensor
    info: dict


class RolloutDataset:
    def __init__(self, timestep):
        # One time step is 4 frames (0.133 seconds)
        self.timestep = timestep
        # One sample is 16 time steps (2.1333 seconds)
        self.sample = 16
        # One espisode is 16 Samples (34.1328 seconds)
        self.episode = 16

        self.memory = defaultdict(list)
        self.size = 0

    def push(self, uid, state, reward, done, info, action, logprob, filter):
        memory = self.memory[uid]
        self.size += 1
        memory.append(Observation(
            action,
            state,
            logprob,
            reward,
            done,
            filter,
            info
        ))

    def reset(self):
        self.memory = defaultdict(list)
        self.size = 0

    @property
    def game_count(self) -> int:
        return len(self.memory)

    def game_size(self, uid) -> int:
        uid = list(self.memory.keys())[uid]
        return len(self.memory.get(uid, []))

    def __getitem__(self, uid, time):
        #  Make sure our index are correct
        uid = uid % len(self.memory)
        data = self.memory[uid]
        time = (time % len(data)) + self.timestep
        # ==

        sample = data[time - self.timestep:time]
        return sample


class RolloutSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def new_sample(self):
        # Make a tensor for a single game
        uid = torch.randint(self.dataset.game_count, (1,)).item()
        time = self.dataset.game_size(uid)

        obs: List[Observation] = self.dataset[uid, time]

        states = torch.stack([m.state for m in obs])
        action = torch.stack([m.action for m in obs])
        logprob = torch.stack([m.logprob for m in obs])

        # Compute reward
        done = torch.stack([m.is_terminal for m in obs])
        reward = torch.stack([m.reward for m in obs])

        return states, action, logprob, reward


class LocalTrainEngine:
    def __init__(self, obssize, batch=16, timestep=16):
        self.dataset = RolloutDataset(timestep)
        self.sampler = RolloutSampler(self.dataset, batch)
        self.actor_critic = ActorCritic(batch, timestep, obssize)
        self.ppo_epochs = 10
        self.loss = nn.MSELoss()
        self.eps_clip = 1e-3

        # self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr, betas=betas)

    @property
    def weights(self):
        return None

    def push(self, uid, state, reward, done, info, action, logprob):
        self.dataset.push(uid, state, reward, done, info, action, logprob)

    def train(self):
        batch = self.sampler.new_sample()
        print(batch)

        # for _ in range(self.ppo_epochs):
        #     state, action, logprobs = memory.states, memory.action, action.logprobs
        #
        #     action_logprobs, state_value, dist_entropy = self.actor_critic.evaluate(state, action)
        #
        #     ratios = torch.exp(action_logprobs - logprobs.detach())
        #
        #     advantages = rewards - state_value.detach()
        #
        #     surr1 = ratios * advantages
        #     surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        #     loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_value, rewards) - 0.01 * dist_entropy
        #
        #     # take gradient step
        #     self.optimizer.zero_grad()
        #     loss.mean().backward()
        #     self.optimizer.step()

        # Save Actor

        # Save Policy

        # Update Actor


