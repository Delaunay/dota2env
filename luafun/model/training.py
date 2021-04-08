from collections import defaultdict
import torch
import torch.nn as nn

from luafun.model.actor_critic import ActorCritic


class TrainEngine:
    def __init__(self, train, args):
        pass

    @property
    def weights(self):
        """Returns new weights if ready, returns none if not ready"""
        return None

    def push(self, *args):
        pass


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class LocalTrainEngine:
    def __init__(self):
        self.memory = defaultdict(Memory)
        self.actor_critic = ActorCritic()
        self.ppo_epochs = 10
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr, betas=betas)
        self.loss = nn.MSELoss()
        self.eps_clip = 1e-3

    @property
    def weights(self):
        return None

    def push(self, uid, state, reward, done, info, action, logprobs):
        memory = self.memory[uid]

        memory.states.append(state)
        memory.actions.append(action)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

    def train(self):
        for _ in range(self.ppo_epochs):
            state, action, logprobs = memory.states, memory.action, action.logprobs

            action_logprobs, state_value, dist_entropy = self.actor_critic.evaluate(state, action)

            ratios = torch.exp(action_logprobs - logprobs.detach())

            advantages = rewards - state_value.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_value, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Save Actor

        # Save Policy

        # Update Actor


