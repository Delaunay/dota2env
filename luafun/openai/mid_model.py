import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = None
        self.critic = None

    def act(self, state):
        """Infer the action to take"""
        with torch.no_grad():
            action_probs = self.actor(state)

            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            action = dist.sample()

            return action, action_logprobs

    def eval(self, state, action):
        """Compute the value of the action for training"""
        with torch.grad:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            value = self.critic(state)
            return value, action_logprobs, dist_entropy
            
