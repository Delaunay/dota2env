import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Generic Implementation of the ActorCritic you should subclass to implement your model"""
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor: nn.Module = None
        self.critic: nn.Module = None

    def actor_weights(self):
        """Returns the actor weights"""
        return self.actor.state_dict()

    def load_actor(self, w):
        """Update the actors weights"""
        self.actor.load_state_dict(w)

    def act(self, state):
        """Infer the action to take, we only need actor when doing inference"""
        with torch.no_grad():
            action_probs = self.actor(state)

            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            action = dist.sample()

            return action

    def eval(self, state, action):
        """Compute the value of the action for training"""
        with torch.enable_grad():
            # Do the forward pass so we have gradients for the optimization
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            # Value the current state
            value = self.critic(state)
            return value, action_logprobs, dist_entropy

    def forward(self):
        raise NotImplementedError


class BaseActorCritic(ActorCritic):
    """Default actor critic implementation for debugging"""

    def __init__(self, state_dim, action_dim, n_latent_var):
        super(BaseActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )
