import torch
import torch.nn as nn


class CategoryEmbedder(nn.Module):
    """Takes a hot encoded vector and returns a compact representation of the category

    This is used so the network can learn what abilities do.
    It essentially takes a meaningless one-hot vector and transform it into a vector.
    """
    def __init__(self, in_size, n_latent, out_size=128):
        super(CategoryEmbedder, self).__init__()

        self.embedder = nn.Sequential(
            nn.Linear(in_size, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, out_size),
        )
        # b x in_size => b x out_size

    def forward(self, x):
        return self.embedder(x)


class AbilityEmbedder(CategoryEmbedder):
    """Tansform a one hot encoded ability into a compact vector
    We hope that abilities that are alike will produce similar output vector (abelian group)

    i.e Poison Attack should be close to Liquid Fire because both are attack modifiers with slow
    The one-hot encoded vector is purely arbitrary the output vector should show some meaning

    Notes
    -----
    Dota2 has 120 Heores x 4 Abilities + 208 Items = 688 abilities,
    but a few abilities share similar attributes so we picked 512 has the number of latent
    variables
    """
    def __init__(self, n_latent=512):
        super(CategoryEmbedder, self).__init__(in_size=1024, n_latent=n_latent, out_size=120)


class HeroEmbedder(nn.Module):
    """HeroSummary takes batch of ability processed by the AbilityEmbedder and returns a compact vector
    representing a summary of the hero.
    """

    def __init__(self, in_size=(24, 120), n_latent=240, out_size=120):
        super(HeroSummary, self).__init__()
        n_abilities, size = in_size

        # b x #abilities x #out_size => b x #out_size
        self.summary = nn.Sequential(
            nn.Conv1d(n_abilities, 1, kernel_size=3)
            nn.Flatten()
            nn.Linear(in_size, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, out_size),
        )

    def forward(self, x):
        return self.embedder(x)


class SelectionCategorical(nn.Module):
    """Select a Categorical value from a state

    Notes
    -----

    """

    def __init__(self, state_shape, n_latent_var, n_classes):
        super(HeroModel, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(state_shape, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_classes),
            nn.Softmax(dim=-1)
        )

    def eval(self, x):
        """Evaluation mode without exploration"""
        with torch.no_grad():
            action_probs = self.selector(x)
            return torch.argmax(action_probs)

    def forward(self, x):
        action_probs = self.selector(x)

        # instead of picking the most likely
        # we sample from a distribution
        # this makes our actor discover need strategies
        dist = Categorical(action_probs)
        action = dist.sample()

        # Used for the cost function
        logprobs = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprobs, entropy


class HeroModel(nn.Module):
    """Change the batch size to group Hero computation

    Notes
    -----
    The approach is different from OpenAI.
    OpenAI computed a set of action that could be performed at a given time
    and then selected one of those actions.

    We are having a small network for each action argument
    """

    def __init__(self):
        super(HeroModel, self).__init__()

        # ~25 actions
        self.action = SelectionCategorical()
        # ~12 Ability 4 + 6 + 2 + talent ?
        # we can reuse that one for sAbility (Learn)
        self.ability = SelectionCategorical()

        # 208 Items to select from
        self.item = SelectionCategorical()

        self.ability_embedder = AbilityEmbedder()
        self.hero_embedder = HeroEmbedder()

        # Vector location

        # Unit Selection



    def forward(self, state):
        pass


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
