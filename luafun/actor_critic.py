import torch
import torch.nn as nn
import torch.distributions as distributions

import luafun.stitcher as obs
import luafun.game.action as actions
import luafun.game.constants as const


class CategoryEncoder(nn.Module):
    """Takes a hot encoded vector and returns a compact representation of the category

    This is used so the network can learn what abilities do.
    It essentially takes a meaningless one-hot vector and transform it into a vector.

    Notes
    -----
    This does not need to be deep, this is simply mapping a category to a vector.


    Examples
    --------
    The example below highlight that a one-hot encoded vector multiplied to a weight matrix
    simply select the column of the weight matrix.

    Applied to our case, this means we transform a category into a learned vector representation
    for that item/ability

    >>> cat_count = 10
    >>> vector_size = 5

    >>> category = torch.zeros((cat_count,))
    >>> category[0] = 1

    >>> category_vector = torch.randn((vector_size,))
    >>> category_vector.shape
    torch.Size([5])

    >>> encoder = CategoryEncoder(cat_count, vector_size)
    >>> encoder.linear.weight.shape
    torch.Size([5, 10])

    >>> encoder.linear.weight[:, 0] = category_vector

    >>> batch =  torch.zeros((1, cat_count))
    >>> batch[0, :] = category

    >>> result = encoder(batch)
    >>> result.shape
    torch.Size([1, 5])

    >>> (result - category_vector).square().sum().item() < 1e-5
    True
    """
    def __init__(self, in_size, out_size=128):
        super(CategoryEncoder, self).__init__()
        self.linear: nn.Module = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x):
        return self.linear(x)


class AbilityEncoder(CategoryEncoder):
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
        super(AbilityEncoder, self).__init__(in_size=1024, out_size=120)


class HeroEncoder(CategoryEncoder):
    """Encode a hero into a vector"""

    def __init__(self, out_size=128):
        super(HeroEncoder, self).__init__(in_size=const.HERO_COUNT, out_size=out_size)

    def forward(self, x):
        return self.embedder(x)


class SelectionCategorical(nn.Module):
    """Select a Categorical value from a state

    Notes
    -----

    """

    def __init__(self, state_shape, n_classes, n_hidden=None):
        super(SelectionCategorical, self).__init__()
        if n_hidden is None:
            n_hidden = int(n_classes)

        self.selector = nn.Sequential(
            nn.Linear(state_shape, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
            nn.Softmax(dim=-1)
        )

    def argmax(self, x):
        """Evaluation mode without exploration"""
        with torch.no_grad():
            action_probs = self.selector(x)
            return torch.argmax(action_probs)

    def sampled(self, x):
        action_probs = self.selector(x)

        # instead of picking the most likely
        # we sample from a distribution
        # this makes our actor discover need strategies
        dist = distributions.Categorical(action_probs)
        action = dist.sample()

        # Used for the cost function
        logprobs = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprobs, entropy


# I think the most challenging is probably how to make the network select an Entity
# Entity come and go and we need to select one for a few action, this include trees
#   * Trees might be fairly far from the hero (Timbersaw) or fairly close (tango)
#   * Enemy entities could be far and close as well (Spirit breaker)
#   * allies need to be there as well for buff spell
#
#  We fix that issue by simply finding the tree closest to the selected location
class EntitySelector(nn.Module):
    """Select the entity we want to move/attack/use ability on.
    It takes a variable number of entites it needs to select from

    """
    def __init__(self):
        super(EntitySelector, self).__init__()


class ItemPurchaser(SelectionCategorical):
    """Select the item we want to buy"""
    def __init__(self, hidden_state_size, item_count):
        super(ItemPurchaser, self).__init__()


class SimpleDrafter(nn.Module):
    """Works by encoding one-hot vectors of heroes to a dense vector which makes the internal state of the model.
    Decoders are then used to extract which pick or ban should be made given the information present

    Notes
    -----
    The drafter as a whole can only be trained on finished matches, but
    underlying parts of the network such as the ``hero_encoder`` can and will
    be trained continuously during the game.

    Examples
    --------
    >>> _ = torch.manual_seed(0)
    >>> draft_status = torch.zeros((obs.DraftFields.Size, const.HERO_COUNT))

    Set draft status

    >>> for i in range(obs.DraftFields.Size):
    ...     draft_status[i][0] = 1

    Batched version

    >>> batch_size = 5
    >>> draft_batch = torch.zeros((batch_size, obs.DraftFields.Size, const.HERO_COUNT))

    Insert draft to batch

    >>> draft_batch[0, :] = draft_status
    >>> draft_batch.shape
    torch.Size([5, 24, 121])

    >>> drafter = SimpleDrafter()
    >>> select, ban = drafter(draft_batch)
    >>> select.shape
    torch.Size([5])
    >>> hero = select[0]
    >>> hero
    tensor(69)
    """

    # Check logic
    # >>> batch = torch.randn((batch_size, obs.DraftFields.Size, const.HERO_COUNT))
    # >>> batch.shape
    # torch.Size([5, 24, 121])
    # >>> flat_draft = batch.view(batch_size * obs.DraftFields.Size, const.HERO_COUNT)
    # >>> flat_draft.shape
    # torch.Size([120, 121])
    #
    # >>> for b in range(batch_size):
    # ...     for d in range(obs.DraftFields.Size):
    # ...         fb = batch[b, d, :]
    # ...         nb = flat_draft[b * obs.DraftFields.Size + d, :]
    # ...         print((nb - fb).square().sum())

    def __init__(self):
        super(SimpleDrafter, self).__init__()
        self.encoded_vector = 64
        self.hero_encoder = CategoryEncoder(const.HERO_COUNT, self.encoded_vector)

        self.hidden_size = obs.DraftFields.Size * self.encoded_vector

        self.hero_select = SelectionCategorical(self.hidden_size, const.HERO_COUNT)
        self.hero_ban    = SelectionCategorical(self.hidden_size, const.HERO_COUNT)

    def forward(self, draft):
        # draft         : (batch_size x 24 x const.HERO_COUNT)
        # flat_draft    : (batch_size * 24 x const.HERO_COUNT)
        # encoded_flat  : (batch_size * 24 x 64)
        # encoded_draft : (batch_size x 24 * 64)

        batch_size = draft.shape[0]
        flat_draft = draft.view(batch_size * obs.DraftFields.Size, const.HERO_COUNT)

        encoded_flat = self.hero_encoder(flat_draft)
        encoded_draft = encoded_flat.view(batch_size, self.hidden_size)

        select, _, _ = self.hero_select(encoded_draft)
        ban, _, _ = self.hero_ban(encoded_draft)

        return select, ban


class HeroModel(nn.Module):
    """Change the batch size to group Hero computation

    Notes
    -----
    The approach is different from OpenAI.
    OpenAI computed a set of action that could be performed at a given time
    and then selected one of those actions.

    We are having a small network for each action argument

    OpenAI used 16 observations

    Examples
    --------

    >>> from luafun.game.action import ARG

    >>> _ = torch.manual_seed(0)
    >>> input_size = 1024
    >>> seq = 16
    >>> batch_size = 10

    >>> model = HeroModel(batch_size, seq, input_size)

    >>> batch = torch.randn(batch_size, seq, input_size)

    >>> with torch.no_grad():
    ...     act, _, _ = model.sampled(batch)

    Returns the actions to take for each bots/players
    >>> act[ARG.action].shape
    torch.Size([10])

    >>> player = 0

    Selected action
    >>> act[ARG.action][player]
    tensor(12)

    Vector location
    >>> act[ARG.vLoc][player]
    tensor([ 0.0105, -0.0105])

    Ability to use
    >>> act[ARG.nSlot][player]
    tensor(15)

    Item Swap
    >>> act[ARG.ix2][player]
    tensor(12)

    Item to buy
    >>> act[ARG.sItem][player]
    tensor(134)
    """

    def __init__(self, batch_size, seq, input_size):
        super(HeroModel, self).__init__()
        # preprocess a spacial observation with a specialized network
        self.state_preprocessor = nn.Module()

        # Process our flatten world observation vector
        # Generates a hidden state that is decoded by smaller network
        # which returns the actual action to take

        self.hidden_size = int(input_size * 0.55)
        self.input_size = int(input_size)
        self.lstm_layer = 3

        # input of shape  (batch, seq_len, input_size)
        # output of shape (batch, seq_len, hidden_size)
        self.internal_model = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layer,
            bias=True,
            batch_first=True,
        )

        self.batch_size = batch_size
        # Learn the initial parameters
        self.h0_init = nn.Parameter(torch.zeros(self.lstm_layer, self.batch_size, self.hidden_size))
        self.c0_init = nn.Parameter(torch.zeros(self.lstm_layer, self.batch_size, self.hidden_size))

        self.h0 = None
        self.c0 = None

        ability_count = len(actions.ItemSlot) + len(actions.AbilitySlot)

        # Those sub networks are small and act as state decoder
        # to return the precise action that is the most suitable
        self.ability = SelectionCategorical(self.hidden_size, ability_count)
        self.action = SelectionCategorical(self.hidden_size, len(actions.Action))
        self.swap = SelectionCategorical(self.hidden_size, len(actions.ItemSlot))
        self.item = SelectionCategorical(self.hidden_size, const.ITEM_COUNT)

        # Normalized Vector location
        ph = int(self.hidden_size / 2)
        self.position = nn.Sequential(
            nn.Linear(self.hidden_size, ph),
            nn.ReLU(),
            nn.Linear(ph, 2),
            nn.Softmax(dim=-1)
        )

        # Unit is retrieved from position
        # self.unit = SelectionHandle()

        # Tree ID is retrieved from position
        # self.tree = SelectTree()

        # Rune ID is retrieved from position
        # self.runes = SelectionCategorical(self.hidden_size, len(actions.RuneSlot) + 1)

        self.ability_embedder = AbilityEncoder()
        self.hero_embedder = HeroEncoder()

    def argmax(self, x):
        """Pure inference, no exploration"""
        with torch.no_grad():
            if self.h0 is None:
                hidden, (hn, cn) = self.internal_model(x, (self.h0_init, self.c0_init))
            else:
                hidden, (hn, cn) = self.internal_model(x, (self.h0, self.c0))

            self.h0, self.c0 = hn, cn

            # select the last state
            hidden = hidden[:, -1]

            # unit = self.unit(hidden)
            # tree = self.tree(hidden)
            # rune = self.runes(hidden)

            # Sampled action
            action  = self.action.argmax(hidden)
            ability = self.ability.argmax(hidden)
            swap    = self.swap.argmax(hidden)
            item    = self.item.argmax(hidden)

            # change the output from [0, 1] to [-1, 1]
            vec = self.position(hidden) * 2 - 1

            msg = {
                actions.ARG.action: action,
                actions.ARG.vLoc: vec,
                actions.ARG.sItem: item,
                actions.ARG.nSlot: ability,
                actions.ARG.ix2: swap
            }

            log_progs = None
            entropy = None

            return msg, log_progs, entropy

    def sampled(self, x):
        """Inference with space exploration"""
        if self.h0 is None:
            hidden, (hn, cn) = self.internal_model(x, (self.h0_init, self.c0_init))
        else:
            hidden, (hn, cn) = self.internal_model(x, (self.h0, self.c0))

        self.h0, self.c0 = hn, cn

        hidden = hidden[:, -1]
        # unit = self.unit(hidden)
        # tree = self.tree(hidden)
        # rune = self.runes(hidden)

        # Sampled action
        action,  lp_ac, ent_ac = self.action.sampled(hidden)
        ability, lp_ab, ent_ab = self.ability.sampled(hidden)
        swap,    lp_sw, ent_sw = self.swap.sampled(hidden)
        item,    lp_it, ent_it = self.item.sampled(hidden)

        # change the output from [0, 1] to [-1, 1]
        vec = self.position(hidden) * 2 - 1

        msg = {
            actions.ARG.action: action,
            actions.ARG.vLoc: vec,
            actions.ARG.sItem: item,
            actions.ARG.nSlot: ability,
            actions.ARG.ix2: swap,
        }

        log_progs = (lp_ac, lp_ab, lp_sw, lp_it)
        entropy   = (ent_ac, ent_ab, ent_sw, ent_it)

        return msg, log_progs, entropy

    def forward(self, x):
        return self.sampled(x)


class ActorCritic(nn.Module):
    """Generic Implementation of the ActorCritic you should subclass to implement your model"""
    def __init__(self, actor, critic):
        super(ActorCritic, self).__init__()
        self.actor: nn.Module = actor
        self.critic: nn.Module = critic

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

            dist = distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprobs = dist.log_prob(action)

            return action

    def eval(self, state, action):
        """Compute the value of the action for training"""
        with torch.enable_grad():
            # Do the forward pass so we have gradients for the optimization
            action_probs = self.actor(state)
            dist = distributions.Categorical(action_probs)
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
