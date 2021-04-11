import torch
import torch.nn as nn
import torch.distributions as distributions

from luafun.draft import DraftFields
from luafun.game.action import Action, AbilitySlot, ARG
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

    def forward(self, x):
        return self.selector(x)


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
    """Select the item we want to buy

    """
    def __init__(self, hidden_state_size, item_count=const.ITEM_COUNT):
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
    >>> draft_status = torch.zeros((DraftFields.Size, const.HERO_COUNT))

    Set draft status

    >>> for i in range(DraftFields.Size):
    ...     draft_status[i][0] = 1

    Batched version

    >>> batch_size = 5
    >>> draft_batch = torch.zeros((batch_size, DraftFields.Size, const.HERO_COUNT))

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

        self.hidden_size = DraftFields.Size * self.encoded_vector

        self.hero_select = SelectionCategorical(self.hidden_size, const.HERO_COUNT)
        self.hero_ban    = SelectionCategorical(self.hidden_size, const.HERO_COUNT)

    def forward(self, draft):
        # draft         : (batch_size x 24 x const.HERO_COUNT)
        # flat_draft    : (batch_size * 24 x const.HERO_COUNT)
        # encoded_flat  : (batch_size * 24 x 64)
        # encoded_draft : (batch_size x 24 * 64)

        batch_size = draft.shape[0]
        flat_draft = draft.view(batch_size * DraftFields.Size, const.HERO_COUNT)

        encoded_flat = self.hero_encoder(flat_draft)
        encoded_draft = encoded_flat.view(batch_size, self.hidden_size)

        probs_select = self.hero_select(encoded_draft)
        pros_ban = self.hero_ban(encoded_draft)

        dist = distributions.Categorical(probs_select)
        select = dist.sample()
        # action_logprobs = dist.log_prob(select)
        # dist_entropy = dist.entropy()

        dist = distributions.Categorical(pros_ban)
        ban = dist.sample()
        # action_logprobs = dist.log_prob(ban)
        # dist_entropy = dist.entropy()

        return select, ban


class HeroModel(nn.Module):
    """Change the batch size to group Hero computation.
    This Module returns only the probabilities the actual action will be selected later,
    when the action filter is applied

    Notes
    -----
    The approach is different from OpenAI.
    OpenAI computed a set of action that could be performed at a given time
    and then selected one of those actions.

    We are having a small network for each action argument

    OpenAI approach was to make network select action from a set of possible actions and select unit from a set of
    possible unit.

    We are trying not to do that, our network select from the set of all actions.

    The returned vector act as an attention mechanism as it is with this vector that
    entities will be selected


    Dota has 208 items but this does not count the recipes and item level.
    In reality we need to choose among 242 options.

    Heroes have only 6 abilities but we need to learn 8 talents as well.
    6 abilities + 8 talents + 6 items + 2 items = 22 actions

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
    ...     act = model(batch)

    Returns the actions to take for each bots/players
    >>> act[ARG.action].shape
    torch.Size([10, 32])

    >>> player = 0

     Which action we want to use by probabilities
    >>> act[ARG.action][player]
    tensor([0.0291, 0.0303, 0.0330, 0.0347, 0.0268, 0.0281, 0.0314, 0.0324, 0.0276,
            0.0353, 0.0272, 0.0326, 0.0353, 0.0298, 0.0324, 0.0280, 0.0269, 0.0354,
            0.0285, 0.0263, 0.0275, 0.0370, 0.0314, 0.0343, 0.0334, 0.0298, 0.0284,
            0.0356, 0.0335, 0.0298, 0.0353, 0.0332])

    Vector location
    >>> act[ARG.vLoc][player]
    tensor([-0.0377,  0.0377])

    Which ability we want to use by probabilities
    >>> act[ARG.nSlot][player]
    tensor([0.0258, 0.0211, 0.0266, 0.0246, 0.0256, 0.0248, 0.0222, 0.0252, 0.0227,
            0.0250, 0.0270, 0.0203, 0.0273, 0.0224, 0.0252, 0.0261, 0.0207, 0.0266,
            0.0242, 0.0240, 0.0210, 0.0244, 0.0274, 0.0240, 0.0267, 0.0234, 0.0253,
            0.0251, 0.0250, 0.0265, 0.0251, 0.0223, 0.0267, 0.0235, 0.0263, 0.0270,
            0.0215, 0.0208, 0.0225, 0.0271, 0.0208])

    Item Swap probabilities
    >>> act[ARG.ix2][player]
    tensor([0.0470, 0.0580, 0.0639, 0.0531, 0.0612, 0.0557, 0.0458, 0.0738, 0.0472,
            0.0654, 0.0647, 0.0609, 0.0666, 0.0558, 0.0590, 0.0599, 0.0619])

    Item to buy
    >>> act[ARG.sItem][player].shape
    torch.Size([288])

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

        ability_count = len(AbilitySlot)

        # Those sub networks are small and act as state decoder
        # to return the precise action that is the most suitable
        self.ability = SelectionCategorical(self.hidden_size, ability_count)
        self.action = SelectionCategorical(self.hidden_size, len(Action))
        self.swap = SelectionCategorical(self.hidden_size, len(const.ItemSlot))
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

    def forward(self, x):
        """Inference with space exploration"""
        print(x)
        
        if self.h0 is None:
            hidden, (hn, cn) = self.internal_model(x, (self.h0_init, self.c0_init))
        else:
            hidden, (hn, cn) = self.internal_model(x, (self.h0, self.c0))

        self.h0, self.c0 = hn, cn

        hidden = hidden[:, -1]

        # Sampled action
        action  = self.action(hidden)
        ability = self.ability(hidden)
        swap    = self.swap(hidden)
        item    = self.item(hidden)

        # change the output from [0, 1] to [-1, 1]
        vec = (self.position(hidden) * 2 - 1)

        msg = {
            ARG.action: action,
            ARG.vLoc: vec,
            ARG.sItem: item,
            ARG.nSlot: ability,
            ARG.ix2: swap,
        }

        return msg


class ActionSampler:
    """Select and preprocess action returned by our model"""
    CATEGORICAL_FIELDS = [ARG.action, ARG.sItem, ARG.nSlot, ARG.ix2]

    def argmax(self, msg, filter):
        """Inference only, no exploration"""
        fields = ActionSampler.CATEGORICAL_FIELDS
        logprobs = None
        entropy = None

        for i, field in enumerate(fields):
            prob = msg[field]

            # Apply the filter here
            prob = filter(prob)

            # Sample the action
            selected = torch.argmax(prob)
            msg[field] = selected

        return msg, logprobs, entropy

    def sampled(self, msg, filter):
        """Inference with exploration to help training"""
        fields = ActionSampler.CATEGORICAL_FIELDS

        logprobs = [None] * len(fields)
        entropy = [None] * len(fields)

        for i, field in enumerate(fields):
            # instead of picking the most likely
            # we sample from a distribution
            # this makes our actor discover need strategies
            prob = msg[field]

            # Apply the filter here
            prob = filter(prob)

            # Sample the action
            dist = distributions.Categorical(prob)
            selected = dist.sample()

            # Used for the cost function
            lp_sel = dist.log_prob(selected)
            en_sel = dist.entropy()

            logprobs[i] = lp_sel
            entropy[i] = en_sel
            msg[field] = selected

        return msg, logprobs, entropy


class ActorCritic(nn.Module):
    def __init__(self, batch_size, seq, input_size):
        super(ActorCritic, self).__init__()

        self.actor = HeroModel(batch_size, seq, input_size)
        self.critic = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, 1)
        )

    def infer(self, state):
        """Infer next move"""
        with torch.no_grad():
            return self.actor(state)

    def evaluate(self, state, action):
        """Evaluate the action taken """
        action_probs = self.action_layer(state)
        dist = distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)

        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

