import torch
import torch.nn as nn
import torch.distributions as distributions

from luafun.draft import DraftFields
import luafun.game.constants as const

from luafun.model.components import CategoryEncoder


class SparseDrafter(nn.Module):
    """This drafter try to use the natural sparsity that is coming from the input.
    """

    def __init__(self):
        super(SparseDrafter, self).__init__()

    def forward(self, x):
        pass


class LSTMDrafter(nn.Module):
    """
    Notes
    -----
    This is hopefully better than the simple drafter because it is aware than
    decisions are linked together

    Trained as stalled after a few epochs unfortunately
    reached 8.85727 after epoch 4358,
    reached 8.7028 after epoch 7292

    Examples
    --------
    >>> import torch
    >>> from luafun.dataset.draft import Dota2PickBan

    >>> dataset = Dota2PickBan('/home/setepenre/work/LuaFun/opendota_CM_20210421.zip', patch='7.29')
    >>> state, pic, ban = dataset[0]
    >>> state.shape
    torch.Size([12, 24, 122])

    12 Decision to make each decision has a state of 24 selected/ban heroes

    >>> batch = torch.stack([state, state])
    >>> batch.shape
    torch.Size([2, 12, 24, 122])

    >>> drafter = LSTMDrafter()
    >>> pick, ban = drafter(batch)
    >>> pick.shape
    torch.Size([2, 12, 122])
    """

    def __init__(self):
        super(LSTMDrafter, self).__init__()
        self.encoded_vector = 128

        self.hero_encoder = CategoryEncoder(const.HERO_COUNT, self.encoded_vector)
        self.lstm_in = DraftFields.Size * self.encoded_vector
        self.lstm_hidden = 256
        self.lstm_layer = 4

        self.rnn = nn.LSTM(
            # nonlinearity='tanh',
            batch_first=True,
            input_size=self.lstm_in,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layer,
            bidirectional=False,
        )

        # self.h0_init = nn.Parameter(torch.zeros(self.lstm_layer, self.batch_size, self.lstm_hidden))
        # self.c0_init = nn.Parameter(torch.zeros(self.lstm_layer, self.batch_size, self.lstm_hidden))

        self.hero_select = nn.Sequential(
            nn.Linear(self.lstm_hidden, const.HERO_COUNT),
            nn.Softmax(dim=-1)
        )

        self.hero_ban = nn.Sequential(
            nn.Linear(self.lstm_hidden, const.HERO_COUNT),
            nn.Softmax(dim=-1)
        )

    def encode_draft_fast(self, draft):
        # 2, 12, 24, 122
        bs, sq, hero, size = draft.shape

        #  flat_draft = torch.flatten(draft, end_dim=2)
        flat_draft = torch.flatten(draft, end_dim=2)

        # (2 * 12 * 24) x 122
        encoded_flat = self.hero_encoder(flat_draft)

        # 2 x 12 x 24 x (embedding_size: 128)
        encoded_draft = encoded_flat.view(bs, sq, DraftFields.Size, self.encoded_vector)

        # 2 x 12 x (24 * 128)
        encoded_draft = torch.flatten(encoded_draft, start_dim=2)
        return encoded_draft

    def encode_draft(self, draft):
        # 2, 12, 24, 122
        bs, sq, hero, _ = draft.shape

        result = torch.zeros(bs, sq, self.lstm_in).cpu()

        for b in range(bs):
            for s in range(sq):
                # 24 x self.encoded_vector
                encoded_hero = self.hero_encoder(draft[b, s])
                result[b, s] = torch.flatten(encoded_hero)

        return result

    def forward(self, draft):
        # 2, 12, 24, 122
        bs, sq, hero, size = draft.shape

        # 2 x 12 x (24 * 128)
        encoded_draft = self.encode_draft_fast(draft)

        common, _ = self.rnn(encoded_draft)  # , (self.h0_init, self.c0_init))

        #  torch.Size([2, 12, 256])
        common = torch.flatten(common, end_dim=1)

        #  torch.Size([2, 12, 122])
        pick = self.hero_select(common).view(bs, sq, const.HERO_COUNT)
        ban = self.hero_ban(common).view(bs, sq, const.HERO_COUNT)

        return pick, ban


class SimpleDrafter(nn.Module):
    """Works by encoding one-hot vectors of heroes to a dense vector which makes the internal state of the model.
    Decoders are then used to extract which pick or ban should be made given the information present

    Notes
    -----
    The drafter as a whole can only be trained on finished matches, but
    underlying parts of the network such as the ``hero_encoder`` can and will
    be trained continuously during the game.

    Seems to be impossible to train (from 9.5849 reached 9.1360 after 4300 epochs.
    Might be missing capacity to understand

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
    torch.Size([5, 121])

    >>> hero_ids = torch.argmax(select, 1)
    >>> hero_ids.shape
    torch.Size([5])
    >>> hero_ids[0]
    tensor(9)
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
        self.encoded_vector = 32

        self.hero_encoder = CategoryEncoder(const.HERO_COUNT, self.encoded_vector)
        self.hidden_size = DraftFields.Size * self.encoded_vector

        state_shape = self.hidden_size
        n_hidden = 512
        self.common = nn.Sequential(
            nn.Linear(state_shape, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.hero_select = nn.Sequential(
            nn.Linear(n_hidden, const.HERO_COUNT),
            nn.Softmax(dim=-1)
        )

        self.hero_ban = nn.Sequential(
            nn.Linear(n_hidden, const.HERO_COUNT),
            nn.Softmax(dim=-1)
        )

    def forward(self, draft):
        # draft         : (batch_size x 24 x const.HERO_COUNT)
        # flat_draft    : (batch_size * 24 x const.HERO_COUNT)
        # encoded_flat  : (batch_size * 24 x 64)
        # encoded_draft : (batch_size x 24 * 64)

        batch_size = draft.shape[0]
        flat_draft = draft.view(batch_size * DraftFields.Size, const.HERO_COUNT)

        encoded_flat = self.hero_encoder(flat_draft)
        encoded_draft = encoded_flat.view(batch_size, self.hidden_size)

        common = self.common(encoded_draft)

        probs_select = self.hero_select(common)
        probs_ban = self.hero_ban(common)

        return probs_select, probs_ban

    def sample(self, probs_select, probs_ban):
        dist = distributions.Categorical(probs_select)
        select = dist.sample()
        # action_logprobs = dist.log_prob(select)
        # dist_entropy = dist.entropy()

        dist = distributions.Categorical(probs_ban)
        ban = dist.sample()
        # action_logprobs = dist.log_prob(ban)
        # dist_entropy = dist.entropy()

        return select, ban


class DraftJudge(nn.Module):
    """Returns which faction is more likely to win the game given a draft"""
    def __init__(self):
        super(DraftJudge, self).__init__()

    def forward(self, x):
        pass
