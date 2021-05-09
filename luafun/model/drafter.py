from enum import IntEnum, auto

import torch
import torch.nn as nn
import torch.distributions as distributions

from luafun.draft import DraftFields
import luafun.game.constants as const

from luafun.model.components import CategoryEncoder, SkillBias


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

    def __init__(self, hero_encoder=None):
        super(LSTMDrafter, self).__init__()

        if hero_encoder is None:
            hero_encoder = CategoryEncoder(const.HERO_COUNT, 128)

        self.hero_vec = hero_encoder.out_size
        self.hero_encoder = hero_encoder

        self.lstm_in = DraftFields.Size * self.hero_vec
        self.lstm_hidden = 256
        self.lstm_layer = 4
        self.dummy_offset = const.HERO_LOOKUP.from_name('npc_dota_hero_target_dummy')['offset']

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

        self.critic = nn.Sequential(
            nn.Linear(self.lstm_hidden, 1),
        )

    def encode_draft_fast(self, draft):
        # 2, 12, 24, 122
        bs, sq, hero, size = draft.shape

        #  flat_draft = torch.flatten(draft, end_dim=2)
        flat_draft = torch.flatten(draft, end_dim=2)

        # (2 * 12 * 24) x 122
        encoded_flat = self.hero_encoder(flat_draft)

        # 2 x 12 x 24 x (embedding_size: 128)
        encoded_draft = encoded_flat.view(bs, sq, DraftFields.Size, self.hero_vec)

        # 2 x 12 x (24 * 128)
        encoded_draft = torch.flatten(encoded_draft, start_dim=2)
        return encoded_draft

    def encode_draft(self, draft):
        # 2, 12, 24, 122
        bs, sq, hero, _ = draft.shape

        result = torch.zeros(bs, sq, self.lstm_in).cpu()

        for b in range(bs):
            for s in range(sq):
                # 24 x self.hero_vec
                encoded_hero = self.hero_encoder(draft[b, s])
                result[b, s] = torch.flatten(encoded_hero)

        return result

    def filter(self, pick_probs, ban_probs, reserved):
        pick_filter = torch.ones_like(pick_probs)
        ban_filter = torch.ones_like(ban_probs)

        pick_filter[:, :, self.dummy_offset] = 0
        ban_filter[:, :, self.dummy_offset] = 0

        # set the probability of selected heroes to 0
        # No incorrect action can be made
        if reserved:
            for r in reserved:
                pick_filter[:, :, r] = 0
                ban_filter[:, :, r] = 0

        pick_probs = pick_probs * pick_filter
        ban_probs = ban_probs * ban_filter

        return pick_probs, ban_probs

    def action(self, draft, prev=None, reserved=None):
        # seq == 1
        draft = draft.unsqueeze(1)
        bs, sq, hero, size = draft.shape

        # 2 x 12 x (24 * 128)
        encoded_draft = self.encode_draft_fast(draft)

        common, next = self.rnn(encoded_draft, prev)  # , (self.h0_init, self.c0_init))

        #  torch.Size([2, 12, 256])
        common = torch.flatten(common, end_dim=1)

        #  torch.Size([2, 12, 122])
        pick_probs = self.hero_select(common).view(bs, sq, const.HERO_COUNT)
        ban_probs  = self.hero_ban(common).view(bs, sq, const.HERO_COUNT)

        pick_probs, ban_probs = self.filter(pick_probs, ban_probs, reserved)

        pick_dist = distributions.Categorical(pick_probs)
        ban_dist = distributions.Categorical(ban_probs)

        pick = pick_dist.sample()
        ban = ban_dist.sample()

        pick_logprobs = pick_dist.log_prob(pick)
        ban_logprobs = ban_dist.log_prob(ban)

        pick_entropy = pick_dist.entropy()
        ban_entropy = ban_dist.entropy()

        value = self.critic(common)

        return (pick, ban), (pick_logprobs, ban_logprobs), (pick_entropy, ban_entropy), value, next

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

    def __init__(self, hero_encoder=None):
        super(SimpleDrafter, self).__init__()

        if hero_encoder is None:
            hero_encoder = CategoryEncoder(const.HERO_COUNT, 128)

        self.hero_vec = hero_encoder.out_size
        self.hero_encoder = hero_encoder
        self.hidden_size = DraftFields.Size * self.hero_vec

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


# SELECT AVG(gold_per_min), MIN(gold_per_min), MAX(gold_per_min), stddev(gold_per_min)
# FROM
#     player_matches
# INNER JOIN match_patch USING(match_id)
# INNER JOIN matches USING(match_id)
# WHERE
#     matches.start_time >= 1608249726 AND
#     matches.start_time <= 1617981499 AND
#     match_patch.patch = '7.28'
#
class JudgeEstimatesNorm:
    DurationAvg = 2425  # 2425 = 40 min
    DurationStd = 586   # 10 minutes
    DurationMin = 375   # 6 minutes
    DurationMax = 8937  # 2h 30 minutes

    # 7.28 pro matches: 416.422	7	1302	172.9726
    # anything higher must be a crazy stomp
    GoldPerMinAvg = 435
    GoldPerMinStd = 154.88
    GoldPerMinMin = 96
    GoldPerMinMax = 1607

    # 7.28 pro matches: 16739.2685	0	178220	11413.4691
    # 100 0000 is similar to zeus ulting all the time
    HeroDamageAvg = 23344
    HeroDamageStd = 14983
    HeroDamageMin = 0
    HeroDamageMax = 178220

    # 7.28 pro matches: 2225.2344	0	31983	3867.2067
    # 8000 is probably close to max as well
    TowerDamageAvg = 2744
    TowerDamageStd = 3989
    TowerDamageMin = 0
    TowerDamageMax = 33082

    KillsAvg = 5.23
    KillsMin = 0
    KillsMax = 41
    KillsStd = 4.39

    DeathAvg = 5.32
    DeathMin = 0
    DeathMax = 25
    DeathStd = 3.47

    AssistsAvg = 11.27
    AssistsMin = 0
    AssistsMax = 45
    AssistsStd = 7

    XpPerMinAvg = 510.45
    XpPerMinMin = 15
    XpPerMinMax = 1153
    XpPerMinStd = 182.23


class JudgeEstimates(IntEnum):
    Duration = 0
    # Gold Per Minute include Last Hits
    GoldPerMin0 = 1
    GoldPerMin1 = 2
    GoldPerMin2 = 3
    GoldPerMin3 = 4
    GoldPerMin4 = 5
    GoldPerMin5 = 6
    GoldPerMin6 = 7
    GoldPerMin7 = 8
    GoldPerMin8 = 9
    GoldPerMin9 = 10
    # How much that hero do damage
    HeroDamage0 = 11
    HeroDamage1 = 12
    HeroDamage2 = 13
    HeroDamage3 = 14
    HeroDamage4 = 15
    HeroDamage5 = 16
    HeroDamage6 = 17
    HeroDamage7 = 18
    HeroDamage8 = 19
    HeroDamage9 = 20
    # How much that hero pushes
    TowerDamage0 = 21
    TowerDamage1 = 22
    TowerDamage2 = 23
    TowerDamage3 = 24
    TowerDamage4 = 25
    TowerDamage5 = 26
    TowerDamage6 = 27
    TowerDamage7 = 28
    TowerDamage8 = 29
    TowerDamage9 = 30
    Size = auto()


class DraftJudge(nn.Module):
    """Returns which faction is more likely to win the game given a draft

    Notes
    -----
    The judge also can train stats estimators, the goal of the estimators is to constraint the
    HeroEncoder further by forcing it to understand multiple dimensions

    Examples
    --------

    >>> judge = DraftJudge(None)
    >>> draft = torch.randn(24, 122)
    >>> batch = torch.stack([draft, draft])
    >>> result = judge(batch)
    >>> result.shape
    torch.Size([2])
    >>> result
    tensor([0.4986, 0.4986], grad_fn=<ViewBackward>)
    """
    def __init__(self, hero_encoder, compute_estimates=False):
        super(DraftJudge, self).__init__()

        if hero_encoder is None:
            hero_encoder = CategoryEncoder(const.HERO_COUNT, 128)

        self.hero_encoder = hero_encoder
        self.hero_vec = self.hero_encoder.out_size

        self.encoded_draft_size = DraftFields.Size * self.hero_vec

        self.common = nn.Sequential(
            nn.Linear(self.encoded_draft_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.rank_bias = SkillBias(256)

        self.win_layer = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax(dim=-1),
        )
        self.compute_estimates = compute_estimates

        # estimates
        self.estimates = nn.Sequential(
            nn.Linear(self.encoded_draft_size, JudgeEstimates.Size),
        )

    def forward(self, draft, rank=None):
        bs, ds, hc = draft.shape

        draft_flat = torch.flatten(draft, end_dim=1)
        encoded_flat = self.hero_encoder(draft_flat, rank)
        encoded_draft = encoded_flat.view(bs, ds * self.hero_vec)

        cm = self.common(encoded_draft)
        if rank is not None:
            rank = rank[:, 0, :]
            cm = cm + self.rank_bias(rank)

        estimates = None
        if self.compute_estimates:
            estimates = self.estimates(encoded_draft)

        return self.win_layer(cm), estimates
