import torch
import torch.nn as nn

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
        self.out_size = out_size
        self.linear: nn.Linear = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x):
        return self.linear(x)


class SkillBias(nn.Module):
    def __init__(self, vector):
        super(SkillBias, self).__init__()
        self.skill_bias = CategoryEncoder(const.Rank.Size, vector)
        nn.init.constant_(self.skill_bias.linear.weight, 0)

    def forward(self, rank):
        return self.skill_bias(rank)


class HeroEmbedding(nn.Module):
    """We want to be able to extract as much information as possible from the data we have
    as such we need to be able to learn from low skilled game as well.
    To capture the noise coming from low skill bias network are added

    Examples
    --------
    >>> import luafun.game.constants as const

    Make a single hero
    >>> hero = torch.zeros((const.HERO_COUNT,))
    >>> hero[const.HERO_LOOKUP.from_name('npc_dota_hero_techies')['offset']] = 1

    Make a batch of heroes
    >>> heroes = torch.stack([hero, hero])

    Make a rank
    >>> rank = torch.zeros((const.Rank.Size,))
    >>> rank[const.Rank.Ancient0] = 1

    Make batch of ranks
    >>> ranks = torch.stack([rank, rank])

    >>> hero_encoder = HeroEmbedding()
    >>> hero_encoder(heroes, ranks).shape
    torch.Size([2, 128])
    """
    def __init__(self, vector=128):
        super(HeroEmbedding, self).__init__()
        self.out_size = vector
        self.baseline = CategoryEncoder(const.HERO_COUNT, vector)

        # lower MMR are just bad at some things
        self.skill_bias = SkillBias(vector)

        # lower MMR dont understand the hero
        # self.hero_bias = CategoryEncoder(const.HERO_COUNT * const.Rank.Size, vector)
        self.hero_bias = MultiCategoryEncoderFlat([i.value for i in const.Rank], const.HERO_COUNT, vector)

    def forward(self, heroes, rank=None):
        bs = heroes.shape[0]
        base = self.baseline(heroes)

        if rank is None:
            return base

        rank = torch.flatten(rank, end_dim=1)
        skill_bias = self.skill_bias(rank)

        _, rank_id = torch.max(rank, 1)
        hero_bias = self.hero_bias(heroes, rank_id)

        return base + skill_bias + hero_bias


class CategoryEncoderDict(nn.Module):
    """

    Example
    -------
    >>> encoder = CategoryEncoderDict(["bloodseeker", "arc_warden"], (6,))
    >>> encoder(["bloodseeker", "bloodseeker"]).shape
    torch.Size([2, 6])
    """
    def __init__(self, keys, shape):
        super(CategoryEncoderDict, self).__init__()

        self.encoders = {
            str(k): nn.Parameter(torch.ones(shape)) for k in keys
        }

    def parameters(self, recurse: bool = True):
        for _, v in self.encoders.items():
            yield v

    def forward(self, x):
        return torch.stack([self.encoders[k] for k in x])


class MultiCategoryEncoderFlat(nn.Module):
    """
    Examples
    --------
    >>> obs = torch.randn((10,))
    >>> batch = torch.stack([obs, obs])

    >>> rank = torch.stack([torch.tensor(1), torch.tensor(2)])

    >>> encoder = MultiCategoryEncoder([1, 2], 10, 5)
    >>> encoder(batch, rank).shape
    torch.Size([2, 5])
    """
    def __init__(self, classes, n_category, out_vector):
        super(MultiCategoryEncoderFlat, self).__init__()
        self.out_vector = out_vector

        self.n_class = len(classes)
        self.n_category = n_category
        self.total = self.n_class * self.n_category

        self.encoder = CategoryEncoder(self.total, out_vector)
        nn.init.constant_(self.encoder.linear.weight, 0)

    def forward(self, batch, classes):
        bs = batch.shape[0]
        result = torch.zeros((bs, self.total)).cuda()

        for i, (c, obs) in enumerate(zip(classes, batch)):
            s = self.n_class * c
            e = s + self.n_category
            result[i, s:e] = obs

        return self.encoder(result)


class MultiCategoryEncoder(nn.Module):
    """
    Examples
    --------
    >>> obs = torch.randn((10,))
    >>> batch = torch.stack([obs, obs])

    >>> rank = torch.stack([torch.tensor(1), torch.tensor(2)])

    >>> encoder = MultiCategoryEncoder([1, 2], 10, 5)
    >>> encoder(batch, rank).shape
    torch.Size([2, 5])
    """
    def __init__(self, classes, n_category, out_vector):
        super(MultiCategoryEncoder, self).__init__()
        self.out_vector = out_vector
        self.encoders = nn.ModuleDict({
            str(k): CategoryEncoder(n_category, out_vector) for k in classes
        })

        for _, v in self.encoders.items():
            nn.init.constant_(v.linear.weight, 0)

    def forward(self, batch, classes):
        bs = batch.shape[0]
        result = torch.zeros((bs, self.out_vector)).cuda()

        for i, (c, obs) in enumerate(zip(classes, batch)):
            result[i, :] = self.encoders[str(c.item())](obs)

        return result


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
