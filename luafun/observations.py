from dataclasses import dataclass
from enum import IntEnum, auto

import torch

from luafun.utils.options import option
from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE


def enumitems(enumclass):
    return enumclass.__members__.items()


def print_tensor(tensor, enumclass, pfun=print):
    """Print a tensor using an enum as index names

    Examples
    --------
    >>> worldstate_tensor = torch.zeros((CommonState.Size,))
    >>> print_tensor(worldstate_tensor, CommonState)
                 GameTime: 0.0
                TimeOfDay: 0.0
               TimeToNext: 0.0
           TimeSpawnCreep: 0.0
         TimeSpawnNeutral: 0.0
          TimeSpawnBounty: 0.0
           TimeSpawnRunes: 0.0
    TimeSinceEnemyCourier: 0.0
             RoshSpawnMin: 0.0
             RoshSpawnMax: 0.0
                RoshAlive: 0.0
                 RoshDead: 0.0
               RoshCheese: 0.0
            RoshRefresher: 0.0
                 RoshAghs: 0.0
           RoshHealthRand: 0.0
            GlyphCooldown: 0.0
       GlyphCooldownEnemy: 0.0
                 StockGem: 0.0
               StockSmoke: 0.0
            StockObserver: 0.0
              StockSentry: 0.0
            StockRaindrop: 0.0

    >>> batch_size = 3
    >>> worldstate_batch = torch.zeros((batch_size, CommonState.Size))
    >>> print_tensor(worldstate_batch, CommonState)
                 GameTime: tensor([0., 0., 0.])
                TimeOfDay: tensor([0., 0., 0.])
               TimeToNext: tensor([0., 0., 0.])
           TimeSpawnCreep: tensor([0., 0., 0.])
         TimeSpawnNeutral: tensor([0., 0., 0.])
          TimeSpawnBounty: tensor([0., 0., 0.])
           TimeSpawnRunes: tensor([0., 0., 0.])
    TimeSinceEnemyCourier: tensor([0., 0., 0.])
             RoshSpawnMin: tensor([0., 0., 0.])
             RoshSpawnMax: tensor([0., 0., 0.])
                RoshAlive: tensor([0., 0., 0.])
                 RoshDead: tensor([0., 0., 0.])
               RoshCheese: tensor([0., 0., 0.])
            RoshRefresher: tensor([0., 0., 0.])
                 RoshAghs: tensor([0., 0., 0.])
           RoshHealthRand: tensor([0., 0., 0.])
            GlyphCooldown: tensor([0., 0., 0.])
       GlyphCooldownEnemy: tensor([0., 0., 0.])
                 StockGem: tensor([0., 0., 0.])
               StockSmoke: tensor([0., 0., 0.])
            StockObserver: tensor([0., 0., 0.])
              StockSentry: tensor([0., 0., 0.])
            StockRaindrop: tensor([0., 0., 0.])
    """
    if len(tensor.shape) > 1:
        print_batch_vector(tensor, enumclass, pfun=pfun)
    else:
        print_vector(tensor, enumclass, pfun=pfun)


def print_batch_vector(tensor, enumclass, pfun=print):
    for name, value in enumitems(enumclass):
        if name == 'Size':
            continue

        pfun(f'{name:>21}: {tensor[:, value]}')


def print_vector(tensor, enumclass, pfun=print):
    for name, value in enumitems(enumclass):
        if name == 'Size':
            continue

        pfun(f'{name:>21}: {tensor[value]}')


class CommonState(IntEnum):
    GameTime              = 0
    TimeOfDay             = auto()
    TimeToNext            = auto()
    TimeSpawnCreep        = auto()
    TimeSpawnNeutral      = auto()
    TimeSpawnBounty       = auto()
    TimeSpawnRunes        = auto()
    TimeSinceEnemyCourier = auto()
    RoshSpawnMin          = auto()
    RoshSpawnMax          = auto()
    RoshAlive             = auto()
    RoshDead              = auto()
    RoshCheese            = auto()
    RoshRefresher         = auto()
    RoshAghs              = auto()
    RoshHealthRand        = auto()
    GlyphCooldown         = auto()
    GlyphCooldownEnemy    = auto()
    StockGem              = auto()
    StockSmoke            = auto()
    StockObserver         = auto()
    StockSentry           = auto()
    StockRaindrop         = auto()
    Size                  = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, CommonState)


# OpenAI == 22
assert CommonState.Size == 23


class UnitState(IntEnum):
    X                        = 0
    Y                        = auto()
    Z                        = auto()
    FacingCos                = auto()
    FacingSin                = auto()
    IsAttacking              = auto()
    TimeSinceLastAttack      = auto()
    MaxHealth                = auto()
    Health00                 = auto()
    Health01                 = auto()
    Health02                 = auto()
    Health03                 = auto()
    Health04                 = auto()
    Health05                 = auto()
    Health06                 = auto()
    Health07                 = auto()
    Health08                 = auto()
    Health09                 = auto()
    Health10                 = auto()
    Health11                 = auto()
    Health12                 = auto()
    Health13                 = auto()
    Health14                 = auto()
    Health15                 = auto()
    AttackDamage             = auto()
    AttackSpeed              = auto()
    PhysicalResistance       = auto()
    IsGlyphed                = auto()
    GlyphTime                = auto()
    MovementSpeed            = auto()
    IsAlly                   = auto()
    IsNeutral                = auto()
    AnimationCycleTime       = auto()
    EtaIncomingProjectile    = auto()
    NumberOfUnitAttackingMe  = auto()
    # ShrineCooldown         = auto()
    ToCurrentUnitdx          = auto()
    ToCurrentUnitdy          = auto()
    ToCurrentUnitl           = auto()
    IsCurrentHeroAttackingIt = auto()
    IsAttackingCurrentHero   = auto()
    EtaProjectileToHero      = auto()
    UnitTypeHERO             = auto()
    UnitTypeCREEP_HERO       = auto()
    UnitTypeLANE_CREEP       = auto()
    UnitTypeJUNGLE_CREEP     = auto()
    UnitTypeROSHAN           = auto()
    UnitTypeTOWER            = auto()
    UnitTypeBARRACKS         = auto()
    UnitTypeSHRINE           = auto()
    UnitTypeFORT             = auto()
    UnitTypeBUILDING         = auto()
    UnitTypeCOURIER          = auto()
    UnitTypeWARD             = auto()
    Size                     = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, UnitState)


# OpenAI == 43
assert UnitState.Size == 53


class HeroUnit(IntEnum):
    IsAlive              = 0
    NumberOfDeath        = auto()
    HeroInSight          = auto()
    LastSeenTime         = auto()
    IsTeleporting        = auto()
    TeleportTargetX      = auto()
    TeleportTargetY      = auto()
    TeleportChannelTime  = auto()
    CurrentGold          = auto()
    Level                = auto()
    ManaMax              = auto()
    Mana                 = auto()
    ManaRegen            = auto()
    HealthMax            = auto()
    Health               = auto()
    HealthRegen          = auto()
    MagicResitance       = auto()
    Strength             = auto()
    Agility              = auto()
    Intelligence         = auto()
    IsInvisibility       = auto()
    IsUsingAbility       = auto()
    NumberOfAllied       = auto()
    NumberOfAlliedCreeps = auto()
    NumberOfEnemy        = auto()
    NumberOfEnemyCreeps  = auto()
    Size                 = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, HeroUnit)


# OpenAI == 25
assert HeroUnit.Size == 26


class PreviousActionState(IntEnum):
    Action      = 0
    LocationX   = auto()
    LocationY   = auto()
    LocationZ   = auto()
    AbilitySlot = auto()
    Item        = auto()
    Ix2         = auto()
    Size        = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, PreviousActionState)


# OpenAI == 310
assert PreviousActionState.Size == 7


class AllyHeroState(IntEnum):
    HasBuyBack             = 0
    BuyBackCost            = auto()
    BuyBackCooldown        = auto()
    NumberOfEmptyInventory = auto()
    NumberOfEmptyBackPack  = auto()
    LaneAssignRoam         = auto()
    LaneAssignTop          = auto()
    LaneAssignMid          = auto()
    LaneAssignBot          = auto()
    # It has terrain info 14x14 grid
    TerrainStart           = auto()
    TerrainEnd             = TerrainStart + 196
    Size                   = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, AllyHeroState)


# OpenAI == 211
assert AllyHeroState.Size == 206


# Merge this with Units/Heroes ?
class ModifierState(IntEnum):
    RemainingDuration      = 0
    StackCount             = auto()
    # ModifierEmbeddingStart = auto()
    # ModifierEmbeddingEnd   = ModifierEmbeddingStart + 128
    Size                   = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, ModifierState)


# OpenAI == 2
assert ModifierState.Size == 2


class ItemState(IntEnum):
    Inventory    = 0
    BackPack     = auto()
    Stash        = auto()
    Charges      = auto()
    IsCooldown   = auto()
    CooldownTime = auto()
    IsDisabled   = auto()
    SwapCoolDown = auto()
    ToggledState = auto()
    Locked       = auto()
    StateStr     = auto()
    StateAgi     = auto()
    StateInt     = auto()
    # ItemEmbeddingStart = auto()
    # ItemEmbeddingEnd   = ItemEmbeddingStart + 128
    Size                = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, ItemState)


# OpenAI == 13
assert ItemState.Size == 13


class AbilityState(IntEnum):
    Cooldown = 0
    InUse    = auto()
    Castable = auto()
    Level1   = auto()
    Level2   = auto()
    Level3   = auto()
    Level4   = auto()
    # ItemEmbeddingStart = auto()
    # ItemEmbeddingEnd   = ItemEmbeddingStart + 128
    Size               = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, AbilityState)


# OpenAI == 7
assert AbilityState.Size == 7


class RuneState(IntEnum):
    Visible    = 0
    LocationX  = auto()
    LocationY  = auto()
    DistanceH0 = auto()
    DistanceH1 = auto()
    DistanceH2 = auto()
    DistanceH3 = auto()
    DistanceH4 = auto()
    DistanceH5 = auto()
    DistanceH6 = auto()
    DistanceH7 = auto()
    DistanceH8 = auto()
    DistanceH9 = auto()
    Size       = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, RuneState)


# OpenAI == 15
assert RuneState.Size == 13


class Minimap10x10Tile(IntEnum):
    Visible = 0
    AlliedCreep = auto()
    EnemyCreep = auto()
    AlliedWard = auto()
    EnemyWard = auto()
    EnemyHero = auto()
    CellX = auto()
    CellY = auto()
    Size = auto()

    @staticmethod
    def print(tensor):
        print_tensor(tensor, Minimap10x10Tile)


class DropItem:
    X = 0
    Y = auto()
    Item = auto()


# OpenAI == 9
assert Minimap10x10Tile.Size == 8


class StateBuilder:
    def __init__(self):
        modifier_cnt = 10
        heroes_cnt = 10
        ally_cnt = 10
        unit_cnt = 10
        item_cnt = 17
        ability_cnt = 6
        rune_cnt = 6
        timesteps = 16

        # Hero state size
        self.total_size = (
            CommonState.Size +
            UnitState.Size * unit_cnt +
            HeroUnit.Size * heroes_cnt +
            AllyHeroState.Size * ally_cnt +
            PreviousActionState.Size * ally_cnt +
            ModifierState.Size * modifier_cnt * heroes_cnt +
            ItemState.Size * item_cnt * heroes_cnt +
            AbilityState.Size * ability_cnt * heroes_cnt +
            RuneState.Size * rune_cnt +
            Minimap10x10Tile.Size * 9
        )

        # OpenAI advertised ∼16,000 inputs
        #   we get 5,933
        # for reference an imagenet picture is about 150,528/196,608 inputs
        #
        # Output = 25 + 2 + 40 + 17 + 208 = 292

        # print('         ', self.total_size, self.total_size * 16)
        # use imagenet as a point of reference
        assert self.total_size * timesteps < 3 * 256 * 256

    def update_rune(self):
        pass


def generate_game_batch(state, player_ids):
    size = StateBuilder().total_size
    batch = torch.zeros((len(player_ids), size))
    cache = {
        TEAM_RADIANT: dict(),
        TEAM_DIRE: dict()
    }

    radiant, dire = state

    for b, pid in enumerate(player_ids):
        faction = TEAM_RADIANT
        faction_state = radiant
        if pid > 4:
            faction = TEAM_DIRE
            faction_state = dire

        generate_player(pid, faction_state, batch[b, :], cache[faction])

    return batch


class FullState(IntEnum):
    WorldStateS = 0
    WorldStateE = len(CommonState)
    MyHeroS = auto()
    MyHeroE = MyHeroS + len(AllyHeroState)
    Ally1S = auto()
    Ally1E = Ally1S + len(AllyHeroState)
    Ally2S = auto()
    Ally2E = Ally2S + len(AllyHeroState)
    Ally3S = auto()
    Ally3E = Ally3S + len(AllyHeroState)
    Ally4S = auto()
    Ally4E = Ally4S + len(AllyHeroState)
    Enemy0S = auto()
    Enemy0E = Enemy0S + len(HeroUnit)
    Enemy1S = auto()
    Enemy1E = Enemy1S + len(HeroUnit)
    Enemy2S = auto()
    Enemy2E = Enemy2S + len(HeroUnit)
    Enemy3S = auto()
    Enemy3E = Enemy3S + len(HeroUnit)
    Enemy4S = auto()
    Enemy4E = Enemy4S + len(HeroUnit)


def generate_player(pid, state, tensor, cache):
    myhero = state._players[pid]


    #


print(f"ImageNet size {3 * 256 * 256}")
StateBuilder()


class ObservationSampler:
    """
    Parameters
    ----------
    timestep: int
        Number of observation needed for our model when sampling
    """
    def __init__(self, dataset, timestep):
        pass


@dataclass
class Entry:
    action: float
    state: None
    logprob: float
    reward: float
    done: bool
    newstate: None


class GameDataset:
    """Store all the states of a given game and allow states to be sampled for a given timestep"""
    def __init__(self, timestep=option('timestep', 16)):
        self._actions = []
        self._states = []
        self._logprobs = []
        self._rewards = []
        self._dones = []
        self._newstate = []
        self.timestep = timestep

    def rewards(self, item, timestep, device, gamma):
        rewards = []
        discounted_reward = 0

        # TODO: we should discount by over all time
        for reward, done in zip(reversed(self._rewards), reversed(self._dones)):
            if done:
                discounted_reward = 0

            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return rewards
        
    def __len__(self):
        return len(self._states) - self.timestep

    def states(self, item, timestep, device):
        return self._states[item - self.timestep:item]

    def actions(self, item, timestep, device):
        return self._actions[item - self.timestep:item]

    def logprobs(self, item, timestep, device):
        return self._logprobs[item - self.timestep:item]

    def __getitem__(self, item) -> Entry:
        return Entry(
            self._actions[item],
            self._states[item],
            self._logprobs[item],
            self._rewards[item],
            self._dones[item],
            self._newstate[item])

    def append(self, action, state, logprob, reward, newstate, done):
        self._actions.append(action)
        self._states.append(state)
        self._logprobs.append(logprob)
        self._rewards.append(reward)
        self._dones.append(done)
        self._newstate.append(newstate)
