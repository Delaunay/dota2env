from dataclasses import dataclass
from enum import IntEnum, auto

from luafun.utils.options import option


class WorldState(IntEnum):
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


# OpenAI == 22
assert WorldState.Size == 23


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
    UnitTypeINVALID          = auto()
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


# OpenAI == 43
assert UnitState.Size == 54


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


# OpenAI == 211
assert AllyHeroState.Size == 206


# Merge this with Units/Heroes ?
class ModifierState(IntEnum):
    RemainingDuration      = 0
    StackCount             = auto()
    # ModifierEmbeddingStart = auto()
    # ModifierEmbeddingEnd   = ModifierEmbeddingStart + 128
    Size                   = auto()


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
            WorldState.Size +
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

        print('         ', self.total_size, self.total_size * 16)
        # use imagenet as a point of reference
        assert self.total_size * timesteps < 3 * 256 * 256

    def update_rune(self):
        pass


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
