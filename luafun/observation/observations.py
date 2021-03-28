from dataclasses import dataclass
from enum import IntEnum, auto

import torch

from luafun.utils.options import option
from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE


def enumitems(enumclass):
    return enumclass.__members__.items()


def print_tensor(tensor, enumclass, pfun=print, i=None):
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
        print_batch_vector(tensor, enumclass, pfun=pfun, i=i)
    else:
        print_vector(tensor, enumclass, pfun=pfun, i=i)


def print_batch_vector(tensor, enumclass, pfun=print, i=None):
    for name, value in enumitems(enumclass):
        if name == 'Size':
            continue

        pfun(f'{name:>21}: {tensor[:, value]}')


def print_vector(tensor, enumclass, pfun=print, i=None):
    for name, value in enumitems(enumclass):
        if name == 'Size':
            continue

        msg = f'{enumclass.__name__:>15} {name:>25}: {tensor[value]}'
        if i is not None:
            msg = f'{i:>3d}{msg}'
        else:
            msg = f'   {msg}'

        pfun(msg)


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
    def print(tensor, i=None):
        print_tensor(tensor, CommonState, i=i)


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
    # IsGlyphed                = auto()
    # GlyphTime                = auto()
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
    def print(tensor, i=None):
        print_tensor(tensor, UnitState, i=i)


# OpenAI == 43
assert UnitState.Size == 51


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
    def print(tensor, i=None):
        print_tensor(tensor, HeroUnit, i=i)


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
    def print(tensor, i=None):
        print_tensor(tensor, PreviousActionState, i=i)


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
    def print(tensor, i=None):
        print_tensor(tensor, AllyHeroState, i=i)


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
    def print(tensor, i=None):
        print_tensor(tensor, ModifierState, i=i)


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
    def print(tensor, i=None):
        print_tensor(tensor, ItemState, i=i)


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
    def print(tensor, i=None):
        print_tensor(tensor, AbilityState, i=i)


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
    def print(tensor, i=None):
        print_tensor(tensor, RuneState, i=i)


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
    def print(tensor, i=None):
        print_tensor(tensor, Minimap10x10Tile, i=i)


class DropItem:
    X = 0
    Y = auto()
    Item = auto()


# OpenAI == 9
assert Minimap10x10Tile.Size == 8
