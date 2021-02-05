"""This module encodes the game action into a ML friendly format"""
from enum import IntEnum
from typing import Tuple
from luafun.game.ipc_send import new_ipc_message, TEAM_RADIANT, TEAM_DIRE


class AbilitySlot(IntEnum):
    """List all the abilities available to a given hero"""
    # Inventory
    Item0 = 0
    Item1 = 1
    Item2 = 2
    Item3 = 3
    Item4 = 4
    Item5 = 5
    Bakcpack1 = 6
    Bakcpack2 = 7
    Bakcpack3 = 8
    Stash1 = 9
    Stash2 = 10
    Stash3 = 11
    Stash4 = 12
    Stash5 = 13
    Stash6 = 14
    Item15 = 15         # TP
    Item16 = 16         # Neutral ?
    Q = 17              # Q                 | invoker_quas
    W = 18              # W                 | invoker_wex
    E = 19              # E                 | invoker_exort
    D = 20              # D generic_hidden  | invoker_empty1
    F = 21              # F generic_hidden  | invoker_empty2
    R = 22              # R                 | invoker_invoke
    Ablity6 = 23        # .                 | invoker_cold_snap
    Ablity7 = 24        # .                 | invoker_ghost_walk
    Ablity8 = 25        # .                 | invoker_tornado
    Ablity9 = 26        # .                 | invoker_emp
    Ablity10 = 27       # .                 | invoker_alacrity
    Ablity11 = 28       # .                 | invoker_chaos_meteor
    Ablity12 = 29       # .                 | invoker_sun_strike
    Ablity13 = 30       # .                 | invoker_forge_spirit
    Ablity14 = 31       # .                 | invoker_ice_wall
    Ablity15 = 32       # .                 | invoker_deafening_blast
    Talent11 = 33       # Talent 1  (usually but the talent offset can be shifted)
    Talent12 = 34       # Talent 2  example: rubick, invoker, etc..
    Talent21 = 35       # Talent 3
    Talent22 = 36       # Talent 4  98 heroes follow the pattern above
    Talent31 = 37       # Talent 5
    Talent32 = 38       # Talent 6
    Talent41 = 39       # Talent 7
    Talent42 = 40       # Talent 8


assert len(AbilitySlot) == 41, '41 abilities'


# When looking at Action you might think that dota is not that complex
# nevertheless you need to take into account that when calling UseAbility
# you have to choose among ~1000 unique abilities (120 heroes * 4 + 155 items)
# the abilities are context depend each heroes can have
#   ~4 ability + tp ability
#   ~6 Items + neutral item
#
# NB: To take outpost, you can attack them using AttackUnit action
class Action(IntEnum):
    """List all the actions available to a hero"""
    Stop                          = 0
    MoveToLocation                = 1   # ( vLocation )
    MoveDirectly                  = 2   # ( vLocation )
    MoveToUnit                    = 3   # ( hUnit )
    AttackUnit                    = 4   # ( hUnit, bOnce = True )
    AttackMove                    = 5   # ( vLocation )
    UseAbility                    = 6   # ( hAbility )
    UseAbilityOnEntity            = 7   # ( hAbility, hTarget )
    UseAbilityOnLocation          = 8   # ( hAbility, vLocation )
    UseAbilityOnTree              = 9   # ( hAbility, iTree )
    PickUpRune                    = 10  # ( nRune )
    PickUpItem                    = 11  # ( hItem )
    DropItem                      = 12  # ( hItem, vLocation )
    PurchaseItem                  = 13  # ( sItemName )
    SellItem                      = 14  # ( hItem )
    DisassembleItem               = 15  # ( hItem )
    SetItemCombineLock            = 16  # ( hItem, bLocked )
    # The index order does not matter
    # bots automatically transfer items from stash to inventory when possible
    SwapItems                     = 17  # ( index1, index2 )
    Buyback                       = 18  # ()
    Glyph                         = 19  # ()
    LevelAbility                  = 20  # ( sAbilityName )

    # Courier Action bundled to the hero
    CourierBurst                  = 21
    # hidden ability; cannot use
    # CourierEnemySecret            = 23
    CourierReturn                 = 22
    CourierSecret                 = 23
    CourierTakeStash              = 24
    CourierTransfer               = 25

    # Tensor cores work better with a multiple of 8
    # This gives us room to grow
    NotUsed1 = 26
    NotUsed2 = 27
    NotUsed3 = 28
    NotUsed4 = 29
    NotUsed5 = 30
    NotUsed6 = 31

    # The action exist but it is not necessary
    # Courier                         # ( hCourier, nAction )
    # UseShrine                       # ( hShrine )
    # MovePath                        # ( tWaypoints )


assert len(Action) == 32, '32 actions'


class DraftAction(IntEnum):
    EnableDraft = 0
    SelectHero = 1
    BanHero = 2
    Lane = 3


# Argument index
class ActionArgument(IntEnum):
    action   = 0
    vLoc     = 1
    hUnit    = 2    # this should be handle
    nSlot    = 3    # Slot (item or ability)
    iTree    = 4    # This is problematic we have 2000+ trees
    nRune    = 5    # This could be bundled as an enum like inventory slots
    sItem    = 6    # Needed to buy item
    ix2      = 7


ARG = ActionArgument
# 2":{"0":11,"2":355}


# boilerplate to help humans send bot like action to lua
# this only to debug & allows human to control the lua bots from python
class PlayerAction:
    """Player action builder"""
    def __init__(self, act: dict):
        self.act = act

    def MoveToLocation(self, vLocation: Tuple[float, float]):
        self.act[ARG.action] = Action.MoveToLocation
        self.act[ARG.vLoc] = vLocation

    def MoveDirectly(self, vLocation: Tuple[float, float]):
        self.act[ARG.action] = Action.MoveDirectly
        self.act[ARG.vLoc] = vLocation

    def MoveToUnit(self, hUnit: int):
        self.act[ARG.action] = Action.MoveToUnit
        self.act[ARG.hUnit] = hUnit

    def AttackUnit(self, hUnit: int):
        self.act[ARG.action] = Action.AttackUnit
        self.act[ARG.hUnit] = hUnit

    def AttackMove(self, vLocation: Tuple[float, float]):
        self.act[ARG.action] = Action.AttackMove
        self.act[ARG.vLoc] = vLocation

    def UseAbility(self, hAbility: int):
        self.act[ARG.action] = Action.UseAbility
        self.act[ARG.nSlot] = hAbility

    def UseAbilityOnEntity(self, hAbility: int, hTarget: int):
        self.act[ARG.action] = Action.UseAbilityOnEntity
        self.act[ARG.nSlot] = hAbility
        self.act[ARG.hUnit] = hTarget

    def UseAbilityOnLocation(self, hAbility: int, vLoc: Tuple[float, float]):
        self.act[ARG.action] = Action.UseAbilityOnLocation
        self.act[ARG.nSlot] = hAbility
        self.act[ARG.vLoc] = vLoc

    def UseAbilityOnTree(self, hAbility: int, iTree: int):
        self.act[ARG.action] = Action.UseAbilityOnTree
        self.act[ARG.nSlot] = hAbility
        self.act[ARG.iTree] = iTree

    def PickUpRune(self, nRune: int):
        self.act[ARG.action] = Action.PickUpRune
        self.act[ARG.nRune] = nRune

    def PickUpItem(self, hItem: int):
        self.act[ARG.action] = Action.PickUpItem
        self.act[ARG.hUnit] = hItem

    def DropItem(self, hItem: int, vLocation: Tuple[float, float]):
        self.act[ARG.action] = Action.DropItem
        self.act[ARG.vLoc] = vLocation
        self.act[ARG.nSlot] = hItem

    def PurchaseItem(self, sItemName: str):
        self.act[ARG.action] = Action.PurchaseItem
        self.act[ARG.sItem] = sItemName

    def SellItem(self, hItem: int):
        self.act[ARG.action] = Action.SellItem
        self.act[ARG.nSlot] = hItem

    def DisassembleItem(self, hItem: int):
        self.act[ARG.action] = Action.DisassembleItem
        self.act[ARG.nSlot] = hItem

    def SetItemCombineLock(self, hItem):
        self.act[ARG.action] = Action.SetItemCombineLock
        self.act[ARG.nSlot] = hItem

    def SwapItems(self, nslot: int, index2: int):
        # The index order does not matter
        self.act[ARG.action] = Action.SwapItems
        self.act[ARG.nSlot] = nslot
        self.act[ARG.ix2] = index2

    def Buyback(self):
        self.act[ARG.action] = Action.Buyback

    def Stop(self):
        self.act[ARG.action] = Action.Stop

    def Glyph(self):
        self.act[ARG.action] = Action.Glyph

    def LevelAbility(self, nSlot: int):
        self.act[ARG.action] = Action.LevelAbility
        self.act[ARG.nSlot] = nSlot

    def CourierBurst(self):
        self.act[ARG.action] = Action.CourierBurst

    # Ability is hidden
    # def CourierEnemySecret(self):
    #    self.act[ARG.action] = Action.CourierEnemySecret

    def CourierReturn(self):
        self.act[ARG.action] = Action.CourierReturn

    def CourierSecret(self):
        self.act[ARG.action] = Action.CourierSecret

    def CourierTakeStash(self):
        self.act[ARG.action] = Action.CourierTakeStash

    def CourierTransfer(self):
        self.act[ARG.action] = Action.CourierTransfer


class DraftBuilder:
    """Drafting action builder"""
    def __init__(self, fac: dict):
        self.fac = fac

    def select(self, hero: int, lane: int):
        """Select a hero and assign it to a particular lane"""
        self.fac[DraftAction.SelectHero] = hero
        self.fac[DraftAction.Lane] = lane

    def ban(self, hero: int):
        """Ban does not work for bots, which makes sense in the case of players wanting to practice annoying hero"""
        self.fac[DraftAction.BanHero] = hero


class IPCMessageBuilder:
    """Helper to help users build action using code. Mainly used for testing purposes"""
    def __init__(self, game=None):
        self.message = new_ipc_message()
        self.game = game

    def player(self, idx: int) -> PlayerAction:
        """Prepare to send an action to a given hero"""
        faction = TEAM_RADIANT

        if idx > 4:
            faction = TEAM_DIRE

        return PlayerAction(self.message[faction][idx])

    def hero_selection(self, faction: int) -> DraftBuilder:
        """Prepare to draft a hero for a given faction"""
        self.message[faction]['HS'] = {
            DraftAction.EnableDraft: 1,
            DraftAction.SelectHero: None,
            DraftAction.BanHero: None,
            DraftAction.Lane: None
        }
        return DraftBuilder(self.message[faction]['HS'])

    def build(self) -> dict:
        """Returns the resulting action message"""
        return self.message

    def send(self):
        """If game was set, send the message to the game"""
        if self.game is not None:
            return self.game.send_message(self.build())


def player_space():
    """Returns the full action space of a Dota2 bot

    Examples
    --------
    >>> s = player_space()
    >>> s.seed(0)
    >>> for k, v in s.sample().items():
    ...     print(k, v)
    ActionArgument.action 16
    ActionArgument.vLoc [-0.8912799  0.9307819]
    ActionArgument.hUnit 112
    ActionArgument.nSlot 14
    ActionArgument.iTree 1083
    ActionArgument.nRune 0
    ActionArgument.sItem 112
    ActionArgument.ix2 16

    """
    from gym import spaces
    import numpy as np
    import luafun.game.constants as const

    action = spaces.Discrete(len(Action))
    vloc = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    # We set the max number of unit on the map to 256
    # the ids are remapped to actual handle id
    hUnit = spaces.Discrete(256)
    abilities = spaces.Discrete(len(AbilitySlot))
    # Tree ID
    tree = spaces.Discrete(const.TREE_COUNT)
    runes = spaces.Discrete(len(const.RuneSlot))
    items = spaces.Discrete(const.ITEM_COUNT)
    ix2 = spaces.Discrete(len(const.ItemSlot))

    return spaces.Dict({
        ARG.action: action,
        ARG.vLoc: vloc,
        ARG.hUnit: hUnit,
        ARG.nSlot: abilities,
        ARG.iTree: tree,
        ARG.nRune: runes,
        ARG.sItem: items,
        ARG.ix2: ix2
    })


def team_space(s: int):
    """Returns the full action space of a Dota2 bot team

    Examples
    --------
    >>> s = team_space(0)
     >>> s.seed(0)
    >>> for k, v in s.sample().items():
    ...     print(k, v)
    0 OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])
    1 OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])
    2 OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])
    3 OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])
    4 OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])
    HS OrderedDict([('ban', 112), ('lane', 0), ('select', 112)])
    """
    from gym import spaces
    import luafun.game.constants as const

    return spaces.Dict({
        f'{s + 0}': player_space(),
        f'{s + 1}': player_space(),
        f'{s + 2}': player_space(),
        f'{s + 3}': player_space(),
        f'{s + 4}': player_space(),

        # Hero Selection
        'HS': spaces.Dict({
            'select': spaces.Discrete(const.HERO_COUNT),
            'ban': spaces.Discrete(const.HERO_COUNT),
            'lane': spaces.Discrete(len(const.Lanes))
        })
    })


def action_space():
    """Returns the full action space of a Dota2 bots for all teams

    Examples
    --------
    >>> s = action_space()
    >>> s.seed(0)
    >>> for k, v in s.sample().items():
    ...     print(k, v)
    2 OrderedDict([('0', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('1', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('2', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('3', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('4', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('HS', OrderedDict([('ban', 112), ('lane', 0), ('select', 112)]))])
    3 OrderedDict([('5', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('6', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('7', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('8', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('9', OrderedDict([(<ActionArgument.action: 0>, 16), (<ActionArgument.vLoc: 1>, array([-0.8912799,  0.9307819], dtype=float32)), (<ActionArgument.hUnit: 2>, 112), (<ActionArgument.nSlot: 3>, 14), (<ActionArgument.iTree: 4>, 1083), (<ActionArgument.nRune: 5>, 0), (<ActionArgument.sItem: 6>, 112), (<ActionArgument.ix2: 7>, 16)])), ('HS', OrderedDict([('ban', 112), ('lane', 0), ('select', 112)]))])
    """
    from gym import spaces

    full_space = spaces.Dict({
        TEAM_RADIANT: team_space(0),
        TEAM_DIRE: team_space(5),
    })

    return full_space
