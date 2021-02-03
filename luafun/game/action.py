"""This file works on encoding the game action into a ML friendly format"""
from enum import IntEnum
from luafun.game.ipc_send import new_ipc_message, TEAM_RADIANT, TEAM_DIRE


# Game mapping
class Lanes(IntEnum):
    Roam = 0
    Top = 1
    Mid = 2
    Bot = 3


class RuneSlot(IntEnum):
    PowerUpTop = 0
    PowerUpBottom = 1
    BountyRiverTop = 2
    BountyRadiant = 3
    BountyRiverBottom = 4
    BountyDire = 5


# indices are zero based with 0-5 corresponding to inventory, 6-8 are backpack and 9-15 are stash
class ItemSlot(IntEnum):
    # Inventory
    Item0 = 0
    Item1 = 1
    Item2 = 2
    Item3 = 3
    Item4 = 4
    Item5 = 5
    # backpack
    Item6 = 6
    Item7 = 7
    Item8 = 8
    # Stash
    Item9 = 9
    Item10 = 10
    Item11 = 11
    Item12 = 12
    Item13 = 13
    Item14 = 14
    Item15 = 15     # TP
    Item16 = 16     # Neutral ?


assert len(ItemSlot) == 17, '17 item slots'


_n = len(ItemSlot)


class SpellSlot(IntEnum):
    Q = _n + 0
    W = _n + 1
    E = _n + 2
    D = _n + 3
    F = _n + 4
    R = _n + 5


# might have to normalize talent so it is easier to learn
class AbilitySlot(IntEnum):
    Ablity0 = 0         # Q
    Ablity1 = 1         # W
    Ablity2 = 2         # E
    Ablity3 = 3         # D generic_hidden
    Ablity4 = 4         # F generic_hidden
    Ablity5 = 5         # R                 This is standard
    Ablity6 = 6
    Ablity7 = 7
    Ablity8 = 8
    Ablity9 = 9         # Talent 1  (usually but the talent offset can be shifted)
    Ablity10 = 10       # Talent 2  example: rubick, invoker, etc..
    Ablity11 = 11       # Talent 3
    Ablity12 = 12       # Talent 4  98 heroes follow the pattern above
    Ablity13 = 13       # Talent 5
    Ablity14 = 14       # Talent 6
    Ablity15 = 15       # Talent 7
    Ablity16 = 16       # Talent 8
    Ablity17 = 17
    Ablity18 = 18
    Ablity19 = 19
    Ablity20 = 20
    Ablity21 = 21
    Ablity22 = 22
    Ablity23 = 23


assert len(AbilitySlot) == 24, '24 abilities'


# When looking at Action you might think that dota is not that complex
# nevertheless you need to take into account that when calling UseAbility
# you have to choose among ~1000 unique abilities (120 heroes * 4 + 155 items)
# the abilities are context depend each heroes can have
#   ~4 ability + tp ability
#   ~6 Items + neutral item
#
# NB: To take outpost, you can attack them using AttackUnit action
class Action(IntEnum):
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


# Could bundle the courier action as a hero action
class CourierAction(IntEnum):
    BURST               = 0
    # hidden
    # ENEMY_SECRET_SHOP   = 1
    RETURN              = 2
    SECRET_SHOP         = 3
    TAKE_STASH_ITEMS    = 4
    TRANSFER_ITEMS      = 5
    # bots will have to do 2 actions for those
    # not a big deal IMO
    # TAKE_AND_TRANSFER_ITEMS
    # COURIER_ACTION_SIDE_SHOP
    # COURIER_ACTION_SIDE_SHOP2


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
class Player:
    def __init__(self, act):
        self.act = act

    def MoveToLocation(self, vLocation):
        self.act[ARG.action] = Action.MoveToLocation
        self.act[ARG.vLoc] = vLocation

    def MoveDirectly(self, vLocation):
        self.act[ARG.action] = Action.MoveDirectly
        self.act[ARG.vLoc] = vLocation

    def MoveToUnit(self, hUnit):
        self.act[ARG.action] = Action.MoveToUnit
        self.act[ARG.hUnit] = hUnit

    def AttackUnit(self, hUnit):
        self.act[ARG.action] = Action.AttackUnit
        self.act[ARG.hUnit] = hUnit

    def AttackMove(self, vLocation):
        self.act[ARG.action] = Action.AttackMove
        self.act[ARG.vLoc] = vLocation

    def UseAbility(self, hAbility):
        self.act[ARG.action] = Action.UseAbility
        self.act[ARG.nSlot] = hAbility

    def UseAbilityOnEntity(self, hAbility, hTarget):
        self.act[ARG.action] = Action.UseAbilityOnEntity
        self.act[ARG.nSlot] = hAbility
        self.act[ARG.hUnit] = hTarget

    def UseAbilityOnLocation(self, hAbility, vLoc):
        self.act[ARG.action] = Action.UseAbilityOnLocation
        self.act[ARG.nSlot] = hAbility
        self.act[ARG.vLoc] = vLoc

    def UseAbilityOnTree(self, hAbility, iTree):
        self.act[ARG.action] = Action.UseAbilityOnTree
        self.act[ARG.nSlot] = hAbility
        self.act[ARG.iTree] = iTree

    def PickUpRune(self, nRune):
        self.act[ARG.action] = Action.PickUpRune
        self.act[ARG.nRune] = nRune

    def PickUpItem(self, hItem):
        self.act[ARG.action] = Action.PickUpItem
        self.act[ARG.hUnit] = hItem

    def DropItem(self, hItem, vLocation):
        self.act[ARG.action] = Action.DropItem
        self.act[ARG.vLoc] = vLocation
        self.act[ARG.nSlot] = hItem

    def PurchaseItem(self, sItemName):
        self.act[ARG.action] = Action.PurchaseItem
        self.act[ARG.sItem] = sItemName

    def SellItem(self, hItem):
        self.act[ARG.action] = Action.SellItem
        self.act[ARG.nSlot] = hItem

    def DisassembleItem(self, hItem):
        self.act[ARG.action] = Action.DisassembleItem
        self.act[ARG.nSlot] = hItem

    def SetItemCombineLock(self, hItem):
        self.act[ARG.action] = Action.SetItemCombineLock
        self.act[ARG.nSlot] = hItem

    def SwapItems(self, nslot, index2):
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

    def LevelAbility(self, nSlot):
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


def action_space():
    from gym import spaces
    import numpy as np
    import luafun.game.constants as const

    def player_space():
        action = spaces.Discrete(len(Action))
        vloc = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # We set the max number of unit on the map to 256
        # the ids are remapped to actual handle id
        hUnit = spaces.Discrete(256)
        abilities = spaces.Discrete(len(ItemSlot) + len(AbilitySlot))
        # Tree ID
        tree = spaces.Discrete(const.TREE_COUNT)
        runes = spaces.Discrete(len(RuneSlot))
        items = spaces.Discrete(const.ITEM_COUNT)
        ix2 = spaces.Discrete(len(ItemSlot))

        return spaces.Tuple((
            action,
            vloc,
            hUnit,
            abilities,
            tree,
            runes,
            items,
            ix2))

    def team_space(s):
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
                'lane': spaces.Discrete(len(Lanes))
            })
        })

    full_space = spaces.Dict({
        f'{TEAM_RADIANT}': team_space(0),
        f'{TEAM_DIRE}': team_space(5),
    })
    return full_space


class DraftAction(IntEnum):
    EnableDraft = 0
    SelectHero = 1
    BanHero = 2
    Lane = 3


class HeroSelection:
    def __init__(self, fac):
        self.fac = fac

    def select(self, hero, lane):
        self.fac[DraftAction.SelectHero] = hero
        self.fac[DraftAction.Lane] = lane

    def ban(self, hero):
        self.fac[DraftAction.BanHero] = hero


class IPCMessageBuilder:
    def __init__(self, game=None):
        self.message = new_ipc_message()
        self.game = game

    def player(self, idx):
        faction = TEAM_RADIANT

        if idx > 4:
            faction = TEAM_DIRE

        return Player(self.message[faction][idx])

    def hero_selection(self, faction):
        self.message[faction]['HS'] = {
            DraftAction.EnableDraft: 1,
            DraftAction.SelectHero: None,
            DraftAction.BanHero: None,
            DraftAction.Lane: None
        }
        return HeroSelection(self.message[faction]['HS'])

    def build(self):
        return self.message

    def send(self):
        if self.game is not None:
            return self.game.send_message(self.build())
