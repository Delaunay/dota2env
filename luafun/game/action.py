from enum import IntEnum
from luafun.game.ipc_send import new_ipc_message, TEAM_RADIANT, TEAM_DIRE

# Action Type
    # BOT_ACTION_TYPE_NONE
    # BOT_ACTION_TYPE_IDLE
    # BOT_ACTION_TYPE_MOVE_TO
    # BOT_ACTION_TYPE_MOVE_TO_DIRECTLY
    # BOT_ACTION_TYPE_ATTACK
    # BOT_ACTION_TYPE_ATTACKMOVE
    # BOT_ACTION_TYPE_USE_ABILITY
    # BOT_ACTION_TYPE_PICK_UP_RUNE
    # BOT_ACTION_TYPE_PICK_UP_ITEM
    # BOT_ACTION_TYPE_DROP_ITEM
    # BOT_ACTION_TYPE_SHRINE
    # BOT_ACTION_TYPE_DELAY

# When looking at Action you might think that dota is not that complex
# nevertheless you need to take into account that when calling UseAbility
# you have to choose among ~1000 unique abilities (120 heroes * 4 + 155 items)
# the abilities are context depend each heroes can have
#   ~4 ability + tp ability
#   ~6 Items + neutral item
class Action(IntEnum):
    MoveToLocation                = 0   # ( vLocation )
    MoveDirectly                  = 1   # ( vLocation )
    MoveToUnit                    = 2   # ( hUnit )
    AttackUnit                    = 3   # ( hUnit, bOnce = True )
    AttackMove                    = 4   # ( vLocation )
    UseAbility                    = 5   # ( hAbility )
    UseAbilityOnEntity            = 6   # ( hAbility, hTarget )
    UseAbilityOnLocation          = 7   # ( hAbility, vLocation )
    UseAbilityOnTree              = 8   # ( hAbility, iTree )
    PickUpRune                    = 9   # ( nRune )
    PickUpItem                    = 10  # ( hItem )
    DropItem                      = 11  # ( hItem, vLocation )
    Delay                         = 12  # ( fDelay )
    PurchaseItem                  = 13  # ( sItemName )
    SellItem                      = 14  # ( hItem )
    DisassembleItem               = 15  # ( hItem )
    SetItemCombineLock            = 16  # ( hItem, bLocked )
    SwapItems                     = 17  # ( index1, index2 )
    Buyback                       = 18  # ()
    Glyph                         = 19  # ()
    LevelAbility                  = 20  # ( sAbilityName )
    # TODO: check how to implement thos
    # seems to be a regular ability that can be used
    # might not even be needed
    TakeOutpost                   = 21  # ()

    # Courier Action bundled to the hero
    CourierBurst                  = 22
    CourierEnemySecret            = 23
    CourierReturn                 = 24
    CourierSecret                 = 25
    CourierTakeStash              = 26
    CourierTransfert              = 27

    # Tensor cores work better with a multiple of 8
    # This gives us room to grow
    NotUsed1 = 28
    NotUsed2 = 29
    NotUsed3 = 30
    NotUsed4 = 31

    # The action exist but it is not necessary
    # Courier                         # ( hCourier, nAction )
    # UseShrine                       # ( hShrine )
    # MovePath                        # ( tWaypoints )

# Could bundle the courier action as a hero action
class CourierAction(IntEnum):
    BURST               = 0
    ENEMY_SECRET_SHOP   = 1
    RETURN              = 2
    SECRET_SHOP         = 3
    TAKE_STASH_ITEMS    = 4
    TRANSFER_ITEMS      = 5
    # bots will have to do 2 actions for those
    # not a big deal IMO
    # TAKE_AND_TRANSFER_ITEMS
    # COURIER_ACTION_SIDE_SHOP
    # COURIER_ACTION_SIDE_SHOP2

assert len(Action) == 32, '32 actions'

# Argument index
class ActionArgument(IntEnum):
    action   = 0
    vLoc     = 1
    hUnit    = 2
    hAbility = 3
    hTarget  = 4
    iTree    = 5
    nRune    = 6
    fDelay   = 7
    sItem    = 8
    hItem    = 9
    ix1      = 10
    ix2      = 11
    sAbility = 12


ARG = ActionArgument

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
        self.act[ARG.hAbility] = hAbility

    def UseAbilityOnEntity(self, hAbility, hTarget):
        self.act[ARG.action] = Action.UseAbilityOnEntity
        self.act[ARG.hAbility] = hAbility
        self.act[ARG.hTarget] = hTarget

    def UseAbilityOnLocation(self, hAbility, vLocation):
        self.act[ARG.action] = Action.UseAbilityOnLocation
        self.act[ARG.hAbility] = hAbility
        self.act[ARG.vLocation] = vLocation

    def UseAbilityOnTree(self, hAbility, iTree):
        self.act[ARG.action] = Action.UseAbilityOnTree
        self.act[ARG.hAbility] = hAbility
        self.act[ARG.iTree] = iTree

    def PickUpRune(self, nRune):
        self.act[ARG.action] = Action.PickUpRune
        self.act[ARG.nRune] = nRune

    def PickUpItem(self, hItem):
        self.act[ARG.action] = Action.PickUpItem
        self.act[ARG.hItem] = hItem

    def DropItem(self, hItem, vLocation):
        self.act[ARG.action] = Action.DropItem
        self.act[ARG.vLoc] = vLocation
        self.act[ARG.hItem] = hItem

    def Delay(self, fDelay):
        self.act[ARG.action] = Action.Delay
        self.act[ARG.fDelay] = fDelay

    def PurchaseItem(self, sItemName):
        self.act[ARG.action] = Action.PurchaseItem
        self.act[ARG.sItem] = sItemName

    def SellItem(self, hItem):
        self.act[ARG.action] = Action.SellItem
        self.act[ARG.hItem] = hItem

    def DisassembleItem(self, hItem):
        self.act[ARG.action] = Action.DisassembleItem
        self.act[ARG.hItem] = hItem

    def SetItemCombineLock(self, hItem):
        self.act[ARG.action] = Action.SetItemCombineLock
        self.act[ARG.hItem] = hItem

    def SwapItems(self, index1, index2):
        self.act[ARG.action] = Action.SwapItems
        self.act[ARG.ix1] = index1
        self.act[ARG.ix2] = index2

    def Buyback(self):
        self.act[ARG.action] = Action.Buyback

    def Glyph(self):
        self.act[ARG.action] = Action.Glyph

    def LevelAbility(self, sAbilityName):
        self.act[ARG.action] = Action.LevelAbility
        self.act[ARG.sAbility] = sAbilityName

    def TakeOutpost(self):
        self.act[ARG.action] = Action.TakeOutpost

    def CourierBurst(self):
        self.act[ARG.action] = Action.CourierBurst

    def CourierEnemySecret(self):
        self.act[ARG.action] = Action.CourierEnemySecret

    def CourierReturn(self):
        self.act[ARG.action] = Action.CourierReturn

    def CourierSecret(self):
        self.act[ARG.action] = Action.CourierSecret

    def CourierTakeStash(self):
        self.act[ARG.action] = Action.CourierTakeStash

    def CourierTransfert(self):
        self.act[ARG.action] = Action.CourierTransfert


class IPCMessageBuilder:
    def __init__(self, game=None):
        self.message = new_ipc_message()
        self.game = game

    def player(self, idx):
        faction = TEAM_RADIANT

        if idx > 4:
            faction = TEAM_DIRE

        return Player(self.message[faction][idx])

    def build(self):
        return self.message

    def send(self):
        if self.game is not None:
            return self.game.send_message(self.build())
