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


class ItemSlot(IntEnum):
    Item0 = 0
    Item1 = 1
    Item2 = 2
    Item3 = 3
    Item4 = 4
    Item5 = 5
    Item6 = 6
    Item7 = 7
    Item8 = 8
    Item9 = 9
    Item10 = 10
    Item11 = 11
    Item12 = 12
    Item13 = 13
    Item14 = 14
    Item15 = 15


assert len(ItemSlot) == 16, '16 item slots'


class AbilitySlot(IntEnum):
    Ablity0 = 0
    Ablity1 = 1
    Ablity2 = 2
    Ablity3 = 3
    Ablity4 = 4
    Ablity5 = 5
    Ablity6 = 6
    Ablity7 = 7
    Ablity8 = 8
    Ablity9 = 9
    Ablity10 = 10
    Ablity11 = 11
    Ablity12 = 12
    Ablity13 = 13
    Ablity14 = 14
    Ablity15 = 15
    Ablity16 = 16
    Ablity17 = 17
    Ablity18 = 18
    Ablity19 = 19
    Ablity20 = 20
    Ablity21 = 21
    Ablity22 = 22


assert len(AbilitySlot) == 23, '23 abilities'


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
    # TODO: check how to implement those
    # seems to be a regular ability that can be used
    # might not even be needed
    TakeOutpost                   = 21  # ()

    # Courier Action bundled to the hero
    CourierBurst                  = 22
    # CourierEnemySecret            = 23
    CourierReturn                 = 23
    CourierSecret                 = 24
    CourierTakeStash              = 25
    CourierTransfert              = 26

    # Tensor cores work better with a multiple of 8
    # This gives us room to grow
    NotUsed1 = 27
    NotUsed2 = 28
    NotUsed3 = 29
    NotUsed4 = 30
    NotUsed5 = 31

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
    iTree    = 4
    nRune    = 5
    fDelay   = 6
    sItem    = 7
    hItem    = 8
    ix1      = 9
    ix2      = 10


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
        self.act[ARG.hItem] = hItem

    def DropItem(self, hItem, vLocation):
        self.act[ARG.action] = Action.DropItem
        self.act[ARG.vLoc] = vLocation
        self.act[ARG.nSlot] = hItem

    def Delay(self, fDelay):
        self.act[ARG.action] = Action.Delay
        self.act[ARG.fDelay] = fDelay

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

    def SwapItems(self, index1, index2):
        self.act[ARG.action] = Action.SwapItems
        self.act[ARG.ix1] = index1
        self.act[ARG.ix2] = index2

    def Buyback(self):
        self.act[ARG.action] = Action.Buyback

    def Glyph(self):
        self.act[ARG.action] = Action.Glyph

    def LevelAbility(self, nSlot):
        self.act[ARG.action] = Action.LevelAbility
        self.act[ARG.nSlot] = nSlot

    def TakeOutpost(self):
        self.act[ARG.action] = Action.TakeOutpost

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
