from enum import IntEnum


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

class ActionArgument(IntEnum):
    location        = 0
    Ability         = 1
    AbilityName     = 2 # For leveling up
    Item            = 3
    ItemName        = 4 # For purchasing
    Unit            = 5
    ItemSlotPair    = 6 # For Swapping
    CourierAction   = 7
    DelayTime       = 8
    Rune            = 9
