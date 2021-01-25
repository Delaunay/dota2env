from enum import IntEnum


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
    Courier                       = 18  # ( hCourier, nAction )
    Buyback                       = 19  # ()
    Glyph                         = 20  # ()
    LevelAbility                  = 21  # ( sAbilityName )
    TakeOutpost                   = 22  # ()

    # Tensor cores work better with a multiple of 8
    # This gives us room to grow
    NotUsed1 = 23   # MovePath                        # ( tWaypoints )
    NotUsed2 = 24   # UseShrine                       # ( hShrine )
    NotUsed3 = 25
    NotUsed4 = 26
    NotUsed5 = 27
    NotUsed6 = 28
    NotUsed7 = 29
    NotUsed8 = 30
    NotUsed9 = 31


class CourierAction(IntEnum):
    BURST               = 0
    ENEMY_SECRET_SHOP   = 1
    RETURN              = 2
    SECRET_SHOP         = 3
    TAKE_STASH_ITEMS    = 4
    TRANSFER_ITEMS      = 5
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
