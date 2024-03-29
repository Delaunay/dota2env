from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List


class Type:
    pass


class ActionData:
    pass

# .CMsgBotWorldState.Vector
@dataclass
class Vector:
    x: float
    y: float
    z: float

# .CMsgBotWorldState.Player
@dataclass
class Player:
    player_id: Optional[int]
    hero_id: Optional[int]
    is_alive: Optional[bool]
    respawn_time: Optional[float]
    kills: Optional[int]
    deaths: Optional[int]
    assists: Optional[int]
    team_id: Optional[int]

# .CMsgBotWorldState.Ability
@dataclass
class Ability:
    handle: Optional[int]
    ability_id: Optional[int]
    slot: Optional[int]
    caster_handle: Optional[int]
    level: Optional[int]
    cast_range: Optional[int]
    channel_time: Optional[float]
    cooldown_remaining: Optional[float]
    is_activated: Optional[bool]
    is_toggled: Optional[bool]
    is_in_ability_phase: Optional[bool]
    is_channeling: Optional[bool]
    is_stolen: Optional[bool]
    is_fully_castable: Optional[bool]
    charges: Optional[int]
    secondary_charges: Optional[int]
    is_combined_locked: Optional[bool]
    power_treads_stat: Optional[int]

# .CMsgBotWorldState.DroppedItem
@dataclass
class DroppedItem:
    item_id: Optional[int]
    location: Optional[Vector]

# .CMsgBotWorldState.RuneInfo
@dataclass
class RuneInfo:
    type: Optional[int]
    location: Optional[Vector]
    status: Optional[int]
    time_since_seen: Optional[float]

# .CMsgBotWorldState.TeleportInfo
@dataclass
class TeleportInfo:
    player_id: Optional[int]
    location: Optional[Vector]
    time_remaining: Optional[float]

# .CMsgBotWorldState.Modifier
@dataclass
class Modifier:
    name: Optional[str]
    stack_count: Optional[int]
    ability_handle: Optional[int]
    ability_id: Optional[int]
    remaining_duration: Optional[float]
    auxiliary_units_handles: List[int]

# .CMsgBotWorldState.UnitType
class UnitType(IntEnum):
    INVALID = 0
    HERO = 1
    CREEP_HERO = 2
    LANE_CREEP = 3
    JUNGLE_CREEP = 4
    ROSHAN = 5
    TOWER = 6
    BARRACKS = 7
    SHRINE = 8
    FORT = 9
    BUILDING = 10
    COURIER = 11
    WARD = 12

# .CMsgBotWorldState.LinearProjectile
@dataclass
class LinearProjectile:
    handle: Optional[int]
    caster_handle: Optional[int]
    caster_unit_type: Optional[UnitType]
    caster_player_id: Optional[int]
    ability_handle: Optional[int]
    ability_id: Optional[int]
    location: Optional[Vector]
    velocity: Optional[Vector]
    radius: Optional[int]

# .CMsgBotWorldState.TrackingProjectile
@dataclass
class TrackingProjectile:
    caster_handle: Optional[int]
    caster_unit_type: Optional[UnitType]
    caster_player_id: Optional[int]
    ability_handle: Optional[int]
    ability_id: Optional[int]
    location: Optional[Vector]
    velocity: Optional[int]
    is_dodgeable: Optional[bool]
    is_attack: Optional[bool]

# .CMsgBotWorldState.AvoidanceZone
@dataclass
class AvoidanceZone:
    location: Optional[Vector]
    caster_handle: Optional[int]
    caster_unit_type: Optional[UnitType]
    caster_player_id: Optional[int]
    ability_handle: Optional[int]
    ability_id: Optional[int]
    radius: Optional[int]

# .CMsgBotWorldState.CourierState
class CourierState(IntEnum):
    COURIER_STATE_INIT = -1
    COURIER_STATE_IDLE = 0
    COURIER_STATE_AT_BASE = 1
    COURIER_STATE_MOVING = 2
    COURIER_STATE_DELIVERING_ITEMS = 3
    COURIER_STATE_RETURNING_TO_BASE = 4
    COURIER_STATE_DEAD = 5

# .CMsgBotWorldState.Action.Courier
@dataclass
class Courier:
    handle: Optional[int]
    state: Optional[CourierState]
    player_id: Optional[int]

# .CMsgBotWorldState.Action.Courier
@dataclass
class Courier:
    unit: int
    courier: int
    action: int

# .CMsgBotWorldState.EventAbility
@dataclass
class EventAbility:
    ability_id: Optional[int]
    player_id: Optional[int]
    unit_handle: Optional[int]
    location: Optional[Vector]
    is_channel_start: Optional[bool]

# .CMsgBotWorldState.EventDamage
@dataclass
class EventDamage:
    damage: Optional[int]
    victim_player_id: Optional[int]
    victim_unit_handle: Optional[int]
    attacker_player_id: Optional[int]
    attacker_unit_handle: Optional[int]
    ability_id: Optional[int]

# .CMsgBotWorldState.EventCourierKilled
@dataclass
class EventCourierKilled:
    team_id: Optional[int]
    courier_unit_handle: Optional[int]
    killer_player_id: Optional[int]
    killer_unit_handle: Optional[int]

# .CMsgBotWorldState.EventRoshanKilled
@dataclass
class EventRoshanKilled:
    killer_player_id: Optional[int]
    killer_unit_handle: Optional[int]

# .CMsgBotWorldState.EventTree
@dataclass
class EventTree:
    tree_id: Optional[int]
    destroyed: Optional[bool]
    respawned: Optional[bool]
    location: Optional[Vector]
    delayed: Optional[bool]

# .CMsgBotWorldState.Unit
@dataclass
class Unit:
    handle: Optional[int]
    unit_type: Optional[UnitType]
    name: Optional[str]
    team_id: Optional[int]
    level: Optional[int]
    location: Optional[Vector]
    is_alive: Optional[bool]
    player_id: Optional[int]
    bounding_radius: Optional[int]
    facing: Optional[int]
    ground_height: Optional[int]
    vision_range_daytime: Optional[int]
    vision_range_nighttime: Optional[int]
    health: Optional[int]
    health_max: Optional[int]
    health_regen: Optional[float]
    mana: Optional[int]
    mana_max: Optional[int]
    mana_regen: Optional[float]
    base_movement_speed: Optional[int]
    current_movement_speed: Optional[int]
    anim_activity: Optional[int]
    anim_cycle: Optional[float]
    base_damage: Optional[int]
    base_damage_variance: Optional[int]
    bonus_damage: Optional[int]
    attack_damage: Optional[int]
    attack_range: Optional[int]
    attack_speed: Optional[float]
    attack_anim_point: Optional[float]
    attack_acquisition_range: Optional[int]
    attack_projectile_speed: Optional[int]
    attack_target_handle: Optional[int]
    attack_target_name: Optional[str]
    attacks_per_second: Optional[int]
    last_attack_time: Optional[float]
    bounty_xp: Optional[int]
    bounty_gold_min: Optional[int]
    bounty_gold_max: Optional[int]
    is_channeling: Optional[bool]
    active_ability_handle: Optional[int]
    is_attack_immune: Optional[bool]
    is_blind: Optional[bool]
    is_block_disabled: Optional[bool]
    is_disarmed: Optional[bool]
    is_dominated: Optional[bool]
    is_evade_disabled: Optional[bool]
    is_hexed: Optional[bool]
    is_invisible: Optional[bool]
    is_invulnerable: Optional[bool]
    is_magic_immune: Optional[bool]
    is_muted: Optional[bool]
    is_nightmared: Optional[bool]
    is_rooted: Optional[bool]
    is_silenced: Optional[bool]
    is_specially_deniable: Optional[bool]
    is_stunned: Optional[bool]
    is_unable_to_miss: Optional[bool]
    has_scepter: Optional[bool]
    abilities: List[Ability]
    items: List[Ability]
    modifiers: List[Modifier]
    incoming_tracking_projectiles: List[TrackingProjectile]
    action_type: Optional[int]
    ability_target_handle: Optional[int]
    ability_target_name: Optional[str]
    is_using_ability: Optional[bool]
    primary_attribute: Optional[int]
    is_illusion: Optional[bool]
    respawn_time: Optional[float]
    buyback_cost: Optional[int]
    buyback_cooldown: Optional[float]
    spell_amplification: Optional[float]
    armor: Optional[float]
    magic_resist: Optional[float]
    evasion: Optional[float]
    xp_needed_to_level: Optional[int]
    ability_points: Optional[int]
    reliable_gold: Optional[int]
    unreliable_gold: Optional[int]
    last_hits: Optional[int]
    denies: Optional[int]
    net_worth: Optional[int]
    strength: Optional[int]
    agility: Optional[int]
    intelligence: Optional[int]
    remaining_lifespan: Optional[float]
    flying_courier: Optional[bool]
    shrine_cooldown: Optional[float]
    is_shrine_healing: Optional[bool]

# .CMsgBotWorldState.Actions.OceanAnnotation.Hero
@dataclass
class Hero:
    playerID: int
    valueFunction: Optional[float]
    actionLogp: Optional[float]
    reward: Optional[float]
    internalAction: List[int]
    actionName: Optional[str]
    detailedStats: Optional[bytes]

# .CMsgBotWorldState.Actions.OceanAnnotation
@dataclass
class OceanAnnotation:
    heroes: List[Hero]
    agentID: Optional[str]
    rewards: List[float]
    reward_names: List[str]

# .CMsgBotWorldState.Actions.Header
@dataclass
class Header:
    startTime: Optional[float]
    name: Optional[str]

# .CMsgBotWorldState.Action
@dataclass
class Action:
    actionType: Type
    player: Optional[int]
    actionID: Optional[int]
    actionDelay: Optional[int]
    actionData: ActionData

# .CMsgBotWorldState.Actions
@dataclass
class Actions:
    dota_time: Optional[float]
    actions: List[Action]
    extraData: Optional[str]
    oceanAnnotation: Optional[OceanAnnotation]
    header: Optional[Header]

# .CMsgBotWorldState.Action.MoveToLocation
@dataclass
class MoveToLocation:
    units: List[int]
    location: Vector

# .CMsgBotWorldState.Action.MoveToTarget
@dataclass
class MoveToTarget:
    units: List[int]
    target: int

# .CMsgBotWorldState.Action.AttackMove
@dataclass
class AttackMove:
    units: List[int]
    location: Vector

# .CMsgBotWorldState.Action.AttackTarget
@dataclass
class AttackTarget:
    units: List[int]
    target: int
    once: Optional[bool]

# .CMsgBotWorldState.Action.HoldLocation
@dataclass
class HoldLocation:
    units: List[int]

# .CMsgBotWorldState.Action.Stop
@dataclass
class Stop:
    units: List[int]

# .CMsgBotWorldState.Action.CastLocation
@dataclass
class CastLocation:
    units: List[int]
    abilitySlot: int
    location: Vector

# .CMsgBotWorldState.Action.CastTarget
@dataclass
class CastTarget:
    units: List[int]
    abilitySlot: int
    target: int

# .CMsgBotWorldState.Action.CastTree
@dataclass
class CastTree:
    units: List[int]
    abilitySlot: int
    tree: int

# .CMsgBotWorldState.Action.Cast
@dataclass
class Cast:
    units: List[int]
    abilitySlot: int

# .CMsgBotWorldState.Action.CastToggle
@dataclass
class CastToggle:
    units: List[int]
    abilitySlot: int

# .CMsgBotWorldState.Action.TrainAbility
@dataclass
class TrainAbility:
    ability: str
    level: Optional[int]
    unit: Optional[int]

# .CMsgBotWorldState.Action.DropItem
@dataclass
class DropItem:
    unit: Optional[int]
    slot: Optional[int]
    location: Optional[Vector]

# .CMsgBotWorldState.Action.PickUpItem
@dataclass
class PickUpItem:
    unit: Optional[int]
    itemId: Optional[int]

# .CMsgBotWorldState.Action.PurchaseItem
@dataclass
class PurchaseItem:
    item: Optional[int]
    item_name: Optional[str]
    unit: Optional[int]

# .CMsgBotWorldState.Action.SellItem
@dataclass
class SellItem:
    item: Optional[int]
    slot: Optional[int]
    unit: Optional[int]

# .CMsgBotWorldState.Action.SwapItems
@dataclass
class SwapItems:
    slot_a: int
    slot_b: int
    unit: Optional[int]

# .CMsgBotWorldState.Action.DisassembleItem
@dataclass
class DisassembleItem:
    slot: int

# .CMsgBotWorldState.Action.SetCombineLockItem
@dataclass
class SetCombineLockItem:
    slot: int
    value: bool

# .CMsgBotWorldState.Action.PickupRune
@dataclass
class PickupRune:
    units: List[int]
    rune: int

# .CMsgBotWorldState.Action.Chat
@dataclass
class Chat:
    message: str
    to_allchat: bool

# .CMsgBotWorldState.Action.UseShrine
@dataclass
class UseShrine:
    units: List[int]
    shrine: int

# .CMsgBotWorldState.Action.GetActualIncomingDamage
@dataclass
class GetActualIncomingDamage:
    unit: int
    nDamage: float
    nDamageType: int

# .CMsgBotWorldState.Action.GetEstimatedDamageToTarget
@dataclass
class GetEstimatedDamageToTarget:
    unit: int
    bCurrentlyAvailable: bool
    hTarget: int
    fDuration: float
    nDamageTypes: int

# .CMsgBotWorldState.Action.Glyph
@dataclass
class Glyph:
    unit: int

# .CMsgBotWorldState.Action.SoftReset
@dataclass
class SoftReset:
    minigameConfig: Optional[str]
    snapshotData: Optional[str]

# .CMsgBotWorldState.Action.Buyback
@dataclass
class Buyback:
    unit: int

# .CMsgBotWorldState.Action.ScriptingDebugDrawText
@dataclass
class ScriptingDebugDrawText:
    origin: Vector
    text: str
    bViewCheck: bool
    duration: float

# .CMsgBotWorldState.Action.ScriptingDebugDrawLine
@dataclass
class ScriptingDebugDrawLine:
    origin: Vector
    target: Vector
    r: int
    g: int
    b: int
    ztest: bool
    duration: float

# .CMsgBotWorldState.Action.ScriptingDebugDrawScreenText
@dataclass
class ScriptingDebugDrawScreenText:
    x: float
    y: float
    lineOffset: int
    text: str
    r: int
    g: int
    b: int
    a: int
    duration: float

# .CMsgBotWorldState.Action.ScriptingDebugScreenTextPretty
@dataclass
class ScriptingDebugScreenTextPretty:
    x: float
    y: float
    lineOffset: int
    text: str
    r: int
    g: int
    b: int
    a: int
    duration: float
    font: str
    size: float
    bBold: bool

# .CMsgBotWorldState.Action.ScriptingDebugDrawBox
@dataclass
class ScriptingDebugDrawBox:
    origin: Vector
    minimum: Vector
    maximum: Vector
    r: int
    g: int
    b: int
    a: int
    duration: float

# .CMsgBotWorldState.Action.ScriptingDebugDrawCircle
@dataclass
class ScriptingDebugDrawCircle:
    center: Vector
    vRgb: Vector
    a: float
    rad: float
    ztest: bool
    duration: float

# .CMsgBotWorldState.Action.ScriptingDebugDrawClear
@dataclass
class ScriptingDebugDrawClear:
    pass

# .CMsgBotWorldState.Action.OceanWinGame
@dataclass
class OceanWinGame:
    team: str
    reward: Optional[float]

# .CMsgBotWorldState.Action.OceanReplayCorrectTime
@dataclass
class OceanReplayCorrectTime:
    delta: float

# .CMsgBotWorldState.Action.Type
class Type(IntEnum):
    DOTA_UNIT_ORDER_NONE = 0
    DOTA_UNIT_ORDER_MOVE_TO_POSITION = 1
    DOTA_UNIT_ORDER_MOVE_TO_TARGET = 2
    DOTA_UNIT_ORDER_ATTACK_MOVE = 3
    DOTA_UNIT_ORDER_ATTACK_TARGET = 4
    DOTA_UNIT_ORDER_CAST_POSITION = 5
    DOTA_UNIT_ORDER_CAST_TARGET = 6
    DOTA_UNIT_ORDER_CAST_TARGET_TREE = 7
    DOTA_UNIT_ORDER_CAST_NO_TARGET = 8
    DOTA_UNIT_ORDER_CAST_TOGGLE = 9
    DOTA_UNIT_ORDER_HOLD_POSITION = 10
    DOTA_UNIT_ORDER_TRAIN_ABILITY = 11
    DOTA_UNIT_ORDER_DROP_ITEM = 12
    DOTA_UNIT_ORDER_GIVE_ITEM = 13
    DOTA_UNIT_ORDER_PICKUP_ITEM = 14
    DOTA_UNIT_ORDER_PICKUP_RUNE = 15
    DOTA_UNIT_ORDER_PURCHASE_ITEM = 16
    DOTA_UNIT_ORDER_SELL_ITEM = 17
    DOTA_UNIT_ORDER_DISASSEMBLE_ITEM = 18
    DOTA_UNIT_ORDER_MOVE_ITEM = 19
    DOTA_UNIT_ORDER_CAST_TOGGLE_AUTO = 20
    DOTA_UNIT_ORDER_STOP = 21
    DOTA_UNIT_ORDER_TAUNT = 22
    DOTA_UNIT_ORDER_BUYBACK = 23
    DOTA_UNIT_ORDER_GLYPH = 24
    DOTA_UNIT_ORDER_EJECT_ITEM_FROM_STASH = 25
    DOTA_UNIT_ORDER_CAST_RUNE = 26
    DOTA_UNIT_ORDER_PING_ABILITY = 27
    DOTA_UNIT_ORDER_MOVE_TO_DIRECTION = 28
    DOTA_UNIT_ORDER_PATROL = 29
    DOTA_UNIT_ORDER_VECTOR_TARGET_POSITION = 30
    DOTA_UNIT_ORDER_RADAR = 31
    DOTA_UNIT_ORDER_SET_ITEM_COMBINE_LOCK = 32
    DOTA_UNIT_ORDER_CONTINUE = 33
    ACTION_CHAT = 40
    ACTION_SWAP_ITEMS = 41
    ACTION_USE_SHRINE = 42
    ACTION_COURIER = 43
    RPC_GET_ACTUAL_INCOMING_DAMAGE = 44
    RPC_GET_ESTIMATED_DAMAGE_TO_TARGET = 45
    OCEAN_FULL_UPDATE = 50
    OCEAN_RELOAD_CODE = 51
    OCEAN_SOFT_RESET = 52
    OCEAN_HOLD_FRAMESKIP = 54
    OCEAN_WIN_GAME = 63
    OCEAN_REPLAY_CORRECT_TIME = 64
    SCRIPTING_DEBUG_DRAW_TEXT = 55
    SCRIPTING_DEBUG_DRAW_LINE = 56
    SCRIPTING_DOTA_UNIT_ORDER_MOVE_TO_POSITION = 57
    SCRIPTING_DEBUG_DRAW_SCREEN_TEXT = 58
    SCRIPTING_DEBUG_DRAW_BOX = 59
    SCRIPTING_DEBUG_DRAW_CIRCLE = 60
    SCRIPTING_DEBUG_DRAW_CLEAR = 61
    SCRIPTING_DEBUG_SCREEN_TEXT_PRETTY = 65
    DOTA_UNIT_ORDER_MOVE_DIRECTLY = 62

# @union
@dataclass
class ActionData:
    moveToLocation: Optional[MoveToLocation]
    moveToTarget: Optional[MoveToTarget]
    attackMove: Optional[AttackMove]
    attackTarget: Optional[AttackTarget]
    castLocation: Optional[CastLocation]
    castTarget: Optional[CastTarget]
    castTree: Optional[CastTree]
    cast: Optional[Cast]
    castToggle: Optional[CastToggle]
    holdLocation: Optional[HoldLocation]
    trainAbility: Optional[TrainAbility]
    dropItem: Optional[DropItem]
    pickUpItem: Optional[PickUpItem]
    pickupRune: Optional[PickupRune]
    purchaseItem: Optional[PurchaseItem]
    sellItem: Optional[SellItem]
    disassembleItem: Optional[DisassembleItem]
    setCombineLockItem: Optional[SetCombineLockItem]
    stop: Optional[Stop]
    chat: Optional[Chat]
    swapItems: Optional[SwapItems]
    useShrine: Optional[UseShrine]
    courier: Optional[Courier]
    getActualIncomingDamage: Optional[GetActualIncomingDamage]
    getEstimatedDamageToTarget: Optional[GetEstimatedDamageToTarget]
    glyph: Optional[Glyph]
    softReset: Optional[SoftReset]
    buyback: Optional[Buyback]
    scriptingDebugDrawText: Optional[ScriptingDebugDrawText]
    scriptingDebugDrawLine: Optional[ScriptingDebugDrawLine]
    scriptingDebugDrawScreenText: Optional[ScriptingDebugDrawScreenText]
    scriptingDebugDrawBox: Optional[ScriptingDebugDrawBox]
    scriptingDebugDrawCircle: Optional[ScriptingDebugDrawCircle]
    scriptingDebugDrawClear: Optional[ScriptingDebugDrawClear]
    scriptingDebugScreenTextPretty: Optional[ScriptingDebugScreenTextPretty]
    moveDirectly: Optional[MoveToLocation]
    oceanWinGame: Optional[OceanWinGame]
    oceanReplayCorrectTime: Optional[OceanReplayCorrectTime]

# .CMsgBotWorldState
@dataclass
class CMsgBotWorldState:
    team_id: Optional[int]
    game_time: Optional[float]
    dota_time: Optional[float]
    game_state: Optional[int]
    hero_pick_state: Optional[int]
    time_of_day: Optional[float]
    glyph_cooldown: Optional[float]
    glyph_cooldown_enemy: Optional[int]
    players: List[Player]
    units: List[Unit]
    dropped_items: List[DroppedItem]
    rune_infos: List[RuneInfo]
    incoming_teleports: List[TeleportInfo]
    linear_projectiles: List[LinearProjectile]
    avoidance_zones: List[AvoidanceZone]
    couriers: List[Courier]
    ability_events: List[EventAbility]
    damage_events: List[EventDamage]
    courier_killed_events: List[EventCourierKilled]
    roshan_killed_events: List[EventRoshanKilled]
    tree_events: List[EventTree]
