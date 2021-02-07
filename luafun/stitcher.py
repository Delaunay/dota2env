"""
This module process the world state delta sent by Dota2 Game
and stitch together a consistent state to be observed by the bots
"""

from collections import defaultdict
import copy
from dataclasses import dataclass, field
import json
from typing import List, Dict, Any, Optional

import luafun.game.dota2.state_types as msg
from luafun.draft import DraftStatus
from luafun.proximity import ProximityMapper

GAME_START = 30
NIGHT_DAY_TIME = 5 * 60


@dataclass
class GlobalGameState:
    game_delta: float = 0
    game_time: float = 0  # seconds
    # 0.25-0.75 = DAY
    time_of_day: float = 0.0  # 0-1 / 5 minutes day - 5 minutes night
    time_to_next: float = 0.0
    time_spawn_creep: float = GAME_START  # every 30 seconds
    time_spawn_neutral: float = 1 + GAME_START  # every 1 minutes
    time_spawn_bounty: float = GAME_START  # 0:00 and every 5 minutes after
    time_spawn_runes: float = 4 + GAME_START  # 4:00 and every 2 minutes after
    time_since_enemy_courier: float = 0  # depecrated?
    rosh_spawn_min: float = 0
    rosh_spawn_max: float = 0
    rosh_alive: bool = True
    rosh_dead: bool = False
    rosh_cheese: bool = False
    rosh_refresher: bool = False
    rosh_aghs: bool = False  # New
    rosh_health_rand: float = 0  # Appendix O
    glyph_cooldown: float = 300
    glyph_cooldown_enemy: float = 300
    stock_gem: int = 1
    stock_smoke: int = 2
    stock_observer: int = 2
    stock_sentry: int = 3
    stock_raindrop: int = 0


@dataclass
class Unit:
    x: float = 0
    y: float = 0
    z: float = 0
    facing_cos: float = 0
    facing_sin: float = 0
    is_attacking: bool = 0
    time_since_last_attack: float = 0
    max_health: float = 0
    health_00: float = 0  # Current health
    health_01: float = 0
    health_02: float = 0
    health_03: float = 0
    health_04: float = 0
    health_05: float = 0
    health_06: float = 0
    health_07: float = 0
    health_08: float = 0
    health_09: float = 0
    health_10: float = 0
    health_11: float = 0
    health_12: float = 0
    health_13: float = 0
    health_14: float = 0
    health_15: float = 0
    attack_damage: float = 0
    attack_speed: float = 0
    physical_resistance: float = 0
    is_glyphed: bool = 0
    glyph_time: float = 0
    movement_speed: float = 0
    is_ally: bool = False
    is_neutral: bool = False
    animation_cycle_time: float = 0
    eta_incoming_projectile: float = 0
    number_of_unit_attacking_me: int = 0
    # shrine_cooldown                   # deprecrated
    to_current_unit_dx: float = 0
    to_current_unit_dy: float = 0
    to_current_unit_l: float = 0
    is_current_hero_attacking_it: bool = False
    is_attacking_current_hero: bool = False
    eta_projectile_to_hero: float = 0
    unit_type_INVALID: int = 0
    unit_type_HERO: int = 0
    unit_type_CREEP_HERO: int = 0
    unit_type_LANE_CREEP: int = 0
    unit_type_JUNGLE_CREEP: int = 0
    unit_type_ROSHAN: int = 0
    unit_type_TOWER: int = 0
    unit_type_BARRACKS: int = 0
    unit_type_SHRINE: int = 0
    unit_type_FORT: int = 0

    unit_type_BUILDING: int = 0
    unit_type_COURIER: int = 0
    unit_type_WARD: int = 0


@dataclass
class Heroes:
    pass


@dataclass
class AlliedHeroes:
    pass


def none():
    return None


def dictionary():
    return defaultdict(dict)


@dataclass
class FactionState:
    global_state: GlobalGameState = field(default_factory=GlobalGameState)
    units: List[Unit] = field(default_factory=list)
    heroes: List[Heroes] = field(default_factory=list)
    allied_heroes: List[AlliedHeroes] = field(default_factory=list)
    nearby_map: Any = field(default_factory=none)
    previous_action: Any = field(default_factory=none)
    modifiers: Any = field(default_factory=none)
    item: Any = field(default_factory=none)
    ability: Any = field(default_factory=none)
    pickup: Any = field(default_factory=none)
    dropitems: list = field(default_factory=list)
    courier: dict = field(default_factory=dictionary)
    trees: dict = field(default_factory=dictionary)
    runes: dict = field(default_factory=dictionary)
    draft: Optional[DraftStatus] = None

    # internal data to generate some of the field
    # unit lookup etc...
    _roshan_dead: int = 0
    _players: Dict = field(default_factory=dictionary)
    _couriers: Dict = field(default_factory=dictionary)
    _units: Dict = field(default_factory=dictionary)
    _buildings: Dict = field(default_factory=dictionary)
    _proximity: ProximityMapper = field(default_factory=ProximityMapper)

    # State Management
    _s: int = 0
    _e: int = 0
    _r: int = 0

    def get_entities(self, x, y):
        """Returns the unit/entity that is closest to the specified location"""
        return self._proximity.entities(x, y)

    def __deepcopy__(self, memo):
        state = FactionState(
            copy.deepcopy(self.global_state, memo),
            copy.deepcopy(self.units, memo),
            copy.deepcopy(self.heroes, memo),
            copy.deepcopy(self.allied_heroes, memo),
            copy.deepcopy(self.nearby_map, memo),
            copy.deepcopy(self.previous_action, memo),
            copy.deepcopy(self.modifiers, memo),
            copy.deepcopy(self.item, memo),
            copy.deepcopy(self.ability, memo),
            copy.deepcopy(self.pickup, memo),
        )

        state._roshan_dead = copy.deepcopy(self._roshan_dead, memo),
        state._players = copy.deepcopy(self._players, memo),
        state._units = copy.deepcopy(self._units, memo),
        state._s = self._s
        state._e = self._e
        state._r = self._r
        return state

    def copy(self):
        return copy.deepcopy(self)


def time_to_day_night(game_time):
    """Returns the time to next day-night change"""
    # day starts at 0:00
    if game_time < 0:
        return - game_time

    # every 5 minutes after that
    down = game_time / NIGHT_DAY_TIME
    return (1 - (down - int(down))) * NIGHT_DAY_TIME


def time_spawn_creep(game_time):
    # creep starts at 0:00
    if game_time < 0:
        return - game_time

    # every 30 seconds after that
    down = game_time / 30
    return (1 - (down - int(down))) * 30


def time_spawn_bounty(game_time):
    # bounty spawns at 0 and every 5 minutes until then
    if game_time < 0:
        return - game_time

    # every 5 minutes after that
    return time_to_day_night(game_time)


def time_spawn_runes(game_time):
    # runs spawn after the 4th minutes of the game
    if game_time < 0:
        return - game_time + 4 * 60

    if game_time < 4 * 60:
        return 4 * 60 - game_time

    # every 2 minutes after that
    down = game_time / 2 * 60
    return (1 - (down - int(down))) * 2 * 60


def time_spawn_neutral(game_time):
    if game_time < 0:
        return - game_time + 60

    if game_time < 60:
        return 60 - game_time

    # every 1 minutes after that
    down = game_time / 60
    return (1 - (down - int(down))) * 60


class Stitcher:
    def __init__(self):
        pass

    @property
    def observation_space(self):
        """Returns the observation space that we are stitching"""
        from luafun.observations import StateBuilder
        return StateBuilder()

    @staticmethod
    def initial_state():
        return FactionState()

    def apply_diff(self, state, delta: msg.CMsgBotWorldState):
        """Take a world state delta and apply it to a previous state"""
        self.generic_apply(state, delta)

    def generic_apply(self, state, delta: msg.CMsgBotWorldState):
        """Take a world state delta and apply it to a previous state"""
        state._s += 1

        # delta = json.loads(delta)
        g = state.global_state

        dota_time = delta['dota_time']

        # Game timings
        g.game_delta = dota_time - g.game_time
        g.game_time = dota_time
        g.glyph_cooldown = delta['glyph_cooldown']
        g.glyph_cooldown_enemy = delta['glyph_cooldown_enemy']
        g.time_of_day = delta['time_of_day']
        g.time_to_next = time_to_day_night(g.game_time)
        g.time_spawn_runes = time_spawn_runes(g.game_time)
        g.time_spawn_neutral = time_spawn_neutral(g.game_time)
        g.time_spawn_bounty = time_spawn_bounty(g.game_time)
        g.time_spawn_creep = time_spawn_creep(g.game_time)
        # ---

        # Roshan State
        for event in delta.get('roshan_killed_events', []):
            state._roshan_dead += 1
            g.rosh_spawn_min = g.game_time + 8 * 60
            g.rosh_spawn_max = g.game_time + 11 * 60
            g.rosh_alive = False
            g.rosh_dead = True
            g.rosh_cheese = state._roshan_dead > 1
            g.rosh_refresher = state._roshan_dead > 2 or state._roshan_dead > 3
            g.rosh_aghs = state._roshan_dead > 2 or state._roshan_dead > 3

            # TODO: handle the event and update the player which is holding the aegis
            event = delta.roshan_killed_events[0]

        if g.game_time > g.rosh_spawn_max:
            g.rosh_alive = True
        # --

        # Item Stock
        # stock_gem: int = 1
        # stock_smoke: int = 2
        # stock_observer: int = 2
        # stock_sentry: int = 3
        # stock_raindrop: int = 0
        # ---

        # Unit specific Event
        # This the base player data
        for player in delta.get('players', []):
            pdata = state._players[player['player_id']]

            for field, value in player.items():
                pdata[field] = value

        for unit in delta.get('units', []):
            remove_dead = False

            # Add Hero info into their own struct
            # >>> Units that are constant
            if unit['unit_type'] == msg.UnitType.HERO:
                source = state._players
                key = 'player_id'

            # exist for the the entire game
            elif unit['unit_type'] == msg.UnitType.COURIER:
                source = state._couriers
                key = 'player_id'

            # They stay up for most of the game
            elif unit['unit_type'] in (
            msg.UnitType.BUILDING, msg.UnitType.FORT, msg.UnitType.BARRACKS, msg.UnitType.TOWER):
                source = state._buildings
                key = 'handle'
            # Roshan

            # Neutrals

            # <<<
            # CREEP_HERO, LANE_CREEP
            else:
                remove_dead = True
                source = state._units
                key = 'handle'

            udata = source[unit[key]]

            # this is a edge case that should almost never happen
            # all items are sent
            if unit['unit_type'] == msg.UnitType.HERO:
                if 'items' not in unit:
                    udata['items'] = []
            # ----

            for field, value in unit.items():
                udata[field] = value

            handle = unit['handle']
            pos = unit['location']
            state._proximity.manager.update_position(handle, pos['x'], pos['y'])

            if remove_dead and not udata.get('is_alive', True):
                source.pop(unit[key])
        # ---

        # Courier Event
        for courier in delta.get('couriers', []):
            courier_unit = state.courier[courier['handle']]
            courier_unit['player_id'] = courier['player_id']
            courier_unit['state'] = courier['state']

        for event in delta.get('courier_killed_events', []):
            pass
        # --

        # Tree Event
        for tree_event in delta.get('tree_events', []):
            if tree_event.get('destroyed', False):
                state._proximity.tree.pop_entity(event['tree_id'])
                continue

            if tree_event.get('respawned', False):
                x, y = state._proximity.tid_pos[event['tree_id']]
                state._proximity.tree.add_entity(event['tree_id'], x, y)
        # --

        # Damage Event:
        for dmg in delta.get('damage_events', []):
            pass
        # --

        # Ability Event:
        for ability in delta.get('ability_events', []):
            pass
        # --

        # Rune info
        for rune in delta.get('rune_infos', []):
            x = rune['location']['x']
            y = rune['location']['y']

            rune_id = f'{int(x)}{int(y)}'
            rune_state = state.runes[rune_id]

            for f, value in rune.items():
                rune_state[f] = value

        # --

        # This is not a delta
        # Item Drops (Neutral, Roshan)
        state.dropitems = delta.get('dropped_items', [])
        # --

        state._e += 1
