"""
This module process the world state delta sent by Dota2 Game
and stitch together a consistent state to be observed by the bots
"""

from collections import defaultdict
import copy
from dataclasses import dataclass, field
from math import cos, sin, sqrt
from typing import List, Dict, Any, Optional

import torch

import luafun.game.dota2.state_types as msg
from luafun.utils.ring import RingBuffer
from luafun.draft import DraftStatus
from luafun.proximity import ProximityMapper
from luafun.observations import CommonState, ItemState, AbilityState, RuneState, Minimap10x10Tile
from luafun.observations import UnitState, HeroUnit, AllyHeroState, PreviousActionState, ModifierState
import luafun.game.constants as const

GAME_START = 30
NIGHT_DAY_TIME = 5 * 60


@dataclass
class AlliedHeroes:
    pass


def none():
    return None


def dictionary():
    return defaultdict(dict)


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


UNIT_MODIFIER = 2
HERO_MODIFIER = 10
HERO_ABILITIES = 6
HERO_ITEMS = 17


class Unit:
    def __init__(self):
        self.unit = torch.zeros((UnitState.Size,))
        self.modifiers = [torch.zeros((ModifierState.Size,)) for _ in range(UNIT_MODIFIER)]

    @staticmethod
    def size():
        return UnitState.Size + ModifierState.Size * 2

    def tensor(self):
        u = torch.zeros((Unit.size(),))

        s = 0
        e = UnitState.Size
        u[s:e] = self.unit

        for i in range(len(self.modifiers)):
            s = e
            e = s + ModifierState.Size
            u[s:e] = self.modifiers[i]

        return u


class Player:
    def __init__(self, ally):
        self.is_ally = ally
        self.ally = None

        if self.ally:
            self.ally = torch.zeros((AllyHeroState.Size,))

        self.unit = torch.zeros((UnitState.Size,))
        self.hero = torch.zeros((HeroUnit.Size,))
        self.modifiers = [torch.zeros((ModifierState.Size,)) for _ in range(HERO_MODIFIER)]
        self.items = [torch.zeros((ItemState.Size,)) for _ in range(HERO_ITEMS)]
        self.abilities = [torch.zeros((AbilityState.Size,)) for _ in range(HERO_ABILITIES)]

    @staticmethod
    def size(is_ally):
        base = (
            UnitState.Size +
            HeroUnit.Size +
            ModifierState.Size * HERO_MODIFIER +
            ItemState.Size * HERO_ITEMS +
            AbilityState.Size * HERO_ABILITIES
        )

        if not is_ally:
            return base

        return base + AllyHeroState.Size

    def tensor(self):
        p = torch.zeros((Player.size(self.is_ally),))

        s = 0
        e = UnitState.Size
        p[s:e] = self.unit

        s = e
        e = s + HeroUnit.Size
        p[s:e] = self.hero

        for i in range(len(self.items)):
            s = e
            e = s + ItemState.Size
            p[s:e] = self.items[i]

        for i in range(len(self.abilities)):
            s = e
            e = s + AbilityState.Size
            p[s:e] = self.abilities[i]

        for i in range(len(self.modifiers)):
            s = e
            e = s + ModifierState.Size
            p[s:e] = self.modifiers[i]

        if self.is_ally:
            s = e
            e = s + AllyHeroState.Size
            p[s:e] = self.ally

        return p


def default_buffer(size):
    def make():
        b = RingBuffer(size, None)
        b.offset = size - 1
        return b

    return make


class Stitcher:
    def __init__(self, faction):
        self.roshan_death_count = 0
        self.faction = faction
        self.minimap = None
        self.common = None
        self.draft = None
        self.latest_message = None
        self.heroes = dict()
        self.units = dict()
        self.runes = dict()
        self.health_tracker = defaultdict(default_factory=default_buffer(16))

        # Proximity mapping
        self.proximities = ProximityMapper()

    def get_entities(self, x, y):
        return self.proximities.entities(x, y)

    @property
    def observation_space(self):
        """Returns the observation space that we are stitching"""
        return (
            Player.size(False) * 5 +
            Player.size(True) * 5 +
            Unit.size() * (189 - 10) +
            RuneState.Size * 6 +
            CommonState.Size
        )

    def apply_diff(self, delta: msg.CMsgBotWorldState):
        """Take a world state delta and apply it to a previous state"""
        self.latest_message = delta
        self.generic_apply(delta)

    def generate_batch(self, botids):
        return None

    def generate_player(self, bid):
        state = torch.zeros((self.observation_space,))
        s = 0
        e = 0

        player = self.heroes.get(bid).unit
        px, py = player[UnitState.X], player[UnitState.Y]

        s = e
        e = s + CommonState.Size
        state[s:e] = self.common

        for i in range(6):
            s = e
            e = s + RuneState.Size
            state[s:e] = self.runes.get(i)

        closest = []
        for k, unit in self.units.items():
            x, y = unit[UnitState.X], unit[UnitState.Y]
            dist = (x - px) ** 2 + (y - py) ** 2
            closest.append((k, dist))

        closest.sort()

        for i, (k, _) in enumerate(closest):
            if i > 189 - 10:
                break

            unit = self.units[k]

            s = e
            e = s + UnitState.Size
            state[s:e] = unit

        return None

    def prepare_common(self, state) -> torch.Tensor:
        g = torch.zeros((CommonState.Size,))
        f = CommonState
        dota_time = state['dota_time']

        # Game timings
        g[f.GameTime] = dota_time
        g[f.GlyphCooldown] = state['glyph_cooldown']
        g[f.GlyphCooldownEnemy] = state['glyph_cooldown_enemy']
        g[f.TimeOfDay] = state['time_of_day']
        g[f.TimeToNext] = time_to_day_night(dota_time)
        g[f.TimeSpawnRunes] = time_spawn_runes(dota_time)
        g[f.TimeSpawnNeutral] = time_spawn_neutral(dota_time)
        g[f.TimeSpawnBounty] = time_spawn_bounty(dota_time)
        g[f.TimeSpawnCreep] = time_spawn_creep(dota_time)
        # ---

        # Item Stock
        # TODO: fix this
        g[f.StockGem] = 1
        g[f.StockSmoke] = 2
        g[f.StockObserver] = 2
        g[f.StockSentry] = 3
        g[f.StockRaindrop] = 0

        # Roshan State
        g[f.RoshAlive] = True
        for _ in state.get('roshan_killed_events', []):
            self.roshan_death_count += 1
            g[f.RoshSpawnMin] = dota_time + 8 * 60
            g[f.RoshSpawnMax] = dota_time + 11 * 60
            g[f.RoshAlive] = False
            g[f.RoshDead] = True
            g[f.RoshCheese] = self.roshan_death_count > 1
            g[f.RoshRefresher] = self.roshan_death_count > 2 or self.roshan_death_count > 3
            g[f.RoshAghs] = self.roshan_death_count > 2 or self.roshan_death_count > 3

        if g[f.GameTime] > g[f.RoshSpawnMax]:
            g[f.RoshAlive] = True

        return g

    BARRACKS = 7 - 1
    HERO = 1 - 1
    TOWER = 6 - 1
    ROSHAN = 5 - 1

    def prepare_unit(self, msg) -> torch.Tensor:
        u = torch.zeros((UnitState.Size,))
        f = UnitState

        uid = msg['handle']

        # --- If dead and not a hero ignore that unit
        offset = msg['unit_type'] - 1
        if not msg['is_alive'] and offset not in (self.BARRACKS, self.HERO, self.TOWER, self.ROSHAN):
            self.units.pop(uid, None)
            return None
        # ---

        pos = (msg['location']['x'], msg['location']['y'], msg['location']['z'])
        if not msg['is_alive']:
            pos = (0, 0, 0)

        health: RingBuffer = self.health_tracker[uid]
        health.append(msg['health'])
        health = health.to_list()

        self.proximities.manager.update_position(uid, pos[0], pos[1])
        u[f.X] = pos[0]
        u[f.Y] = pos[1]
        u[f.Z] = pos[2]
        u[f.FacingCos] = cos(msg['facing'])
        u[f.FacingSin] = sin(msg['facing'])
        u[f.IsAttacking] = 'action_type' == 'attacking' or 'attack_target_handle'
        u[f.TimeSinceLastAttack] = msg['last_attack_time']
        u[f.MaxHealth] = msg['health_max']
        u[f.Health00] = health[-1]
        u[f.Health01] = health[-2]
        u[f.Health02] = health[-3]
        u[f.Health03] = health[-4]
        u[f.Health04] = health[-5]
        u[f.Health05] = health[-6]
        u[f.Health06] = health[-7]
        u[f.Health07] = health[-8]
        u[f.Health08] = health[-9]
        u[f.Health09] = health[-10]
        u[f.Health10] = health[-11]
        u[f.Health11] = health[-12]
        u[f.Health12] = health[-13]
        u[f.Health13] = health[-14]
        u[f.Health14] = health[-15]
        u[f.Health15] = health[-16]
        u[f.AttackDamage] = msg['attack_damage']
        u[f.AttackSpeed] = msg['attack_speed']
        u[f.PhysicalResistance] = msg['armor']
        u[f.IsGlyphed] = 0
        u[f.GlyphTime] = 0
        u[f.MovementSpeed] = msg['current_movement_speed']
        u[f.IsAlly] = msg['team_id'] == self.faction
        u[f.IsNeutral] = msg['team_id'] == 1
        u[f.AnimationCycleTime] = msg['anim_cycle']
        u[f.EtaIncomingProjectile] = 0
        u[f.NumberOfUnitAttackingMe] = 0
        # u[f.ShrineCooldown        ] = 0
        u[f.ToCurrentUnitdx] = 0
        u[f.ToCurrentUnitdy] = 0
        u[f.ToCurrentUnitl] = 0
        u[f.IsCurrentHeroAttackingIt] = 0
        u[f.IsAttackingCurrentHero] = 0
        u[f.EtaProjectileToHero] = 0
        u[f.UnitTypeHERO + offset] = 1

        return u

    def prepare_rune(self, rmsg):
        r = torch.zeros((RuneState.Size,))
        f = RuneState

        # RUNE_STATUS_UNKNOWN = 0
        RUNE_STATUS_AVAILABLE = 1
        # RUNE_STATUS_MISSING = 2

        r[f.Visible] = rmsg['status'] == RUNE_STATUS_AVAILABLE
        r[f.LocationX] = rmsg['location']['x']
        r[f.LocationY] = rmsg['location']['y']
        r[f.DistanceH0] = 0
        r[f.DistanceH1] = 0
        r[f.DistanceH2] = 0
        r[f.DistanceH3] = 0
        r[f.DistanceH4] = 0
        r[f.DistanceH5] = 0
        r[f.DistanceH6] = 0
        r[f.DistanceH7] = 0
        r[f.DistanceH8] = 0
        r[f.DistanceH9] = 0

        return r

    def process_trees(self, delta):
        # Tree Event
        for tree_event in delta.get('tree_events', []):
            if tree_event.get('destroyed', False):
                self.proximities.tree.pop_entity(tree_event['tree_id'])
                continue

            if tree_event.get('respawned', False):
                x, y = self.proximities.tid_pos[tree_event['tree_id']]
                self.proximities.tree.add_entity(tree_event['tree_id'], x, y)

    def prepare_hero_unit(self, pmsg):
        h = torch.zeros((HeroUnit.Size,))
        f = HeroUnit

        umsg = pmsg['unit']
        tpitem = umsg['pitems'].get(const.ItemSlot.Item15, dict())

        h[f.IsAlive] = umsg['is_alive']
        h[f.NumberOfDeath] = pmsg['deaths']
        h[f.HeroInSight] = 1
        h[f.LastSeenTime] = 0
        h[f.IsTeleporting] = tpitem.get('is_channeling', False)
        h[f.TeleportTargetX] = 0
        h[f.TeleportTargetY] = 0
        h[f.TeleportChannelTime] = tpitem.get('channel_time', 0)
        h[f.CurrentGold] = umsg['reliable_gold'] + umsg['unreliable_gold']
        h[f.Level] = umsg['level'] / 30
        h[f.ManaMax] = umsg['mana_max']
        h[f.Mana] = umsg['mana']
        h[f.ManaRegen] = umsg['mana_regen']
        h[f.HealthMax] = umsg['health_max']
        h[f.Health] = umsg['health']
        h[f.HealthRegen] = umsg['health_regen']
        h[f.MagicResitance] = umsg['magic_resist']
        h[f.Strength] = umsg['strength']
        h[f.Agility] = umsg['agility']
        h[f.Intelligence] = umsg['intelligence']
        h[f.IsInvisibility] = umsg['is_invisible']
        h[f.IsUsingAbility] = umsg['is_using_ability']
        h[f.NumberOfAllied] = 0
        h[f.NumberOfAlliedCreeps] = 0
        h[f.NumberOfEnemy] = 0
        h[f.NumberOfEnemyCreeps] = 0

        return h

    def process_hero(self, umsg):
        pitems = dict()
        umsg['pitems'] = pitems

        for item in umsg['items']:
            pitems[item['slot']] = item

        pabi = dict()
        umsg['pabi'] = pabi

        for item in umsg['abilities']:
            pitems[item['slot']] = item

    def prepare_ally_hero(self, umsg):
        h = torch.zeros((AllyHeroState.Size,))
        f = AllyHeroState

        has_buyback = umsg['unreliable_gold'] < umsg['buyback_cost'] and umsg['buyback_cooldown'] < 0.001
        h[f.HasBuyBack] = has_buyback
        h[f.BuyBackCost] = umsg['buyback_cost']
        h[f.BuyBackCooldown] = umsg['buyback_cooldown']

        # --
        h[f.LaneAssignRoam] = 0
        h[f.LaneAssignTop] = 0
        h[f.LaneAssignMid] = 0
        h[f.LaneAssignBot] = 0

        h[f.NumberOfEmptyInventory] = 0
        h[f.NumberOfEmptyBackPack] = 0

        return h

    def prepare_ability(self, amsg):
        a = torch.zeros((AbilityState.Size,))
        f = AbilityState

        a[f.Cooldown] = amsg['cooldown_remaining']
        a[f.InUse] = amsg['is_in_ability_phase'] or amsg['is_channeling']
        a[f.Castable] = amsg['is_fully_castable']

        level = amsg['level']

        if level > 0:
            a[f.Level1 + level - 1] = 1

        return a

    def prepare_item(self, imsg):
        i = torch.zeros((ItemState.Size,))
        f = ItemState

        # ITEM_SLOT_TYPE_INVALID
        # ITEM_SLOT_TYPE_MAIN
        # ITEM_SLOT_TYPE_BACKPACK
        # ITEM_SLOT_TYPE_STASH

        i[f.Inventory] = imsg['slot']
        i[f.BackPack] = imsg['slot']
        i[f.Stash] = imsg['slot']
        i[f.Charges] = imsg['charges']
        i[f.IsCooldown] = imsg['cooldown_remaining'] > 0.001
        i[f.CooldownTime] = imsg['cooldown_remaining']
        i[f.IsDisabled] = not imsg['is_activated']
        i[f.SwapCoolDown] = 0
        i[f.ToggledState] = imsg['is_toggled']
        i[f.Locked] = imsg['is_combined_locked']

        i[f.StateStr + imsg['power_treads_stat']] = 1

        return i

    def prepare_hero(self, msg):
        phero = Player(ally=msg['team_id'] == self.faction)
        phero.unit = self.prepare_unit(msg)
        phero.hero = self.prepare_hero_unit(msg)

        if msg['team_id'] == self.faction:
            phero.ally = self.prepare_ability(msg)

        return phero

    def generic_apply(self, delta):
        self.common = self.prepare_common(delta)
        self.proximities.reset()

        players = dict()
        self.process_trees(delta)

        for rune in delta.get('rune_infos', []):
            rid = self.proximities.runes.get_entity(
                rune['location']['x'],
                rune['location']['y']
            )
            self.runes[rid] = self.prepare_rune(rune)

        for player in delta.get('players', []):
            pid = player['player_id']
            players[pid] = player

        for tp in delta.get('incoming_teleports', []):
            pid = tp['player']
            players[pid]['tp'] = tp

        for unit in delta.get('units', []):
            uid = unit['handle']
            player = players.get(uid, None)

            if player is not None:
                player[uid]['unit'] = unit
                self.process_hero(unit)

                hu = self.prepare_hero(player[uid])
                self.heroes[uid] = hu
                continue

            tu = self.prepare_unit(unit)
            self.units[uid] = tu


if __name__ == '__main__':
    a = default_buffer(16)()

    a.append(1)
    a.append(2)

    print(a.to_list()[-1])
