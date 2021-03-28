"""
This module process the world state delta sent by Dota2 Game
and stitch together a consistent state to be observed by the bots
"""

from collections import defaultdict
from dataclasses import dataclass
from math import cos, sin, sqrt

import torch

import luafun.game.dota2.state_types as msg
from luafun.utils.ring import RingBuffer
from luafun.proximity import ProximityMapper
from luafun.observation.observations import CommonState, ItemState, AbilityState, RuneState
from luafun.observation.observations import UnitState, HeroUnit, AllyHeroState, ModifierState
from luafun.game.ipc_send import TEAM_DIRE, TEAM_RADIANT
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


def buildings(faction):
    """Returns a list of building we track

    Examples
    --------
    Allied building comes first and enemies last
    >>> buildings(TEAM_RADIANT)[:2]
    ['npc_dota_goodguys_tower1_bot', 'npc_dota_goodguys_tower1_mid']

    >>> buildings(TEAM_RADIANT)[-2:]
    ['npc_dota_badguys_tower4_2', 'npc_dota_badguys_fort']

    >>> buildings(TEAM_DIRE)[:2]
    ['npc_dota_badguys_tower1_top', 'npc_dota_badguys_tower1_mid']

    >>> buildings(TEAM_DIRE)[-2:]
    ['npc_dota_goodguys_tower4_2', 'npc_dota_goodguys_fort']

    """
    if faction == TEAM_RADIANT:
        me = 'goodguys'
        enemy = 'badguys'
        safe = 'bot'
        off = 'top'
        enemy_outpost = '#DOTA_OutpostName_North'
        outpost = '"#DOTA_OutpostName_South"'

    elif faction == TEAM_DIRE:
        me = 'badguys'
        enemy = 'goodguys'
        safe = 'top'
        off = 'bot'
        outpost = '#DOTA_OutpostName_North'
        enemy_outpost = '"#DOTA_OutpostName_South"'

    else:
        raise RuntimeError('Bad faction')

    b = [
        f'npc_dota_{me}_tower1_{safe}',
        f'npc_dota_{me}_tower1_mid',
        f'npc_dota_{me}_tower1_{off}',
        f'npc_dota_{me}_tower2_{safe}',
        f'npc_dota_{me}_tower2_mid',
        f'npc_dota_{me}_tower2_{off}',
        outpost,
        f'npc_dota_{me}_tower3_{safe}',
        f'npc_dota_{me}_tower3_mid',
        f'npc_dota_{me}_tower3_{off}',
        f'npc_dota_{me}_melee_rax_{safe}',
        f'npc_dota_{me}_melee_rax_mid',
        f'npc_dota_{me}_melee_rax_{off}',
        f'npc_dota_{me}_range_rax_{safe}',
        f'npc_dota_{me}_range_rax_mid',
        f'npc_dota_{me}_range_rax_{off}',
        f'npc_dota_{me}_tower4_1',
        f'npc_dota_{me}_tower4_2',
        f'npc_dota_{me}_fort',

        f'npc_dota_{enemy}_tower1_{safe}',
        f'npc_dota_{enemy}_tower1_mid',
        f'npc_dota_{enemy}_tower1_{off}',
        f'npc_dota_{enemy}_tower2_{safe}',
        f'npc_dota_{enemy}_tower2_mid',
        f'npc_dota_{enemy}_tower2_{off}',
        enemy_outpost,
        f'npc_dota_{enemy}_tower3_{safe}',
        f'npc_dota_{enemy}_tower3_mid',
        f'npc_dota_{enemy}_tower3_{off}',
        f'npc_dota_{enemy}_melee_rax_{safe}',
        f'npc_dota_{enemy}_melee_rax_mid',
        f'npc_dota_{enemy}_melee_rax_{off}',
        f'npc_dota_{enemy}_range_rax_{safe}',
        f'npc_dota_{enemy}_range_rax_mid',
        f'npc_dota_{enemy}_range_rax_{off}',
        f'npc_dota_{enemy}_tower4_1',
        f'npc_dota_{enemy}_tower4_2',
        f'npc_dota_{enemy}_fort',
    ]
    # = 36 Permanent buildings
    assert len(set(b)) == 38
    return b


MAX_UNIT_COUNT = 100


def hero_order(faction, bid):
    """Returns the hero order for the controlling bots

    Examples
    --------
    We are generating the observation for player 1 (radiant)
    so it will appear first, then its teammate and last the enemies
    >>> hero_order(TEAM_RADIANT, 1)
    [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]

    We are generating the observation for player 9
    so it will appear first, then its teammate and last the enemies
    >>> hero_order(TEAM_DIRE, 9)
    [9, 6, 7, 8, 5, 0, 1, 2, 3, 4]

    """
    rad =  [0, 1, 2, 3, 4]
    dire = [5, 6, 7, 8, 9]

    if faction == TEAM_RADIANT:
        horder = rad + dire
        offset = 0

    elif faction == TEAM_DIRE:
        horder = dire + rad
        offset = 5

    else:
        raise RuntimeError('Bad faction')

    horder[0], horder[bid - offset] = horder[bid - offset], horder[0]
    return horder


def distance(x, y, px, py):
    return sqrt((x - px) ** 2 + (y - py) ** 2)


class Stitcher:
    """Stitcher is a class that tracks game state and generate intermediate tensor to
    help us build the player specific observation tensor
    """
    def __init__(self, faction):
        self.roshan_death_count = 0
        self.faction = faction
        self.minimap = None
        self.common = None
        self.draft = None
        self.latest_message = None
        self.heroes = {
            0: Player(ally=TEAM_RADIANT == self.faction),
            1: Player(ally=TEAM_RADIANT == self.faction),
            2: Player(ally=TEAM_RADIANT == self.faction),
            3: Player(ally=TEAM_RADIANT == self.faction),
            4: Player(ally=TEAM_RADIANT == self.faction),
            5: Player(ally=TEAM_DIRE == self.faction),
            6: Player(ally=TEAM_DIRE == self.faction),
            7: Player(ally=TEAM_DIRE == self.faction),
            8: Player(ally=TEAM_DIRE == self.faction),
            9: Player(ally=TEAM_DIRE == self.faction),
        }

        self.units = dict()
        self.runes = dict()
        self.health_tracker = defaultdict(default_buffer(16))
        self.building_names = buildings(faction)
        self.buildings = dict()
        self.building_order = []

        # Proximity mapping
        self.proximities = ProximityMapper()
        self._size = Stitcher.observation_size(faction)

    def get_entities(self, x, y):
        return self.proximities.entities(x, y)

    @staticmethod
    def observation_size(faction):
        """Returns the observation space that we are stitching"""
        return (
                Player.size(False) * 5 +
                Player.size(True) * 5 +
                Unit.size() * len(buildings(faction)) +
                Unit.size() * MAX_UNIT_COUNT +
                RuneState.Size * 6 +
                CommonState.Size
        )

    @property
    def observation_space(self):
        """Returns the observation space that we are stitching"""
        return self._size

    def apply_diff(self, delta: msg.CMsgBotWorldState):
        """Take a world state delta and apply it to a previous state"""
        import time

        self.latest_message = delta
        self.generic_apply(delta)
        e = time.time()

    def generate_batch(self, botids) -> torch.Tensor:
        """Generate an observation for a set of players"""
        state = torch.zeros((len(botids), self.observation_space,))

        for i, pid in enumerate(botids):
            state[i, :] = self.generate_player(pid)

        return state

    def generate_player(self, bid) -> torch.Tensor:
        """Generate an observation for a given player"""
        state = torch.zeros((self.observation_space,))
        s = 0
        e = 0

        player = self.heroes.get(bid)
        if player is None:
            return state

        player = player.unit
        px, py = player[UnitState.X], player[UnitState.Y]

        s = e
        e = s + CommonState.Size
        state[s:e] = self.common

        horder = hero_order(self.faction, bid)

        for i in range(6):
            s = e
            e = s + RuneState.Size
            rune = self.runes.get(i)

            # set hero location
            for i, hid in enumerate(horder):
                hu = self.heroes[hid].unit

                hpos = hu[UnitState.X], hu[UnitState.Y]
                rune[RuneState.DistanceH0 + i] = distance(
                    hpos[0],
                    hpos[1],
                    rune[RuneState.LocationX],
                    rune[RuneState.LocationY])

            state[s:e] = rune

        # Set building info
        for i, uid in self.building_order:
            unit = self.buildings.get(i)

            if unit is None:
                continue

            x, y = unit[UnitState.X], unit[UnitState.Y]
            dist = distance(x, y, px, py)

            unit[UnitState.ToCurrentUnitdx] = x - px
            unit[UnitState.ToCurrentUnitdy] = y - py
            unit[UnitState.ToCurrentUnitl] = dist

            # IsCurrentHeroAttackingIt
            # IsAttackingCurrentHero
            # EtaProjectileToHero

            s = e
            e = s + UnitState.Size

            state[s:e] = unit
        # ---

        # set hero info
        for hid in horder:
            hero: Player = self.heroes[hid]

            s = e
            e = s + Player.size(hero.is_ally)
            state[s:e] = hero.tensor()

        # Set closest unit first
        closest = []
        for k, unit in self.units.items():
            x, y = unit[UnitState.X], unit[UnitState.Y]
            dist = sqrt((x - px) ** 2 + (y - py) ** 2)
            closest.append((k, dist))

        closest.sort()

        for i, (k, dist) in enumerate(closest):
            if i > MAX_UNIT_COUNT:
                break

            unit = self.units[k]
            x, y = unit[UnitState.X], unit[UnitState.Y]

            unit[UnitState.ToCurrentUnitdx] = x - px
            unit[UnitState.ToCurrentUnitdy] = y - py
            unit[UnitState.ToCurrentUnitl] = dist

            # IsCurrentHeroAttackingIt
            # IsAttackingCurrentHero
            # EtaProjectileToHero

            s = e
            e = s + UnitState.Size
            state[s:e] = unit

        return state

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

    def prepare_unit(self, msg, modifier_count) -> torch.Tensor:
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

        # BOT_ACTION_TYPE_NONE = 0
        # BOT_ACTION_TYPE_IDLE = 1
        # BOT_ACTION_TYPE_MOVE_TO = 2
        # BOT_ACTION_TYPE_MOVE_TO_DIRECTLY = 3
        # BOT_ACTION_TYPE_ATTACK = 4
        # BOT_ACTION_TYPE_ATTACKMOVE = 5
        # BOT_ACTION_TYPE_USE_ABILITY
        # BOT_ACTION_TYPE_PICK_UP_RUNE
        # BOT_ACTION_TYPE_PICK_UP_ITEM
        # BOT_ACTION_TYPE_DROP_ITEM
        # BOT_ACTION_TYPE_SHRINE
        # BOT_ACTION_TYPE_DELAY

        self.proximities.manager.update_position(uid, pos[0], pos[1])
        u[f.X] = pos[0]
        u[f.Y] = pos[1]
        u[f.Z] = pos[2]
        u[f.FacingCos] = cos(msg['facing'])
        u[f.FacingSin] = sin(msg['facing'])

        has_target = msg.get('ability_target_handle', None) is not None
        is_attacking = msg.get('action_type', -1) in (4, 5)

        u[f.IsAttacking] = has_target or is_attacking
        u[f.TimeSinceLastAttack] = msg.get('last_attack_time', 0)
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
        # This should be in the modifier list
        # u[f.IsGlyphed] = 0
        # u[f.GlyphTime] = 0
        u[f.MovementSpeed] = msg['current_movement_speed']
        u[f.IsAlly] = msg['team_id'] == self.faction
        u[f.IsNeutral] = msg['team_id'] == 1
        u[f.AnimationCycleTime] = msg['anim_cycle']

        min_eta = 10
        for track in msg.get('incoming_tracking_projectiles', []):
            x, y = track['location']['x'], track['location']['y']
            dist = sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            eta = dist / track['velocity']
            min_eta = min(eta, min_eta)

        u[f.EtaIncomingProjectile] = min_eta

        # Needs more computations to get those
        # u[f.NumberOfUnitAttackingMe] = 0
        # u[f.ShrineCooldown        ] = 0
        # u[f.ToCurrentUnitdx] = 0
        # u[f.ToCurrentUnitdy] = 0
        # u[f.ToCurrentUnitl] = 0
        # u[f.IsCurrentHeroAttackingIt] = 0
        # u[f.IsAttackingCurrentHero] = 0
        # u[f.EtaProjectileToHero] = 0

        u[f.UnitTypeHERO + offset] = 1

        # modifier_count

        return u

    def prepare_rune(self, rmsg):
        r = torch.zeros((RuneState.Size,))
        f = RuneState

        # RUNE_STATUS_UNKNOWN = 0
        RUNE_STATUS_AVAILABLE = 1
        # RUNE_STATUS_MISSING = 2

        pos = rmsg['location']['x'], rmsg['location']['y']

        r[f.Visible] = rmsg['status'] == RUNE_STATUS_AVAILABLE
        r[f.LocationX] = pos[0]
        r[f.LocationY] = pos[1]

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
        h[f.NumberOfDeath] = pmsg.get('deaths', 0)
        h[f.HeroInSight] = 1
        h[f.LastSeenTime] = 0
        h[f.IsTeleporting] = tpitem.get('is_channeling', False)
        h[f.TeleportTargetX] = tpitem.get('location', dict()).get('x', -1)
        h[f.TeleportTargetY] = tpitem.get('location', dict()).get('x', -1)
        h[f.TeleportChannelTime] = tpitem.get('time_remaining', 0)
        h[f.CurrentGold] = umsg.get('reliable_gold', 0) + umsg.get('unreliable_gold', 0)
        h[f.Level] = umsg['level'] / 30
        h[f.ManaMax] = umsg['mana_max']
        h[f.Mana] = umsg['mana']
        h[f.ManaRegen] = umsg['mana_regen']
        h[f.HealthMax] = umsg['health_max']
        h[f.Health] = umsg['health']
        h[f.HealthRegen] = umsg['health_regen']
        h[f.MagicResitance] = umsg['magic_resist']
        h[f.Strength] = umsg.get('strength', 0)
        h[f.Agility] = umsg.get('agility', 0)
        h[f.Intelligence] = umsg.get('intelligence', 0)
        h[f.IsInvisibility] = umsg['is_invisible']
        h[f.IsUsingAbility] = umsg.get('is_using_ability', False)
        h[f.NumberOfAllied] = 0
        h[f.NumberOfAlliedCreeps] = 0
        h[f.NumberOfEnemy] = 0
        h[f.NumberOfEnemyCreeps] = 0

        return h

    def process_hero(self, umsg):
        pitems = dict()
        umsg['pitems'] = pitems

        for item in umsg.get('items', []):
            pitems[item['slot']] = item

        pabi = dict()
        umsg['pabi'] = pabi

        for item in umsg.get('abilities', []):
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

        a[f.Cooldown] = amsg.get('cooldown_remaining', 0)
        a[f.InUse] = amsg.get('is_in_ability_phase', False) or amsg.get('is_channeling', False)
        a[f.Castable] = amsg.get('is_fully_castable', False)

        level = amsg.get('level', 0)

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
        phero.unit = self.prepare_unit(msg['unit'], modifier_count=10)
        phero.hero = self.prepare_hero_unit(msg)

        if msg['team_id'] == self.faction:
            phero.ally = self.prepare_ability(msg)

        return phero

    def init_buildings(self, msg):
        if len(self.buildings) == len(self.building_names):
            return

        self.building_order = []
        self.buildings = dict()

        # we need to track idx because of T4 towers
        # there are 2 towers with the same name
        idx = set()

        for unit in msg.get('units', []):
            uid = unit['handle']
            name = unit.get('name')

            for i, bname in enumerate(self.building_names):

                if bname.startswith(name) and i not in idx:
                    idx.add(i)

                    tu = self.prepare_unit(unit, modifier_count=2)
                    self.buildings[uid] = tu
                    self.building_order.append((i, uid))

        # Sort by index order
        self.building_order.sort(key=lambda x: x[0])

    def generic_apply(self, delta):
        self.common = self.prepare_common(delta)
        self.proximities.reset()

        players = defaultdict(dict)
        self.process_trees(delta)

        for rune in delta.get('rune_infos', []):
            rid = self.proximities.runes.get_entity(
                rune['location']['x'],
                rune['location']['y']
            )
            self.runes[rid] = self.prepare_rune(rune)

        # projectile (arrows, hook)
        # for linear in delta.get('linear_projectiles', []):
        #     pass

        # not supported yet
        # DroppedItem
        # EventAbility
        # EventDamage
        # EventCourierKilled

        # insert player info into unit for easier tensor building
        for player in delta.get('players', []):
            pid = player['player_id']
            players[pid] = player

        for tp in delta.get('incoming_teleports', []):
            pid = tp['player_id']
            players[pid]['tp'] = tp

        # enum
        # UnitType
        # {
        # INVALID = 0;
        # HERO = 1;
        # CREEP_HERO = 2;
        # LANE_CREEP = 3;
        # JUNGLE_CREEP = 4;
        # ROSHAN = 5;
        # TOWER = 6;
        # BARRACKS = 7;
        # SHRINE = 8;
        # FORT = 9;
        # BUILDING = 10;
        # COURIER = 11;
        # WARD = 12;
        # }

        self.init_buildings(delta)

        # Main builder
        for unit in delta.get('units', []):
            uid = unit['handle']

            player = None
            if unit.get('unit_type', 0) == 1:
                pid = unit.get('player_id', -1)
                player = players.get(pid, None)

            if player is not None:
                player['unit'] = unit
                pid = unit['player_id']

                self.process_hero(unit)
                hu = self.prepare_hero(player)
                self.heroes[uid] = hu
                self.heroes[pid] = hu
                continue

            # Standard unit
            tu = self.prepare_unit(unit, modifier_count=2)

            if uid in self.buildings:
                self.buildings[uid] = tu
                continue

            self.units[uid] = tu


def print_state(tensor):
    return TensorInterpret().print(tensor)


class TensorInterpret:
    SIZE = Stitcher.observation_size(TEAM_DIRE)

    def print_hero(self, s, e, tensor, ally, i=None):
        for k in range(HERO_ITEMS):
            s = e
            e = s + ItemState.Size
            ItemState.print(tensor[s:e], i=k)

        for k in range(HERO_ABILITIES):
            s = e
            e = s + AbilityState.Size
            AbilityState.print(tensor[s:e], i=k)

        for k in range(HERO_MODIFIER):
            s = e
            e = s + ModifierState.Size
            ModifierState.print(tensor[s:e], i=k)

        if ally:
            s = e
            e = s + AllyHeroState.Size
            AllyHeroState.print(tensor[s:e], i=i)

        return s, e

    def print(self, tensor, j=None):
        s = 0
        e = s + CommonState.Size
        CommonState.print(tensor[s:e], i=j)

        for i in range(6):
            s = e
            e = s + RuneState.Size
            RuneState.print(tensor[s:e], i=i)

        # buildings
        for i in range(len(buildings(TEAM_RADIANT))):
            s = e
            e = s + UnitState.Size
            UnitState.print(tensor[s:e], i=i)

            for i in range(UNIT_MODIFIER):
                s = e
                e = s + ModifierState.Size
                ModifierState.print(tensor[s:e], i=i)

        # Current Hero
        s, e = self.print_hero(s, e, tensor, True, i=-1)

        # Allies
        for i in range(4):
            s, e = self.print_hero(s, e, tensor, True, i=i)

        # Enemies
        for i in range(5):
            s, e = self.print_hero(s, e, tensor, False, i=i)

        # Units
        for i in range(MAX_UNIT_COUNT):
            s = e
            e = s + UnitState.Size
            UnitState.print(tensor[s:e], i=i)

            for k in range(UNIT_MODIFIER):
                s = e
                e = s + ModifierState.Size
                ModifierState.print(tensor[s:e], i=k)


if __name__ == '__main__':
    a = default_buffer(16)()

    a.append(1)
    a.append(2)

    print(a.to_list()[-1])
