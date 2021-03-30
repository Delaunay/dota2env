from collections import defaultdict

from luafun.game.dota2.state_types import Player, Unit
from luafun.game.ipc_send import TEAM_DIRE, TEAM_RADIANT

Solo = 0
Team = 1

#     Name                 , Reward    , Type
values = [
    ('Win'                ,    5      , Team),
    ('HeroDeath'          ,   -1      , Solo),
    ('CourierDeath'       ,   -2      , Solo),
    ('XPGained'           ,    0.002  , Solo),
    ('GoldGained'         ,    0.006  , Solo),
    ('GoldSpent'          ,    0.0006 , Solo),
    ('HealthChanged'      ,    2      , Solo), # %of health = ( x + 1 - (1 - x) ^ 4) / 2
    ('ManaChanged'        ,    0.75   , Solo),
    ('KilledHero'         ,   -0.6    , Solo),
    ('LastHit'            ,   -0.16   , Solo), # Reduce the reward since we already got a bit chunk from exp & gold
    ('Deny'               ,    0.15   , Solo),
    ('GainedAegis'        ,    5      , Team),
    ('AncientHPChange'    ,    5      , Team),
    ('MegasUnlocked'      ,    4      , Team),
    ('T1Tower'            ,    2.25   , Team),
    ('T2Tower'            ,    3      , Team),
    ('T3Tower'            ,    4.5    , Team),
    ('T4Tower'            ,    2.25   , Team), # 2/3 = building health + 1/3 on destroy
    ('Outpost'            ,    2.25   , Team),
    ('Barracks'           ,    6      , Team), # 2/3 = building health + 1/3 on destroy
    ('LaneAssign'         ,   -0.15   , Solo),
]


class RewardConst:
    Win                =  5
    HeroDeath          = -1
    CourierDeath       = -2
    XPGained           =  0.002
    GoldGained         =  0.006
    GoldSpent          =  0.0006
    HealthChanged      =  2
    ManaChanged        =  0.75
    KilledHero         = -0.6
    LastHit            = -0.16
    Deny               =  0.15
    GainedAegis        =  5
    AncientHPChange    =  5
    MegasUnlocked      =  4
    T1Tower            =  2.25
    T2Tower            =  3
    T3Tower            =  4.5
    T4Tower            =  2.25
    Outpost            =  2.25
    Barracks           =  6
    LaneAssign         = -0.15

    # Implicitly here with gold & exp gain
    # HeroAssist


# See https://dota2.fandom.com/wiki/Experience
TotalExp = 0
ExpNeeded = 1
ExperienceTable = {
    1:  [    0, 	230],
    2:  [  230, 	370],
    3:  [  600, 	480],
    4:  [ 1080, 	580],
    5:  [ 1660, 	600],
    6:  [ 2260, 	720],
    7:  [ 2980, 	750],
    8:  [ 3730, 	890],
    9:  [ 4620, 	930],
    10: [ 5550, 	970],
    11: [ 6520, 	101],
    12: [ 7530, 	105],
    13: [ 8580, 	122],
    14: [ 9805, 	125],
    15: [11055, 	127],
    16: [12330, 	130],
    17: [13630, 	132],
    18: [14955, 	150],
    19: [16455, 	159],
    20: [18045, 	160],
    21: [19645, 	185],
    22: [21495, 	210],
    23: [23595, 	235],
    24: [25945, 	260],
    25: [28545, 	350],
    26: [32045, 	450],
    27: [36545, 	550],
    28: [42045, 	650],
    29: [48545, 	750],
    30: [56045, 	  0],
}


class Reward:
    """Base reward function, takes a state and return its resulting reward level"""
    def __init__(self):
        self.rewards = defaultdict(float)
        self.courier_state = [
            1 for _ in range(0, 10)
        ]
        self.courier_death_tracker = [
            0 for _ in range(0, 10)
        ]

    def player_message(self, pmsg: Player, umsg: Unit, courier: Unit):
        """Computes the 'solo' reward"""
        pid = pmsg['player_id']

        kills = pmsg.get('kills', 0)
        deaths = pmsg.get('deaths', 0)

        current_gold = umsg.get('reliable_gold', 0) + umsg.get('unreliable_gold', 0)
        spent_gold = umsg.get('net_worth', 0) - current_gold

        xp_left = umsg.get('xp_needed_to_level', 0)
        level = umsg['level']
        needed = ExperienceTable[level][ExpNeeded]

        xp_gained = ExperienceTable[level][TotalExp] + (needed - xp_left)

        # ==
        health_max = umsg['health_max']
        health = umsg['health']
        health_pct = health / health_max
        health_reward = (health_pct + 1 - (1 - health_pct) ** 4) / 2
        # ====

        mana_max = umsg['mana_max']
        mana = umsg['mana']
        mana_pct = mana / mana_max

        # ===
        if courier is not None:
            courier_alive = courier['is_alive']
            if self.courier_state[pid] != courier_alive:
                if not courier_alive:
                    self.courier_death_tracker[pid] += 1

                self.courier_state[pid] = courier_alive
        # ===

        reward = (
            RewardConst.KilledHero    * kills +
            RewardConst.HeroDeath     * deaths +
            RewardConst.Deny          * umsg['denies'] +
            RewardConst.LastHit       * umsg['last_hits'] +
            RewardConst.GoldSpent     * spent_gold +
            RewardConst.GoldGained    * current_gold +
            RewardConst.XPGained      * xp_gained +
            RewardConst.ManaChanged   * mana_pct +
            RewardConst.HealthChanged * health_reward +
            RewardConst.CourierDeath  * self.courier_death_tracker[pid] +
            # FIXME: This requires us to define the area of the lanes
            # few rectangles
            RewardConst.LaneAssign * 0
        )

        self.rewards[pid] = reward

    def building_messages(self, umsg: Unit):
        reward = 0

        # ==
        # Destroyed enemy buildings increase our rewards
        team_id = umsg['team_id']
        if team_id == TEAM_RADIANT:
            team_id = TEAM_DIRE
        else:
            team_id = TEAM_RADIANT
        # ==

        def building_reward(value, unit):
            return value * (0.66 * (1 - unit['health'] / unit['health_max']) + 0.34 * (1 - unit['is_alive']))

        if '_tower1_' in umsg['name']:
            reward += building_reward(RewardConst.T1Tower, umsg)

        elif '_tower2_' in umsg['name']:
            reward += building_reward(RewardConst.T2Tower, umsg)

        elif '_tower3_' in umsg['name']:
            reward += building_reward(RewardConst.T3Tower, umsg)

        elif '_tower4_' in umsg['name']:
            reward += building_reward(RewardConst.T4Tower, umsg)

        elif '_rax_' in umsg['name']:
            reward += building_reward(RewardConst.Barracks, umsg)
            self.rewards[200 + team_id] += 1 - int(umsg['is_alive'])

        elif '_fort' in umsg['name']:
            reward += RewardConst.AncientHPChange * (1 - umsg['health'] / umsg['health_max'])
            reward += RewardConst.Win * (1 - umsg['is_alive'])

        elif '_OutpostName' in umsg['name']:
            # Taking the outpost increase our rewards
            reward += RewardConst.Outpost
            team_id = umsg['team_id']

        # reward = (
        #     RewardConst.GainedAegis     * 0 +
        # )

        self.rewards[100 + team_id] += reward

    def clear(self):
        self.rewards[100 + TEAM_RADIANT] = 0
        self.rewards[100 + TEAM_DIRE] = 0
        self.rewards[200 + TEAM_RADIANT] = 0
        self.rewards[200 + TEAM_DIRE] = 0

    def partial_radiant_reward(self) -> float:
        reward = self.rewards[100 + TEAM_RADIANT] + (self.rewards[200 + TEAM_RADIANT] // 6) * RewardConst.MegasUnlocked

        for i in [0, 1, 2, 3, 4]:
            reward += self.rewards[i]

        return reward

    def partial_dire_reward(self) -> float:
        # OpenAI probably use a different reward for each bots and tweak the solo/team
        # proportion using the `team_spirit` params
        # I do not like that this is a cooperation game and support can have a positive impact
        # on the NW of another hero on the team and the reward should reflect that
        # because of the change the reward value should probably be tweaked
        # the solo reward should also diminish through time
        reward = self.rewards[100 + TEAM_DIRE] + (self.rewards[200 + TEAM_RADIANT] // 6) * RewardConst.MegasUnlocked

        for i in [5, 6, 7, 8, 9]:
            reward += self.rewards[i]

        return reward

    def dire_reward(self) -> float:
        r = self.partial_dire_reward() - self.partial_radiant_reward()
        return r

    def radiant_reward(self) -> float:
        r = self.partial_radiant_reward() - self.partial_dire_reward()
        return r
