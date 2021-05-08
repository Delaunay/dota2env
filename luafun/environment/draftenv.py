from luafun.drafter import DraftTracker
import luafun.game.constants as const
from luafun.game.ipc_send import TEAM_DIRE, TEAM_RADIANT

import random

import numpy as np
import gym

v7_27 = [
    # Assume Radiant is Starting Team
    # B1: Banning Phase 1
    # P1: Picking Phase 1
    # BR: Ban Radiant
    # BD: Ban Dire
    # PR: Pick Radiant
    # PD: Pick Dire
    ('B1', ['BR', 'BD', 'BR', 'BD']),
    ('P1', ['PR', 'PD', 'PD', 'PR']),
    ('B2', ['BR', 'BD', 'BR', 'BD', 'BR', 'BD']),
    ('P2', ['PD', 'PR', 'PD', 'PR']),
    ('B3', ['BR', 'BD', 'BR', 'BD']),
    ('P3', ['PR', 'PD']),
]

assert len(v7_27) == 6
PICK_BAN_ORDER = {
    '7.27': v7_27,
    '7.28': v7_27,
    '7.29': v7_27,
}


def radiant_ban(hero_id):
    return (None, hero_id), None


def radiant_pick(hero_id):
    return (hero_id, None), None


def dire_ban(hero_id):
    return None, (None, hero_id)


def dire_pick(hero_id):
    return None, (hero_id, None)


class Dota2DraftEnv(gym.Env):
    """Dota2 Drafting environment

    Examples
    --------
    >>> import luafun.game.constants as const

    >>> env = Dota2DraftEnv(radiant_start=True)
    >>> env.decision_human
    'Radiant ban'

    >>> ban = const.HERO_LOOKUP.from_name('npc_dota_hero_arc_warden')['offset']
    >>> state, reward, done, info = env.step(radiant_ban(ban))
    >>> env.summary()
            Ban01: Arc Warden

    >>> env.decision_human
    'Dire ban'

    >>> ban = const.HERO_LOOKUP.from_name('npc_dota_hero_winter_wyvern')['offset']
    >>> state, reward, done, info = env.step(dire_ban(ban))
    >>> env.summary()
            Ban01: Arc Warden
            Ban02: Winter Wyvern

    >>> env.decision_human
    'Radiant ban'

    >>> ban = const.HERO_LOOKUP.from_name('npc_dota_hero_techies')['offset']
    >>> state, reward, done, info = env.step(radiant_ban(ban))
    >>> env.summary()
            Ban01: Arc Warden
            Ban02: Winter Wyvern
            Ban03: Techies

    >>> env.decision_human
    'Dire ban'

    >>> ban = const.HERO_LOOKUP.from_name('npc_dota_hero_phoenix')['offset']
    >>> state, reward, done, info = env.step(dire_ban(ban))
    >>> env.summary()
            Ban01: Arc Warden
            Ban02: Winter Wyvern
            Ban03: Techies
            Ban04: Phoenix

    >>> env.decision_human
    'Radiant pick'

    >>> pick = const.HERO_LOOKUP.from_name('npc_dota_hero_oracle')['offset']
    >>> state, reward, done, info = env.step(radiant_pick(pick))
    >>> env.summary()
    Radiant Pick0: Oracle
            Ban01: Arc Warden
            Ban02: Winter Wyvern
            Ban03: Techies
            Ban04: Phoenix

    >>> env.decision_human
    'Dire pick'

    >>> pick = const.HERO_LOOKUP.from_name('npc_dota_hero_dark_willow')['offset']
    >>> state, reward, done, info = env.step(dire_pick(pick))
    >>> env.summary()
    Radiant Pick0: Oracle
       Dire Pick5: Dark Willow
            Ban01: Arc Warden
            Ban02: Winter Wyvern
            Ban03: Techies
            Ban04: Phoenix

    """
    def __init__(self, radiant_start=None, version='7.28'):
        self.radiant_start = radiant_start
        self.tracker: DraftTracker = None
        self.radiant: str = None
        self.dire: str = None

        self.phase_lookup = dict()
        self.order = []
        self._load_phase(version)

        self.phase_step = 0
        self.reset(self.radiant_start)
        self.bad_order_penalty = -0.01
        self.booked_id = set()
        self.done = False
        self.radiant_started = False
        self.log_fun = lambda x: x

    @property
    def reserved_offsets(self):
        """List of heroes already picked or ban that cannot be picked or banned"""
        return list(self.booked_id)

    def _load_phase(self, version):
        for phase, decisions in PICK_BAN_ORDER.get(version, v7_27):
            self.order.extend(decisions)

            for d in decisions:
                self.phase_lookup[d] = phase

    def reset(self, radiant_start=None):
        """Reset the environment"""
        self.radiant_started = False

        if radiant_start is None:
            radiant_start = random.random() < 0.5
            self.radiant_started = radiant_start

        self.tracker = DraftTracker()

        if radiant_start:
            self.radiant = 'R'
            self.dire = 'D'
        else:
            self.radiant = 'D'
            self.dire = 'R'

        self.phase_step = 0
        self.booked_id = set()

        state = self.tracker.as_tensor(TEAM_RADIANT), self.tracker.as_tensor(TEAM_DIRE)
        info = None
        reward = (0, 0)
        return state, reward, False, info

    def __enter__(self):
        self.reset(self.radiant_start)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def summary(self):
        """Print a summary of the draft"""
        self.tracker.draft.summary()

    @property
    def phase(self):
        """Returns the current drafting phase"""
        return self.phase_lookup[self.decision]

    @property
    def decision(self):
        """Returns the current decision that is expected"""
        if self.phase_step == len(self.order):
            return 'Finished'

        return self.order[self.phase_step]

    @property
    def decision_human(self):
        """Returns a string describing the expected action, decode the string returned by ``decision``

        Examples
        --------
        >>> from luafun.environment.draftenv import Dota2DraftEnv
        >>> env = Dota2DraftEnv(radiant_start=True)
        >>> while not env.done:
        ...     radiant, dire = env.action_space.sample()
        ...     print(env.decision_human)
        ...     state, reward, done, info = env.step((radiant, dire))
        Radiant ban
        Dire ban
        Radiant ban
        Dire ban
        Radiant pick
        Dire pick
        Dire pick
        Radiant pick
        Radiant ban
        Dire ban
        Radiant ban
        Dire ban
        Radiant ban
        Dire ban
        Dire pick
        Radiant pick
        Dire pick
        Radiant pick
        Radiant ban
        Dire ban
        Radiant ban
        Dire ban
        Radiant pick
        Dire pick
        Finished
        """
        dec = self.decision
        action = None
        faction = None

        if dec[0] == 'P':
            action = 'pick'
        elif dec[0] == 'B':
            action = 'ban'
        else:
            return dec

        if dec[1] == self.radiant:
            faction = 'Radiant'
        else:
            faction = 'Dire'

        return f'{faction} {action}'

    def render(self, mode=None):
        """Part of Gym API but not used, use summary to print the drafting state"""
        pass

    def step(self, action):
        """Execute one action pick/ban

        Parameters
        ----------
        action: Tuple[Tuple[Pick, Ban], Tuple[Pick, Ban]]
            Tuple of actions

        Notes
        -----
        Only the expected action is taken by the environment.
        i.e if radiant has to make a pick only the pick is taken and everything else is discarded.
        This is meant to simplify developing drafting AIs as they do not need to understand the drafting steps
        to work.

        You can use ``env.decision_human`` to know which action is expected

        The environment does not enforce pick/ban uniqueness but it will apply a penalty to the reward if not respected.
        The penalty can be tweaked using the attribute ``bad_order_penalty``

        Examples
        --------
        >>> env = Dota2DraftEnv(radiant_start=True)
        >>> rad_pick, rad_ban = (1, 2)
        >>> dire_pick, dire_ban = (3, 4)
        >>> env.decision_human
        'Radiant ban'
        >>> (rad_state, dire_state), reward, done, info = env.step(((rad_pick, rad_ban), (dire_pick, dire_ban)))
        """
        radiant_action, dire_action = action

        class Reward:
            def __init__(self):
                self.value = 0

        dire_reward = Reward()
        radiant_reward = Reward()

        self.done = len(self.order) == self.phase_step

        if not self.done:
            dec = self.decision
            self.phase_step += 1

            if dec[1] == self.radiant:
                team = TEAM_RADIANT
                action = radiant_action
                reward = radiant_reward
            elif dec[1] == self.dire:
                team = TEAM_DIRE
                action = dire_action
                reward = dire_reward
            else:
                raise RuntimeError('Unreachable')

            if dec[0] == 'B':
                if action[0] in self.booked_id:
                    reward.value += self.bad_order_penalty
                    self.log_fun(f'[BAN] Applying penalty: {action[1]}')

                self.booked_id.add(action[1])
                self.tracker.ban(team, action[1])
            elif dec[0] == 'P':
                if action[1] in self.booked_id:
                    reward.value += self.bad_order_penalty
                    self.log_fun(f'[PICK] Applying penalty: {action[0]}')

                self.booked_id.add(action[0])
                self.tracker.pick(team, action[0])
            else:
                raise RuntimeError('Unreachable')

        state = self.tracker.as_tensor(TEAM_RADIANT), self.tracker.as_tensor(TEAM_DIRE)
        info = None
        reward = (radiant_reward.value, dire_reward.value)
        return state, reward, self.done, info

    def close(self):
        pass

    @property
    def action_space(self):
        """Returns the expected action space"""
        action = gym.spaces.Tuple((
            gym.spaces.Discrete(const.HERO_COUNT),
            gym.spaces.Discrete(const.HERO_COUNT)
        ))

        return gym.spaces.Tuple((action, action))

    @property
    def observation_space(self):
        # this does not sample the correct values
        # and multi-discrete is ok but not correct shape
        return gym.spaces.Box(0, 1, shape=(24, const.HERO_COUNT), dtype=np.int8)
        # return gym.spaces.MultiDiscrete([const.HERO_COUNT for i in range(24)])


if __name__ == '__main__':
    print(Dota2DraftEnv().observation_space.sample())

    print(Dota2DraftEnv().action_space.sample())

    with Dota2DraftEnv() as env:
        while not env.done:
            radiant, dire = env.action_space.sample()

            state, reward, done, info = env.step((radiant, dire))

        print()
