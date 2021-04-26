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


class Dota2DraftEnv(gym.Env):
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

    @property
    def reserved_offsets(self):
        return list(self.booked_id)

    def _load_phase(self, version):
        for phase, decisions in PICK_BAN_ORDER.get(version, v7_27):
            self.order.extend(decisions)

            for d in decisions:
                self.phase_lookup[d] = phase

    def reset(self, radiant_start=None):
        self.radiant_started = False
        if radiant_start is None:
            radiant_start = random.random() < 0.5
            self.radiant_started = True

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

    @property
    def phase(self):
        return self.phase_lookup[self.decision]

    @property
    def decision(self):
        return self.order[self.phase_step]

    def render(self, mode=None):
        pass

    def step(self, action):
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
                    print(f'[BAN] Applying penalty: {action[0]}')

                self.booked_id.add(action[0])
                self.tracker.ban(team, action[0])
            elif dec[0] == 'P':
                if action[1] in self.booked_id:
                    reward.value += self.bad_order_penalty
                    print(f'[PICK] Applying penalty: {action[0]}')

                self.booked_id.add(action[1])
                self.tracker.pick(team, action[1])
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
