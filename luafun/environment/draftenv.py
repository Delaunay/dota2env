from luafun.drafter import DraftTracker
import luafun.game.constants as const

import numpy as np
import gym

v7_27 = {
    # Assume Radiant is Starting Team
    # B1: Banning Phase 1
    # P1: Picking Phase 1
    # BR: Ban Radiant
    # BD: Ban Dire
    # PR: Pick Radiant
    # PD: Pick Dire
    'B1': ['BR', 'BD', 'BR', 'BD'],
    'P1': ['PR', 'PD', 'PD', 'PR'],
    'B2': ['BR', 'BD', 'BR', 'BD', 'BR', 'BD'],
    'P2': ['PD', 'PR', 'PD', 'PR'],
    'B3': ['BR', 'BD', 'BR', 'BD'],
    'P3': ['PR', 'PD'],
}

assert len(v7_27) == 24
PICK_BAN_ORDER = {
    '7.27': v7_27,
    '7.28': v7_27,
    '7.29': v7_27,
}


class Dota2DraftEnv(gym.Env):
    def __init__(self):
        self.tracker = DraftTracker()

    def reset(self):
        pass

    def render(self, mode=None):
        pass

    def step(self, action):
        pass

    def close(self):
        pass

    @property
    def action_space(self):
        return gym.spaces.Dict({
            'pick': gym.spaces.Discrete(const.HERO_COUNT),
            'ban': gym.spaces.Discrete(const.HERO_COUNT),
        })

    @property
    def observation_space(self):
        # this does not sample the correct values
        # and multi-discrete is ok but not correct shape
        return gym.spaces.Box(0, 1, shape=(24, const.HERO_COUNT), dtype=np.int8)
        # return gym.spaces.MultiDiscrete([const.HERO_COUNT for i in range(24)])


if __name__ == '__main__':
    print(Dota2DraftEnv().observation_space.sample())

    print(Dota2DraftEnv().action_space.sample())
