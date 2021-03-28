import os
import json

from luafun.observation.stitcher import Stitcher, print_state
from luafun.game.dota2.shared import DOTA_GameState


def load_test_data(name, faction):
    data = {
        2: [],
        3: []
    }

    faction_map = {
        'RAD': 2,
        'DIRE': 3
    }

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, name), 'r') as f:
        for line in f:
            fac, msg = line.split(',', maxsplit=1)

            msg = json.loads(msg)

            if msg.get('game_state', 0) != DOTA_GameState. DOTA_GAMERULES_STATE_GAME_IN_PROGRESS:
                continue

            if faction_map[fac] == faction:
                yield msg


def test_stitcher(faction=2):
    stitcher = Stitcher(faction)

    for msg in load_test_data('resources/replay.txt', faction):
        stitcher.apply_diff(msg)
        print(msg)

        state = stitcher.generate_player(0)

        print(state.shape)
        print_state(state)

        break


if __name__ == '__main__':
    test_stitcher()
    pass
