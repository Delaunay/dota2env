import os
import json

from luafun.observation.stitcher import Stitcher


def load_test_data(name):
    data = {
        2: [],
        3: []
    }

    fact = {
        'RAD': 2,
        'DIRE': 3
    }
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, name), 'r') as f:
        for line in f:
            faction, msg = line.split(',', maxsplit=1)
            data[fact[faction]].append(json.loads(msg))

    return data


def test_stitcher(faction=2):
    stitcher = Stitcher(faction)

    for msg in load_test_data('resources/replay.txt')[faction]:
        stitcher.apply_diff(msg)
        print(msg)


if __name__ == '__main__':
    test_stitcher()
    pass
