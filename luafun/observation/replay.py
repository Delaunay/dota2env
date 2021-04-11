import os
import json

from luafun.observation.stitcher import Stitcher
from luafun.game.dota2.shared import DOTA_GameState


class ObservationReplay:
    def __init__(self, filename):
        self.filename = filename
        self.data = {
            2: [],
            3: []
        }

        faction_map = {
            'RAD': 2,
            'DIRE': 3
        }

        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, filename), 'r') as f:
            for line in f:
                fac, msg = line.split(',', maxsplit=1)
                msg = json.loads(msg)
                self.data[faction_map[fac]].append(msg)


class ObservationFactionReplay:
    def __init__(self, faction, replay):
        self.faction = faction
        self.replay = replay
        self.n = len(self.replay.data[self.faction])

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        for msg in self.replay.data[self.faction]:
            yield msg


replays = dict()


def load_replay(filename) -> ObservationReplay:
    replay = replays.get(filename)

    if not replay:
        replay = ObservationReplay(filename)
        replays[filename] = replay

    return replay


def faction_replay(faction, filename):
    return ObservationFactionReplay(faction, load_replay(filename))


def dire_replay(filename):
    return faction_replay(3, filename)


def radiant_replay(filename):
    return faction_replay(2, filename)


def faction_stitcher(faction, filename, players):
    replay = faction_replay(faction, filename)
    stitcher = Stitcher(faction)

    for msg in replay:
        stitcher.apply_diff(msg)

        if msg.get('game_state', 0) == DOTA_GameState.DOTA_GAMERULES_STATE_GAME_IN_PROGRESS:
            state = stitcher.generate_batch(players)
            yield state


def dire_sticher(filename):
    return faction_stitcher(3, filename, [5, 6, 7, 8, 9])


def radiant_sticher(filename):
    return faction_stitcher(2, filename, [0, 1, 2, 3, 4])
