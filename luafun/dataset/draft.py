from collections import defaultdict
import json
import zipfile

import torch
from torch.utils.data.dataset import Dataset

from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE
from luafun.draft import DraftTracker
import luafun.game.constants as const
from luafun.game.game import DOTA_GameMode


DECISION_COUNT = {
    "6.77": 20,
    "6.78": 20,
    "6.79": 20,
    "6.80": 20,
    "6.81": 20,
    "6.82": 20,
    "6.83": 20,
    "6.84": 20,
    "6.85": 20,
    "6.86": 20,
    "6.87": 20,
    "6.88": 20,
    "7.00": 20,
    "7.01": 20,
    "7.02": 20,
    "7.03": 20,
    "7.04": 20,
    "7.05": 20,
    "7.06": 20,
    "7.07": 22,
    "7.08": 22,
    "7.09": 22,
    "7.10": 22,
    "7.11": 22,
    "7.12": 22,
    "7.13": 22,
    "7.14": 22,
    "7.15": 22,
    "7.16": 22,
    "7.17": 22,
    "7.18": 22,
    "7.19": 22,
    "7.20": 22,
    "7.21": 22,
    "7.22": 22,
    "7.23": 22,
    "7.24": 22,
    "7.25": 22,
    "7.26": 22,
    '7.27': 24,
    '7.28': 24,
    "7.29": 24,
}


class ZipDataset(Dataset):
    def __init__(self, filename):
        self.matches = []
        self.dataset = None
        self.filename = filename

    def _lazy(self):
        if self.dataset is None:
            self.dataset = zipfile.ZipFile(self.filename, mode='r')
        return self.dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dataset:
            self.dataset.close()
            self.dataset = None

    def __del__(self):
        if self.dataset:
            self.dataset.close()

    def __len__(self):
        return len(self.matches)

    def load_match(self, name):
        with self._lazy().open(name, 'r') as match:
            return json.load(match)


class Dota2PickBan(ZipDataset):
    """Iterate over dumps of Match details

    Notes
    -----
    The drafting orders have changed over the years but hopefully bots can
    extract some useful information from the drafting itself

    Examples
    --------
    >>> dataset = Dota2PickBan('drafting_captains_mode_p2.zip')
    >>> print(len(dataset))
    1366
    """

    def __init__(self, filename, patch=None):
        super(Dota2PickBan, self).__init__(filename)
        self.patch_count = defaultdict(int)
        self.patch_decision = defaultdict(lambda: defaultdict(int))

        self.dataset = zipfile.ZipFile(filename, mode='r')
        self.matches = self.filter_by(self.dataset, patch)
        self.hero_size = const.HERO_COUNT

        # for multi worker setup the dataset need to be opened on the worker process
        self.dataset.close()
        self.dataset = None

    def show_patches(self):
        return self.patch_count

    def filter_by(self, dataset, patch):
        if patch is None:
            return list(dataset.namelist())

        count = DECISION_COUNT.get(patch, 24)
        matches = []
        for match in dataset.namelist():
            match_data = self.load_match(match)
            match_patch = match_data['patch']

            picks_bans = match_data.get('picks_bans', [])
            if not picks_bans:
                continue

            decision_count = len(picks_bans)
            self.patch_decision[match_patch][decision_count] += 1

            if match_patch == patch:
                if decision_count != count:
                    self.patch_count['error'] += 1

                else:
                    matches.append(match)

            self.patch_count[match_patch] += 1

        return matches

    def __getitem__(self, item):
        """Generates one observations """
        name = self.matches[item]
        match = self.load_match(name)

        picks = match['picks_bans']
        is_radiant_win = match['radiant_win']
        return encode_draft(picks, is_radiant_win, True)


def encode_draft(picks, is_radiant_win, seq, heroes=None):
    picks.sort(key=lambda p: p['order'])
    draft = DraftTracker()

    input = []
    pick_target = []
    ban_target = []

    for pick in picks:
        team = pick['team'] + 2
        assert team in (TEAM_RADIANT, TEAM_DIRE)

        hid = pick['hero_id']

        # sometimes the dota2 dump shows 6 picks
        # not sure if it happens when one hero is picked on both teams or not
        # but we need to ignore fake picks in Rank
        if heroes is not None and hid not in heroes:
            continue

        hero = const.HERO_LOOKUP.from_id(hid)

        hoffset = hero['offset']
        hname = hero['name']

        ypick = -1
        yban = -1

        if pick['is_pick']:
            ypick = hoffset
        else:
            yban = hoffset

        # only generate state for the decision steps
        if (team == TEAM_RADIANT and is_radiant_win) or (team == TEAM_DIRE and not is_radiant_win):
            if seq:
                input.append(draft.as_tensor(team))

            pick_target.append(ypick)
            ban_target.append(yban)

        if pick['is_pick']:
            draft.pick(team, hname)
        else:
            draft.ban(team, hname)

    if seq:
        return torch.stack(input), \
               torch.tensor(pick_target, dtype=torch.int64), \
               torch.tensor(ban_target, dtype=torch.int64)

    return draft.as_tensor(TEAM_RADIANT), torch.tensor(int(is_radiant_win), dtype=torch.long)


class Dota2Matchup(ZipDataset):
    DEFAULT_FILTER = (22,)

    def __init__(self, filename, modes=DEFAULT_FILTER):
        super(Dota2Matchup, self).__init__(filename)
        self.modes = set(modes)
        self.mode_counts = defaultdict(int)

        self.dataset = zipfile.ZipFile(filename, mode='r')
        self.matches = self.filter_by(self.dataset, self.modes)
        self.dataset.close()
        self.dataset = None

    def filter_by(self, dataset, modes):
        if modes is None:
            return

        matches = []

        for match in dataset.namelist():
            match_data = self.load_match(match)
            game_mode = match_data['game_mode']
            picks_bans = match_data.get('picks_bans')
            radiant_win = match_data.get('radiant_win')

            self.mode_counts[DOTA_GameMode(game_mode).name] += 1

            if game_mode in self.modes and picks_bans is not None and radiant_win is not None:
                matches.append(match)

        return matches

    def __getitem__(self, item):
        name = self.matches[item]
        match = self.load_match(name)

        picks = match['picks_bans']
        is_radiant_win = match['radiant_win']
        players = match['players']

        heroes = set()
        for p in players:
            heroes.add(p['hero_id'])

        return encode_draft(picks, is_radiant_win, False, heroes)


if __name__ == '__main__':
    dataset = Dota2Matchup('/home/setepenre/work/LuaFun/drafting_all.zip')

    x, y = dataset[0]
    print(len(dataset))
    print(x.shape)
    print(x.sum(dim=1), x.sum(dim=1).sum(), y)

    for i in range(len(dataset)):
        print(dataset[i][1])


    # dataset = Dota2PickBan('/home/setepenre/work/LuaFun/opendota_CM_20210421.zip', patch='7.29')
    #
    # # x, y = dataset[0]
    # # print(x.shape, y.shape)
    #
    # print(json.dumps(dataset.show_patches(), indent=2))
    # #
    # # for k, v in dataset.patch_decision.items():
    # #     print(f'{k:>20}: {v}')
    # print(json.dumps(dataset.patch_decision, indent=2))
