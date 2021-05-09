from collections import defaultdict
import json
import zipfile

import torch
from torch.utils.data.dataset import Dataset

from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE
from luafun.draft import DraftTracker
import luafun.game.constants as const
from luafun.game.game import DOTA_GameMode
from luafun.model.drafter import JudgeEstimates, JudgeEstimatesNorm


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

            if patch is not None and match_patch != patch:
                continue

            elif decision_count != count:
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
            return list(dataset.namelist())

        matches = []

        for match in dataset.namelist():
            match_data = self.load_match(match)
            game_mode = match_data['game_mode']
            picks_bans = match_data.get('picks_bans')
            radiant_win = match_data.get('radiant_win')
            avg_rank_tier = match_data.get('avg_rank_tier')

            self.mode_counts[DOTA_GameMode(game_mode).name] += 1

            if game_mode in self.modes and picks_bans is not None and radiant_win is not None and avg_rank_tier is not None:
                matches.append(match)

        return matches

    def __getitem__(self, item):
        name = self.matches[item]
        match = self.load_match(name)

        picks = match['picks_bans']
        is_radiant_win = match['radiant_win']

        players = match['players']
        # players.sort(key=lambda p: p['player_slot'])

        meta = torch.zeros(JudgeEstimates.Size)
        meta[JudgeEstimates.Duration] = match['duration']

        heroes = set()
        for p in players:
            heroes.add(p['hero_id'])
            slot = p['player_slot']

            if slot >= 128:
                slot = slot - 128 + 5

            meta[getattr(JudgeEstimates, f'GoldPerMin{slot}')] = \
                (p['gold_per_min'] - JudgeEstimatesNorm.GoldPerMinAvg) / JudgeEstimatesNorm.GoldPerMinStd
            meta[getattr(JudgeEstimates, f'HeroDamage{slot}')] = \
                (p['hero_damage'] - JudgeEstimatesNorm.HeroDamageAvg) / JudgeEstimatesNorm.HeroDamageStd
            meta[getattr(JudgeEstimates, f'TowerDamage{slot}')] = \
                (p['tower_damage'] - JudgeEstimatesNorm.TowerDamageAvg) / JudgeEstimatesNorm.TowerDamageStd

        draft, win = encode_draft(picks, is_radiant_win, False, heroes)

        # Fetch rank data
        rank = None
        if 'avg_rank_tier' in match:
            rank_offset = match['avg_rank_tier'] - 10
            rank = torch.zeros((draft.shape[0], const.Rank.Size,))
            rank[:, rank_offset] = 1

        return draft, win, meta, rank


if __name__ == '__main__':
    from luafun.utils.options import datafile
    from math import sqrt

    dataset = Dota2Matchup(datafile('dataset', 'ranked_allpick_7.28_picks_wip.zip'))

    min_v = float('+inf')
    max_v = float('-inf')
    avg = 0
    count = 0
    std = 0

    for x, y, meta in dataset:
        for i in range(0, 10):
            v = meta[JudgeEstimates.TowerDamage0 + i]

            min_v = min(v, min_v)
            max_v = max(v, max_v)
            avg += v
            count += 1
            std += v * v

    print(f'min {min_v}')
    print(f'max {max_v}')
    print(f'avg {avg / count}')
    print(f'std {sqrt(std / count - (avg / count) ** 2)}')

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
