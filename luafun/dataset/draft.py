from collections import defaultdict
import json
import zipfile

import torch
from torch.utils.data.dataset import Dataset

from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE
from luafun.draft import DraftTracker
import luafun.game.constants as const


class Dota2PickBan(Dataset):
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
        self.patch_count = defaultdict(int)
        self.matches = []
        self.dataset = None

        self.filename = filename
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

        matches = []
        for match in dataset.namelist():
            match_patch = self.load_match(match)['patch']

            if match_patch == patch:
                matches.append(match)

            self.patch_count[match_patch] += 1

        return matches

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

    def __getitem__(self, item):
        """Generates one observations """
        name = self.matches[item]
        match = self.load_match(name)

        picks = match['picks_bans']
        picks.sort(key=lambda p: p['order'])

        is_radiant_win = match['radiant_win']
        draft = DraftTracker()

        input = []
        pick_target = []
        ban_target = []

        for pick in picks:
            team = pick['team'] + 2
            assert team in (TEAM_RADIANT, TEAM_DIRE)

            hid = pick['hero_id']
            hero = const.HERO_LOOKUP.from_id(hid)

            hoffset = hero['offset']
            hname = hero['name']

            # ypick = torch.zeros(self.hero_size, dtype=torch.int)
            # yban = torch.zeros(self.hero_size, dtype=torch.int)

            ypick = -1
            yban = -1

            if pick['is_pick']:
                ypick = hoffset
            else:
                yban = hoffset

            # only generate state for the decision steps
            if (team == TEAM_RADIANT and is_radiant_win) or (team == TEAM_DIRE and not is_radiant_win):
                input.append(draft.as_tensor(team))

                pick_target.append(ypick)
                ban_target.append(yban)

            if pick['is_pick']:
                draft.add(team, hname)
            else:
                draft.ban(team, hname)

        return torch.stack(input), \
               torch.tensor(pick_target, dtype=torch.int64), \
               torch.tensor(ban_target, dtype=torch.int64)


if __name__ == '__main__':
    dataset = Dota2PickBan('/home/setepenre/work/LuaFun/opendota_CM_20210421.zip', patch='7.29')

    # x, y = dataset[0]
    # print(x.shape, y.shape)

    print(json.dumps(dataset.show_patches(), indent=2))
