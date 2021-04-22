from collections import defaultdict
import json
import zipfile

from torch.utils.data.dataset import Dataset

from luafun.game.game import DOTA_GameMode


class Dota2PickBan(Dataset):
    """Iterate over dumps of Match details

    Examples
    --------
    >>> dataset = Dota2PickBan('drafting_captains_mode_p2.zip')
    >>> print(len(dataset))
    1366
    """

    def __init__(self, filename):
        self.matches = []
        self.dataset = None

        self.dataset = zipfile.ZipFile(filename, mode='r')
        self.matches = list(self.dataset.namelist())

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dataset.close()
        self.dataset = None

    def __del__(self):
        if self.dataset:
            self.dataset.close()

    def __len__(self):
        return len(self.matches)

    def load_match(self, name):
        with self.dataset.open(name, 'r') as match:
            return json.load(match)

    def __getitem__(self, item):
        name = self.matches[item]
        match = self.load_match(name)

        picks = match['picks_bans']
        picks.sort(key=lambda x: x['order'])

        is_radiant_win = match['radiant_win']

        return match


if __name__ == '__main__':
    dataset = Dota2PickBan('/home/setepenre/work/LuaFun/opendota_CM_20210421_102701.zip')

    print(dataset[0])
