from collections import defaultdict
import json
import zipfile

from torch.utils.data.dataset import Dataset

from luafun.game.game import DOTA_GameMode


class Dota2MatchDetails(Dataset):
    """Iterate over dumps of Match details

    Examples
    --------
    >>> dataset = Dota2MatchDetails('drafting_captains_mode_p2.zip')
    >>> print(len(dataset))
    1366
    """

    def __init__(self, filename, game_mode: DOTA_GameMode):
        self.matches = []
        self.dataset = None
        self.game_mode = game_mode
        self.dataset = zipfile.ZipFile(filename, mode='r')
        self.matches = []

        # filters
        matches = list(self.dataset.namelist())
        for match in matches:
            try:
                data = self.load_match(match)
                if game_mode is None or data['game_mode'] == game_mode:
                    self.matches.append(match)
            except json.decoder.JSONDecodeError:
                print(f'Skipping {match} because of an error')

    def show_game_modes(self):
        counts = defaultdict(int)
        for match in self.dataset.namelist():
            data = self.load_match(match)
            counts[DOTA_GameMode(data['game_mode']).name] += 1
        return counts

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
        return self.load_match(name)


class Dota2Draft(Dataset):
    def __init__(self, filename):
        self.dataset = Dota2MatchDetails(filename, DOTA_GameMode.DOTA_GAMEMODE_ALL_DRAFT)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        return self.dataset.__exit__(*args)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        obs = self.dataset[item]
        radiant_win = obs['radiant_win']
        picks_bans = obs['picks_bans']
        picks_bans.sort(key=lambda x: x['order'])

        print(json.dumps(obs, indent=0))
        pass


if __name__ == '__main__':

    # dataset = Dota2MatchDetails(
    #     '/home/setepenre/work/LuaFun/all_mode_match_bak.zip',
    #     DOTA_GameMode.DOTA_GAMEMODE_ALL_DRAFT
    # )
    #
    # print(json.dumps(dataset.show_game_modes(), indent=2))

    # print(dataset[1])

    dataset = Dota2Draft(
        '/home/setepenre/work/LuaFun/all_mode_match_bak.zip',
    )

    dataset[2]
