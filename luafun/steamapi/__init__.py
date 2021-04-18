"""Used to query SteamAPI to get dota match information to bootstrap bots"""
from dataclasses import dataclass
import json
from typing import List

import requests

from luafun.utils.options import option
from luafun.game.game import DOTA_GameMode

# set export LUAFUN_STEAM_API=XXXX
STEAM_API_KEY = option('steam.api', None)
DOTA_ID = 570
DOTA_PRIVATE_BETA = 816
DOTA_BETA_TEST = 205790


@dataclass
class MatchHistory_Match_Player:
    account_id: int
    player_slot: int
    hero_id: int

    def is_dire(self):
        return (self.player_slot & 0b10000000) >> 7

    def position(self):
        return self.player_slot & 0b00000111


@dataclass
class MatchHistory_Match:
    match_id: int
    match_seq_num: int
    start_time: int
    lobby_time: int
    players: List[MatchHistory_Match_Player]


@dataclass
class MatchHistory:
    status: int
    statusDetail: str
    num_results: int
    total_results: int
    results_remaining: int
    matches: List[MatchHistory_Match]


@dataclass
class MatchDetail_Player_Unit:
    unitname: str
    item_0: int
    item_1: int
    item_2: int
    item_3: int
    item_4: int
    item_5: int


@dataclass
class MatchDetail_Player:
    account_id: int
    player_slot: int
    hero_id: int
    item_0: int
    item_1: int
    item_2: int
    item_3: int
    item_4: int
    item_5: int
    kills: int
    deaths: int
    assists: int
    leaver_status: int
    last_hits: int
    denies: int
    gold_per_min: float
    xp_per_min: float
    additional_units: List[MatchDetail_Player_Unit]


@dataclass
class MatchDetail_Picks:
    is_pick: bool
    hero_id: int
    team: int
    order: int


@dataclass
class MatchDetail:
    players: List[MatchDetail_Player]
    season: str
    radiant_win: bool
    duration: int
    start_time: int
    match_id: str
    match_seq_num: int
    tower_status_radiant: int
    tower_status_dire: int
    barracks_status_radiant: int
    barracks_status_dire: int
    cluster: str
    first_blood_time: int
    lobby_type: int
    human_players: int
    leagueid: str
    positive_votes: int
    negative_votes: int
    game_mode: int
    picks_bans: List[MatchDetail_Picks]
    flags: str
    engine: int
    radiant_score: int
    dire_score: int


class SteamAPI:
    """

    References
    ----------

    * https://wiki.teamfortress.com/wiki/WebAPI/GetMatchHistory
    * https://dev.dota2.com/forum/dota-2/spectating/replays/webapi/60177-things-you-should-know-before-starting?t=58317
    """
    URL = 'https://api.steampowered.com/IDOTA2Match_{game_id}/{method}/v1'

    def __init__(self):
        pass

    def get_match_history(self) -> MatchHistory:
        params = {
            'mode': DOTA_GameMode.DOTA_GAMEMODE_CM.value,
            'skill': 3,
            'min_players': 10,
            'matches_requested': 100,
            'key': STEAM_API_KEY,
            # 'format': 'json',
            # 'language': 'en_US'
        }

        url = SteamAPI.URL.format(game_id=DOTA_ID, method='GetMatchHistory')
        response = requests.get(url, params=params)
        return response.json().get('result')

    def get_match_detail(self, match_id) -> MatchDetail:
        params = {
            'match_id': match_id,
            'key': STEAM_API_KEY,
        }

        url = SteamAPI.URL.format(game_id=DOTA_ID, method='GetMatchDetails')
        response = requests.get(url, params=params)
        return response.json().get('result')


class FakeSteamAPI:
    def get_match_history(self):
        with open('matches.json', 'r') as f:
            return json.load(f)


def execute_method():
    api = SteamAPI()
    print('here')

    with open('matches.json', 'w') as f:
        data = api.get_match_history()
        f.write(json.dumps(data))


if __name__ == '__main__':
    data = FakeSteamAPI().get_match_history()
    print(json.dumps(data, indent=2))
