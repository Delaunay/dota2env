"""Used to query SteamAPI to get dota match information to bootstrap bots"""
from collections import defaultdict
from dataclasses import dataclass
import json
import time
from typing import List
import zipfile
import os

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
class MatchDetail_Player_AbilityUpgrades:
    ability: int
    time: int
    level: int


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
    backpack_0: int
    backpack_1: int
    backpack_2: int
    item_neutral: int
    kills: int
    deaths: int
    assists: int
    leaver_status: int
    last_hits: int
    denies: int
    gold_per_min: float
    xp_per_min: float
    level: int
    hero_damage: int
    tower_damage: int
    hero_healing: int
    gold: int
    gold_spent: int
    scaled_hero_damage: int
    scaled_tower_damage: int
    scaled_hero_healing: int
    ability_upgrades: List[MatchDetail_Player_AbilityUpgrades]
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
    pre_game_duration: int
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
    leagueid: int
    positive_votes: int
    negative_votes: int
    game_mode: int
    picks_bans: List[MatchDetail_Picks]
    flags: str
    engine: int
    radiant_score: int
    dire_score: int


@dataclass
class LeagueListing:
    name: str
    leagueid: int
    description: str
    tournament_url: str


class LimitExceeded(RuntimeError):
    pass


class ServerError(RuntimeError):
    pass


# TODO: add API query limiter
# TODO: add a dynamic dataset builder that save the result of the queries and keep building it up
class SteamAPI:
    """

    References
    ----------

    * https://wiki.teamfortress.com/wiki/WebAPI/GetMatchHistory
    * https://dev.dota2.com/forum/dota-2/spectating/replays/webapi/60177-things-you-should-know-before-starting?t=58317
    """
    URL = 'https://api.steampowered.com/IDOTA2Match_{game_id}/{method}/v1'
    URL_STATS = 'https://api.steampowered.com/IDOTA2MatchStats_{game_id}/{method}/v1'

    def __init__(self):
        # 100,000 API calls per day.
        # 1 request per second
        # 60 request per minute
        self.max_api_call_day = 100000
        self.start = None
        # make sure we respect the T&C of valve and do not get banned
        self.wait_time = 1
        self.limiter = True
        self.request_count = 0

    def limit_stats(self):
        return self.request_count / self.max_api_call_day

    def handle_errors(self, response):
        if response.status_code == 503:
            time.sleep(30)
            raise ServerError

        if response.status_code != 200:
            time.sleep(30)
            print(f'Received {response.reason}')
            raise ServerError

    def limit(self):
        if self.limiter:
            # sleep a second to never go over the 1 request per second limit
            time.sleep(self.wait_time)
            self.request_count += 1

            if self.start is None:
                self.start = time.time()

            # reset the request count to 0 after a day
            if time.time() - self.start > 24 * 60 * 60:
                self.request_count = 0

            if self.request_count > self.max_api_call_day:
                raise LimitExceeded('Cannot make more requests today')

    def get_match_history(self, mode: int = DOTA_GameMode.DOTA_GAMEMODE_CM.value, skill=3, min_players=10, count=500, league_id=None, date_min=None, start_at_match_id=None) -> MatchHistory:
        # Results are limited 500 per query
        params = {
            'mode': mode,
            'skill': skill,
            'min_players': min_players,
            'matches_requested': count,
            'key': STEAM_API_KEY,
            'league_id': league_id,
            'start_at_match_id': start_at_match_id,
            'account_id': None,
            # Docs say `Start searching for matches equal to or older than this match ID.`
            # which does not make sense, I want newer match not older one
            # date_max
            'date_min': date_min,
            # Optionals
            # 'format': 'json',
            # 'language': 'en_US'
        }

        url = SteamAPI.URL.format(game_id=DOTA_ID, method='GetMatchHistory')
        response = requests.get(url, params=params)
        print(response.url)
        self.handle_errors(response)
        self.limit()
        return response.json().get('result')

    def get_match_detail(self, match_id) -> MatchDetail:
        params = {
            'match_id': match_id,
            'key': STEAM_API_KEY,
        }

        url = SteamAPI.URL.format(game_id=DOTA_ID, method='GetMatchDetails')
        response = requests.get(url, params=params)
        self.handle_errors(response)
        self.limit()
        return response.json().get('result')

    def get_league_listing(self) -> LeagueListing:
        params = {
            'key': STEAM_API_KEY,
        }

        url = SteamAPI.URL.format(game_id=DOTA_ID, method='GetLeagueListing')
        response = requests.get(url, params=params)
        self.handle_errors(response)
        self.limit()
        return response.json().get('result')

    def get_realtime_match_stats(self, match_id) -> MatchDetail:
        params = {
            'match_id': match_id,
            'key': STEAM_API_KEY,
        }

        url = SteamAPI.URL_STATS.format(game_id=DOTA_ID, method='GetRealtimeStats')
        response = requests.get(url, params=params)
        self.handle_errors(response)
        self.limit()
        return response.json().get('result')


class DatasetBuilder:
    def __init__(self, method='query', skill=3, mode=DOTA_GameMode.DOTA_GAMEMODE_CM.value, match_id=None):
        self.api = SteamAPI()
        self.latest_date = None
        self.pending_matches = []
        self.processed_matches = []
        self.known_match = defaultdict(int)
        self.matches = 0
        self.running = True
        self.latest_match = None
        self.skill = skill
        self.mode = mode
        self.method = method
        self.start_match_id = match_id
        self.counts = defaultdict(int)
        # skill=3, min_players=10, count=500
        #   BATCH_SLEEP=1
        #   MATCH_SLEEP=3

        # Tweak the sleeps to avoid duplicates
        # Sleeps are not there for the lulz or even to be nice to valve
        # matches are kind of slow to appear so to avoid getting tons of duplicates
        # we need to sleep a bit so when all the matches are processed we get a fresh batch
        self.batch_sleep = 5
        self.match_sleep = 3
        self.error_retry = 3
        self.error_sleep = 30

    def update_latest_date(self, date):
        if self.latest_date is None:
            self.latest_date = date
        else:
            self.latest_date = max(date, self.latest_date) + 1

    def update_latest_match(self, match):
        if self.latest_match is None:
            self.latest_match = match
        else:
            self.latest_match = max(match, self.latest_match) + 1

    def status(self):
        api_call_limit = f'API: {self.api.limit_stats() * 100:6.2f}%'
        match = f'Matches: {self.matches} (unique: {len(self.known_match) * 100 / self.matches:6.2f}%)'
        return f'{api_call_limit} {match}'

    def write_match(self, dataset, match, data):
        self.counts[DOTA_GameMode(data.get('game_mode', 0)).name] += 1

        with dataset.open(f'{match}.json', 'w') as match_file:
            jsonstr = json.dumps(data)
            match_file.write(jsonstr.encode('utf-8'))

    def brute_search(self, dataset):
        """Use a match id and fetch all the matches around it"""
        match_id = self.latest_match
        k = 0
        err = 0

        while True:
            match_id += 1
            self.latest_match = match_id
            k += 1

            details = self.get_match_detail(match_id)

            if details.get('error') is not None:
                continue

            if details is None:
                time.sleep(self.error_sleep)
                err += 1
                continue

            self.known_match[match_id] += 1
            self.matches += 1

            count = self.known_match[match_id]

            if count == 1:
                self.write_match(dataset, match_id, details)
            else:
                print('Duplicate')

            if err > 10:
                self.running = False
                break

            k = k % 100
            if k == 0:
                print(f'Latest Match: {self.latest_match} | {self.status()} | err {err}')
                print(json.dumps(self.counts, indent=2))

    def get_match_history(self, start_at_match_id):
        for i in range(self.error_retry):
            try:
                return self.api.get_match_history(
                    skill=self.skill,
                    min_players=10,
                    count=500,
                    date_min=self.latest_date,
                    start_at_match_id=start_at_match_id,
                    mode=int(self.mode),
                )
            except KeyboardInterrupt:
                raise
            except:
                print('Error retrying')
                time.sleep(self.error_sleep)

    def get_match_detail(self, match_id):
        for i in range(self.error_retry):
            try:
                return self.api.get_match_detail(match_id)
            except KeyboardInterrupt:
                raise
            except:
                print('Error retrying')
                time.sleep(self.error_sleep)

    def query_search(self, dataset):
        """Fetch new matches"""
        # date_min does not work, we get a lot of duplicate really fast
        # start_at_match_id does not work at all it does seem that the doc was correct and get older match from that id
        remaining = None
        total = None
        results = None
        start_at_match_id = None

        while remaining is None or remaining > 0:
            result = self.get_match_history(start_at_match_id)

            matches = result['matches']
            results = result["num_results"]
            total = result["total_results"]
            remaining = result["results_remaining"]

            if len(matches) > 0:
                start_at_match_id = matches[-1]['match_id'] + 1
                self.pending_matches.extend(matches)

        print(f'+-> Found {len(self.pending_matches)} matches')
        print(f'        - Results: {results}')
        print(f'        - Total: {total}')
        print(f'        - Remaining: {remaining}')

        while self.pending_matches:
            match = self.pending_matches.pop()
            match_id = match['match_id']

            self.update_latest_date(match['start_time'])
            self.update_latest_match(match_id)

            # We shouldnt see this error we to be sure we track it to see
            self.known_match[match_id] += 1
            count = self.known_match[match_id]
            self.matches += 1
            if count != 1:
                print(f'+-+> Match duplicate {count} {self.status()}')
                continue
            # ----
            details = self.get_match_detail(match_id)
            if details is None:
                time.sleep(self.error_sleep)
                continue

            self.write_match(dataset, match_id, details)
            time.sleep(self.match_sleep)

    def run(self, name):
        if os.path.exists(name):
            print('Cannot override existing file')
            return

        server_error = 0
        self.latest_match = self.start_match_id

        with zipfile.ZipFile(name, mode='w') as dataset:
            self.running = True
            while self.running:
                try:
                    if self.method == 'method':
                        self.query_search(dataset)
                    elif self.method == 'brute':
                        self.brute_search(dataset)
                    else:
                        return

                    print(f'+> Batch done')
                    time.sleep(self.batch_sleep)
                    print(f'+> Fetching next 500 matches {self.status()}')
                except KeyboardInterrupt:
                    self.running = False

                except ServerError:
                    server_error += 1
                    print('+> Server error retrying')
                    if server_error > 3:
                        self.running = False

                except LimitExceeded:
                    self.running = False

                except requests.ConnectionError:
                    pass

        print(f'> Unique Match processed: {len(self.known_match)} Total match: {self.matches}')


def remove_duplicate(name='drafting_captains_mode'):
    with zipfile.ZipFile(f'{name}.zip', mode='r') as original:
        with zipfile.ZipFile(f'{name}_unique.zip', mode='w') as new:

            for file in original.namelist():
                match_id, count = file.split('_')

                if count != '1.json':
                    continue

                with original.open(file) as original_match:
                    with new.open(f'{match_id}.json', 'w') as new_match:
                        new_match.write(original_match.read())


def merge(output, *args):
    matches = set()

    with zipfile.ZipFile(output, mode='w') as new:
        for file in args:
            with zipfile.ZipFile(file, mode='r') as original:
                for match in original.namelist():
                    if match in matches:
                        print('skipping duplicate')
                        continue

                    with original.open(match, 'r') as original_match:
                        with new.open(match, 'w') as new_match:
                            new_match.write(original_match.read())

                    matches.add(match)

    print(f'Done Match Count: {len(matches)}')


class FakeSteamAPI:
    def get_match_history(self, *args, **kwargs):
        with open('matches.json', 'r') as f:
            return json.load(f)

    def get_match_details(self, *args, **kwargs):
        with open('match_details.json', 'r') as f:
            return json.load(f)


def execute_get_match_history():
    api = SteamAPI()

    with open('matches.json', 'w') as f:
        data = api.get_match_history()
        f.write(json.dumps(data))


def execute_get_match_detail(id):
    api = SteamAPI()

    with open('match_details.json', 'w') as f:
        data = api.get_match_detail(id)
        f.write(json.dumps(data))


def test_test():
    api = FakeSteamAPI()
    matches = api.get_match_history()['matches']
    match = matches[0]
    match_id = match['match_id']

    details = api.get_match_details(match_id)

    print(match_id)
    # execute_get_match_detail(match_id)

    print(json.dumps(details, indent=2))
    print()


def main():
    """You probably do not want to use this service, it is better to download dumps made by other people
    since it will be much faster than this.

    This code has to work around valve query limitation and it is quite slow.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()

    commands = parser.add_subparsers(dest='command')
    fetch_args = commands.add_parser('fetch')
    fetch_args.add_argument('--method', default='query', type=str, choices=['query', 'brute'],
                            help='Select how we find new matches')
    fetch_args.add_argument('--match', default=5946003057, type=int,
                            help='Match id to use to start the brute force search')
    fetch_args.add_argument('--skill', default=3, type=int, choices=[0, 1, 2, 3],
                            help='skill level (0:Any, 1: Normal, 2: High, 3: Very High)')
    # mode filter does not seem to work
    fetch_args.add_argument('--mode', default=None, type=int, choices=[1, 2, 5, 22],
                            help='Dota game mode (1: All Pick, 2: Captain\'s Mode, 5: All Random. 22: Ranked All picks)')
    fetch_args.add_argument('output', default='drafting_captains_mode.zip', type=str,
                            help='Output zip file')

    merge_args = commands.add_parser('merge')
    merge_args.add_argument('--output', default='merged.zip', type=str,
                            help='Output file')
    merge_args.add_argument('files', type=str, nargs='+',
                            help='Zip files to merge together')
    args = parser.parse_args()

    if args.command == 'fetch':
        builder = DatasetBuilder(skill=args.skill, mode=args.mode, method=args.method, match_id=args.match)
        builder.run(args.output)

    if args.command == 'merge':
        merge(args.output, *args.files)


if __name__ == '__main__':
    # 7991 matches so far
    main()

    # for listing in SteamAPI().get_league_listing():
    #     print(json.dumps(listing))
