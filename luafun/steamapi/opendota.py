from collections import defaultdict
import json
import calendar
import datetime
import os
import zipfile
import time

import requests

from luafun.utils.options import option
from luafun.steamapi.api import WebAPI, ServerError, LimitExceeded
from luafun.game.modes import DOTA_GameMode


def generate_table_view():
    with open('/home/setepenre/work/LuaFun/luafun/steamapi/opendota.json', 'r') as schema:
        data = json.load(schema)

    tables = defaultdict(lambda : defaultdict(dict))

    for spec in data:
        table = tables[spec.pop('table_name')]
        table[spec.pop('column_name')] = spec.pop('data_type')

        if spec:
            print(spec)

    print(json.dumps(tables, indent=2))


def leagues():
    with open('/home/setepenre/work/LuaFun/luafun/steamapi/leagues.json', 'r') as leagues:
        data = json.load(leagues)

    data.sort(key=lambda x: x['leagueid'])
    for league in data:
        if league['tier'] == 'professional':
            print(league)


def get_latest_patch():
    with open('/home/setepenre/work/LuaFun/luafun/steamapi/patch.json', 'r') as patches:
        data = json.load(patches)

    max_value = float('-inf')
    latest_path = None
    for patch in data:
        value = float(patch['patch'])
        if max_value < value:
            max_value = value
            latest_path = patch['patch']

    return latest_path


def get_path_dates():
    query = """
    SELECT 
        DISTINCT match_patch.patch,
        MIN(matches.start_time),
        MAX(matches.start_time)
    FROM matches
    INNER JOIN match_patch USING(match_id)
    GROUP BY
        match_patch.patch
    ;
    """

    return query


#  https://api.opendota.com/api/players/81280209
class OpenDotaAPI(WebAPI):
    URL = 'https://api.opendota.com/api/{method}'
    KEY = option('opendota.api', None)

    def __init__(self):
        super(OpenDotaAPI, self).__init__('opendota')

        now = datetime.datetime.now()
        day_count = calendar.monthrange(now.year, now.month)[1]

        # 50000 per month
        self.max_api_call_day = 50000 // day_count
        # 60 calls per minute
        self.wait_time = 2

    def explore(self, sql):
        params = {
            'sql': sql,
            'key': self.KEY,
        }

        url = self.URL.format(method='explorer')
        response = requests.get(url, params=params)

        self.handle_errors(response)
        self.limit()

        data = response.json()
        rows = data.get('rows')

        if rows is not None:
            return rows

        return data

    def get_player(self, player_id):
        """

        >>> OpenDotaAPI().get_player(81280209)
        {
          "tracked_until": "1621953083",
          "leaderboard_rank": null,
          "solo_competitive_rank": null,
          "profile": {
            "account_id": 81280209,
            "personaname": "Sétepenrê",
            "name": null,
            "plus": false,
            "cheese": 0,
            "steamid": "76561198041545937",
            "avatar": "https://steamcdn-a.akamaihd.net/steamcommunity/public/images/avatars/e6/e68a0b9be84eafb21eadd5fa73a32c995fc7991b.jpg",
            "avatarmedium": "https://steamcdn-a.akamaihd.net/steamcommunity/public/images/avatars/e6/e68a0b9be84eafb21eadd5fa73a32c995fc7991b_medium.jpg",
            "avatarfull": "https://steamcdn-a.akamaihd.net/steamcommunity/public/images/avatars/e6/e68a0b9be84eafb21eadd5fa73a32c995fc7991b_full.jpg",
            "profileurl": "https://steamcommunity.com/id/setepenre/",
            "last_login": "2021-04-10T23:03:15.390Z",
            "loccountrycode": null,
            "is_contributor": false
          },
          "rank_tier": 24,
          "competitive_rank": null,
          "mmr_estimate": {
            "estimate": 2223
          }
        }
        """
        params = {
            'key': self.KEY,
        }

        url = self.URL.format(method='players') + f'/{player_id}'
        response = requests.get(url, params=params)

        self.handle_errors(response)
        self.limit()

        return response.json()

    def get_all_pick_draft(self, count, offset, version="7.28"):
        """Does not work, opendota does not save pick order for public matches only
        pro matches"""
        mode = int(DOTA_GameMode.DOTA_GAMEMODE_ALL_DRAFT)
        start7_28 = 1608249726
        start7_29 = 1617981499

        # Game mode in open dota is not correct
        # matches.game_mode = {mode}        AND
        query = f"""
        SELECT 
            public_matches.match_id,
            public_matches.radiant_win,
            match_patch.patch,
            public_matches.avg_rank_tier,
            public_matches.num_rank_tier,
            picks_bans.is_pick, 
            picks_bans.hero_id, 
            picks_bans.team, 
            picks_bans.ord  
        FROM 
            public_matches,
            picks_bans
        INNER JOIN match_patch USING(match_id)
        INNER JOIN picks_bans USING(match_id)
        WHERE
            public_matches.start_time >= {start7_28} AND
            public_matches.start_time <= {start7_29} AND
            match_patch.patch = '{version}'
        LIMIT {count}
        """

        if offset is not None:
            query = f'{query} OFFSET {offset}'

        query = f'{query};'

        return self.explore(query)

    def get_all_pick_draft_match_id(self, count, offset):
        """Does not work, opendota does not save pick order for public matches only
        pro matches"""
        mode = int(DOTA_GameMode.DOTA_GAMEMODE_ALL_DRAFT)
        start7_28 = 1608249726
        start7_29 = 1617981499

        # Game mode in open dota is not correct
        # matches.game_mode = {mode}        AND
        query = f"""
        SELECT 
            *
        FROM 
            public_matches
        WHERE
            public_matches.start_time >= {start7_28} AND
            public_matches.start_time <= {start7_29} AND
            public_matches.game_mode = 22
        LIMIT {count}
        """

        if offset is not None:
            query = f'{query} OFFSET {offset}'

        query = f'{query};'

        return self.explore(query)

    def get_captains_mode_matches(self, count, offset, version="7.28", mode=2):
        """To make sure your query is fast you should use a time constrain
        so the DB can focus on the partition that matters
        """
        query = f"""
        SELECT 
            matches.match_id,
            matches.radiant_win,
            matches.picks_bans,
            match_patch.patch
        FROM 
            matches
        INNER JOIN match_patch USING(match_id)
        WHERE
            matches.human_players = 10          AND
            matches.game_mode = {mode}          AND
            matches.picks_bans IS NOT NULL
        LIMIT {count}
        """

        if offset is not None:
            query = f'{query} OFFSET {offset}'

        query = f'{query};'

        return self.explore(query)


class DatasetBuilder:
    def __init__(self, api):
        self.running = True
        self.known_match = defaultdict(int)
        self.api = api
        self.error_retry = 3
        self.error_sleep = 30
        self.count = 5000
        self.offset = None

    def query(self, count, offset):
        for i in range(self.error_retry):
            try:
                # return self.api.get_captains_mode_matches(count=count, offset=offset)
                return self.api.get_all_pick_draft(count=count, offset=offset)
            except KeyboardInterrupt:
                raise
            except:
                print('+-> Error retrying')
                time.sleep(self.error_sleep)

    def write_match(self, dataset, data):
        match = data.get('match_id')

        if match is None:
            print(f'+-+> match without id {data}')
            return

        if match in self.known_match:
            print(f'+-+> Skipping duplicate `{match}`')
            print(data)
            return

        self.known_match[match] += 1
        with dataset.open(f'{match}.json', 'w') as match_file:
            jsonstr = json.dumps(data)
            match_file.write(jsonstr.encode('utf-8'))

    def fetch_entries(self, dataset):
        rows = self.query(self.count, self.offset)

        if not isinstance(rows, list):
            print(f'+-+> {rows}')
            return

        if self.offset is None:
            self.offset = 0

        self.offset += self.count

        for match in rows:
            self.write_match(dataset, match)

        print(f'+-> Batch processed (n: {len(self.known_match)}) (API: {self.api.limit_stats() * 100:6.2f})')

        if len(rows) < self.count:
            print(f'+-> Found only {len(rows)} matches')
            self.running = False

    def run(self, name):
        if os.path.exists(name):
            print('Cannot override existing file')
            return

        server_error = 0
        with zipfile.ZipFile(name, mode='w') as dataset:
            while self.running:
                try:
                    self.fetch_entries(dataset)

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

        print(f'> Unique Match processed: {len(self.known_match)}')


def cleanup():
    count = 0
    deleted = 0
    with zipfile.ZipFile('/home/setepenre/work/LuaFun/opendota_CM_20210421.zip', 'r') as original:
        with zipfile.ZipFile('/home/setepenre/work/LuaFun/opendota_CM_20210421_clean.zip', 'w') as clean:
            for file in original.namelist():
                with original.open(file, 'r') as original_match:
                    match = json.load(original_match)

                    if match.get('picks_bans') is None:
                        deleted += 1
                        continue

                    with clean.open(file, 'w') as clean_match:
                        clean_match.write(json.dumps(match).encode())
                        count += 1

    print(count, deleted)



def extract_ranked_all_pick_match_id():
    opendata = OpenDotaAPI()
    offset = 0
    count = 50000

    full_dump = []

    with open('ranked_allpick_7.28.json', 'w') as fp:
        while True:
            try:
                array = opendata.get_all_pick_draft_match_id(count, offset)
                offset += len(array)

                for row in array:
                    fp.write((json.dumps(row) + '\n'))

                print(f'+-> Got {len(array)} match ids')

                if len(array) < count:
                    break

            except ServerError:
                pass


if __name__ == '__main__':
    import json

    # with OpenDotaAPI() as api:
        # get_captains_mode()
        # builder = DatasetBuilder()
        # builder.run('opendota_ranked_all_pick.zip')

        # cleanup()


