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


#  https://api.opendota.com/api/players/81280209
class OpenDotaAPI(WebAPI):
    URL = 'https://api.opendota.com/api/{method}'
    KEY = option('opendota.api', None)

    def __init__(self):
        super(OpenDotaAPI, self).__init__()

        now = datetime.datetime.now()
        day_count = calendar.monthrange(now.year, now.month)[1]

        # 50000 per month
        self.max_api_call_day = 50000 // day_count
        # 60 calls per minute
        self.wait_time = 1

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
    def __init__(self):
        self.running = True
        self.known_match = defaultdict(int)
        self.api = OpenDotaAPI()
        self.error_retry = 3
        self.error_sleep = 30
        self.count = 5000
        self.offset = None

    def get_captains_mode_matches(self, count, offset):
        for i in range(self.error_retry):
            try:
                return self.api.get_captains_mode_matches(count=count, offset=offset)
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
        rows = self.get_captains_mode_matches(self.count, self.offset)

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


if __name__ == '__main__':
    # get_captains_mode()
    builder = DatasetBuilder()
    builder.run('opendota.zip')

    # cleanup()