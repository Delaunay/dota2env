from collections import defaultdict
import json

import requests


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


def get_captains_mode():
    query = f"""
    SELECT 
        picks_bans.match_id,
        picks_bans.is_pick,
        picks_bans.hero_id,
        picks_bans.team,
        picks_bans.ord,
        matches.draft_timings
    FROM 
        picks_bans
    INNER JOIN match_patch USING(match_id)
    INNER JOIN matches USING(match_id)
    WHERE
        match_patch.patch = '7.29' AND
        matches.game_mode = 2
    LIMIT 1;
    """

    url = 'https://api.opendota.com/api/explorer'

    params = {
        'sql': query,
    }

    response = requests.get(url, params=params)
    data = json.dumps(response.json())

    with open('reply.json', 'w') as f:
        f.write(data)
    return data


if __name__ == '__main__':
    # get_captains_mode()
    print(get_captains_mode())