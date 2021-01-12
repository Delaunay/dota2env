import asyncio
from struct import unpack
import subprocess
import uuid
import os
import json
from dataclasses import dataclass

from luafun.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


async def worldstate_listener(port, message_handler, retries=10):
    reader = None

    for i in range(retries):
        try:
            await asyncio.sleep(0.5)
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
        except ConnectionRefusedError:
            pass
        else:
            break

    if reader is None:
        return

    while True:
        data = await reader.read(4)
        if len(data) != 4:
            print('Could not read message length')
            return

        n_bytes = unpack("@I", data)[0]
        data = await reader.read(n_bytes)

        world_state = CMsgBotWorldState()
        world_state.ParseFromString(data)

        message_handler(world_state)

TEAM_RADIANT = 2
TEAM_DIRE = 3
SEND_MSG = 'bots/IPC_recv.lua'
uid = 0


def new_ipc_message():
    return {
        'uid': 0,
        TEAM_RADIANT: {
            0: dict(),
            1: dict(),
            2: dict(),
            3: dict(),
            4: dict(),
        },
        TEAM_DIRE: {
            5: dict(),
            6: dict(),
            7: dict(),
            8: dict(),
            9: dict(),
        }
    }


def write_message(path, data):
    global uid

    f2 = os.path.join(path, SEND_MSG)
    f1 = f2 + '_tmp'

    if os.path.exists(f2):
        os.remove(f2)

    uid += 1
    data['uid'] = uid
    json_string = json.dumps(data, separators=(',', ':'))

    with open(f1, 'w') as file:
        # Keep track of the message we are sending
        file.write(f'return \'{json_string}\'')

    # Renaming is almost always atomic
    os.rename(f1, f2)


def dota2_aguments(ports, game_id=None, game_mode=11, host_timescale=2, ticks_per_observation=30):
    # console_filename = f'console-{game_id}.log'
    console_filename = f'console.log'

    # type > jointeam spec
    # in the console to observe the game
    return [
        # '-dedicated',
        '-botworldstatesocket_threaded',
        '-botworldstatetosocket_frames', '{}'.format(ticks_per_observation),
        '-botworldstatetosocket_radiant', '{}'.format(ports[TEAM_RADIANT]),
        '-botworldstatetosocket_dire', '{}'.format(ports[TEAM_DIRE]),
        # console log file
        '-con_logfile', 'scripts/vscripts/bots/{}'.format(console_filename),
        '-con_timestamp',
        # Enable console in game
        '-console',
        # enable dev assertion
        '-dev',
        # disable VAC
        '-insecure',
        # do not bind the ip
        '-noip',
        '-nowatchdog',  # WatchDog will quit the game if e.g. the lua api takes a few seconds.
        '+clientport', '27006',  # Relates to steam client.
        '+dota_1v1_skip_strategy', '1',
        '+dota_surrender_on_disconnect', '0',
        # Make the game start with bots
        '-fill_with_bots',
        # Local Game Speed
        '+host_timescale', '{}'.format(host_timescale),
        '+hostname dotaservice',
        '+sv_cheats', '1',
        '+sv_hibernate_when_empty', '0',
        # Dota TV settings
        '+tv_delay', '0',
        '+tv_enable', '1',
        '+tv_title', '{}'.format(game_id),
        '+tv_autorecord', '1',
        '+tv_transmitall', '1',
        '+map',
        'start',
        'gamemode', '{}'.format(game_mode),
        # I do not know what this is supposed to do
        # '+sv_lan', '0'
    ]

# Game Mode
# enum DOTA_GameMode {
# 	DOTA_GAMEMODE_NONE = 0;
# 	DOTA_GAMEMODE_AP = 1;       All Pick
# 	DOTA_GAMEMODE_CM = 2;       Captains Mode
# 	DOTA_GAMEMODE_RD = 3;       Random Draft
# 	DOTA_GAMEMODE_SD = 4;       Single Draft
# 	DOTA_GAMEMODE_AR = 5;       All Random
# 	DOTA_GAMEMODE_INTRO = 6;
# 	DOTA_GAMEMODE_HW = 7;
# 	DOTA_GAMEMODE_REVERSE_CM = 8; Reverse Captains Mode
# 	DOTA_GAMEMODE_XMAS = 9;
# 	DOTA_GAMEMODE_TUTORIAL = 10;
# 	DOTA_GAMEMODE_MO = 11;      Melee Only ?
# 	DOTA_GAMEMODE_LP = 12;      Least Played
# 	DOTA_GAMEMODE_POOL1 = 13;   Limited Heroes ?
# 	DOTA_GAMEMODE_FH = 14;
# 	DOTA_GAMEMODE_CUSTOM = 15;
# 	DOTA_GAMEMODE_CD = 16;  Captains draft
# 	DOTA_GAMEMODE_BD = 17;
# 	DOTA_GAMEMODE_ABILITY_DRAFT = 18;
# 	DOTA_GAMEMODE_EVENT = 19;
# 	DOTA_GAMEMODE_ARDM = 20;    All Random Death Match
# 	DOTA_GAMEMODE_1V1MID = 21;
# 	DOTA_GAMEMODE_ALL_DRAFT = 22;  Ranked All Pick
# 	DOTA_GAMEMODE_TURBO = 23;
# 	DOTA_GAMEMODE_MUTATION = 24;
# }


def launch_dota(game_id=None, game_mode=1, host_timescale=2, ticks_per_observation=30):
    """Launch dota and listen to """
    faction_ports = {
        TEAM_RADIANT: 12120,
        TEAM_DIRE: 12121
    }

    if not game_id:
        game_id = str(uuid.uuid1())

    base_path = 'F:/SteamLibrary/steamapps/common/dota 2 beta/'
    script_path = f'{base_path}/game/bin/win64/dota2.exe'
    args = [script_path] + dota2_aguments(faction_ports, game_id, game_mode, host_timescale, ticks_per_observation)
    loop = asyncio.get_event_loop()

    radiant = open('out_radiant.txt', 'w')
    def radiant_message_handler(msg):
        radiant.write(str(msg))

    dire = open('out_dire.txt', 'w')
    def dire_message_handler(msg):
        dire.write(str(msg))

    tasks = asyncio.gather(
        worldstate_listener(faction_ports[TEAM_RADIANT], radiant_message_handler),
        worldstate_listener(faction_ports[TEAM_DIRE], dire_message_handler),
    )

    def cleanup():
        radiant.close()
        dire.close()

    try:
        # This exits immediately but dota is launched on a different process
        subprocess.Popen(args)

        write_message(
            f'{base_path}/game/dota/scripts/vscripts',
            new_ipc_message()
        )

        # update our observation state
        loop.run_until_complete(tasks)
        cleanup()
    except Exception as e:
        cleanup()
        raise e


if __name__ == '__main__':
    launch_dota()

