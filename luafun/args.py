PORT_TEAM_RADIANT = 12120
PORT_TEAM_DIRE = 12121

def dota2_aguments(paths, game_id, game_mode=1, host_timescale=2, dedicated, ticks_per_observation=30):
    additional = []

    if dedicated:
        additional.append('-dedicated')

    return dedicated + [
        '-botworldstatesocket_threaded',
        '-botworldstatetosocket_frames', '{}'.format(ticks_per_observation),
        '-botworldstatetosocket_radiant', '{}'.format(PORT_TEAM_RADIANT),
        '-botworldstatetosocket_dire', '{}'.format(PORT_TEAM_DIRE),
        # console log file
        '-con_logfile', '{}'.format(paths.console_log),
        '-con_timestamp',
        # Enable console in game
        '-console',
        # enable dev assertion
        '-dev',
        # disable VAC
        '-insecure',
        # do not bind the ip
        '-noip',
        # WatchDog will quit the game if e.g. the lua api takes a few seconds.
        '-nowatchdog', 
        # Relates to steam client.
        '+clientport', '27006',
        '+dota_1v1_skip_strategy', '1',
        # Close dota when the game is over
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
