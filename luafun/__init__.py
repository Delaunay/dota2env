host_timescale = 2
ticks_per_observation = 30
TEAM_RADIANT = 12120
TEAM_DIRE = 12121
log = 'log.txt'

args = [
            'dota2.exe',
            '-dedicated',
            '-botworldstatesocket_threaded',
            '-botworldstatetosocket_frames', '{}'.format(ticks_per_observation),
            '-botworldstatetosocket_radiant', '{}'.format(TEAM_RADIANT),
            '-botworldstatetosocket_dire', '{}'.format(TEAM_DIRE),
            '-con_logfile', 'scripts/vscripts/bots/{}'.format(log),
            '-con_timestamp',
            '-console',
            '-dev',
            '-insecure',
            '-fill_with_bots',
            '-noip',
            '-nowatchdog',  # WatchDog will quit the game if e.g. the lua api takes a few seconds.
            '+clientport', '27006',  # Relates to steam client.
            '+dota_surrender_on_disconnect', '0',
            '+host_timescale', '{}'.format(host_timescale),
            '+hostname dotaservice',
            '+sv_cheats', '1',
            '+sv_hibernate_when_empty', '0',
            '+tv_delay', '0',
            '+tv_enable', '1',
            '+tv_autorecord', '1',
            '+tv_transmitall', '1',  # TODO(tzaman): what does this do exactly?
            '+map start gamemode 11'
        ]

print(' '.join(args))