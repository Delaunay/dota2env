from dataclasses import dataclass
from typing import Optional
import uuid

from luafun.game.modes import DOTA_GameMode


@dataclass
class DotaOptions:
    # All Pick, Restricted Heroes, 1v1 mid
    game_mode: DOTA_GameMode = int(DOTA_GameMode.DOTA_GAMEMODE_AP)
    # Speed of the game
    host_timescale: int = 2
    # Headless (no rendering)
    dedicated: bool = True
    # How often do receive world state updates
    #  4 == every 0.1333 seconds (same as openAI)
    # 30 == every seconds
    ticks_per_observation: int = 4

    # Not sure how this is used but we could try to look for it
    game_id: str = str(uuid.uuid1())

    # You mostly should leave those as is
    port_radiant: int = 12120
    port_dire: int = 12121

    # Steam Client port, you should not modify this
    client_port: int = 27006

    def args(self, paths):
        """Generate the commandline arguments to pass down to the dota2 executable"""
        additional = []

        if self.dedicated:
            additional.append('-dedicated')

        from sys import platform

        if platform == "linux" or platform == "linux2":
            additional.append('-gl')

        return additional + [
            '-botworldstatesocket_threaded',
            '-botworldstatetosocket_frames', '{}'.format(self.ticks_per_observation),
            '-botworldstatetosocket_radiant', '{}'.format(self.port_radiant),
            '-botworldstatetosocket_dire', '{}'.format(self.port_dire),
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
            # looked like the cmd args where console command so tried to execute that one
            # in hope of making us spectator at the start of the game but no dice
            # '-jointeam', 'spec',
            # Relates to steam client.
            '+clientport', '{}'.format(self.client_port),
            '+dota_1v1_skip_strategy', '1',
            # Close dota when the game is over
            '+dota_surrender_on_disconnect', '0',
            # Make the game start with bots
            '-fill_with_bots',
            # Local Game Speed
            '+host_timescale', '{}'.format(self.host_timescale),
            '+hostname dotaservice',
            '+sv_cheats', '1',
            '+sv_hibernate_when_empty', '0',
            # I do not know what this is supposed to do
            # probably limiting the server to lan
            # '+sv_lan', '0'
            # Dota TV settings
            '+tv_delay', '0',
            '+tv_enable', '1',
            '+tv_title', '{}'.format(self.game_id),
            '+tv_autorecord', '1',
            '+tv_transmitall', '1',
            # ---
            '+map',
            # Start the game right away
            'start',
            'gamemode', '{}'.format(self.game_mode),
        ]
