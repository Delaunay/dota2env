from dataclasses import dataclass
import uuid

from luafun.game.modes import DOTA_GameMode


@dataclass
class DotaOptions:
    """Controls Dota 2 game console arguments

    Parameters
    ----------
    host_timescale: int
        Speed multiplier

    decicated: bool
        Run the decicated server or rendered game

    ticks_per_observation: int
        Number of frames before an observation is sent back
        Server runs at 30 frame per seconds

    game_id: str
        Game id for logging & replay saves

    port_radiant: int
        Port for radiant updates the game is listening to

    port_dire: int
        Port for dire updates the game is listening to

    interactive: bool
        Do not start the lobby right away and enable human players to jump in

    draft: bool
        Enable bot drafting
    """
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

    interactive: bool = False
    draft: bool = False

    def args(self, paths):
        """Generate the commandline arguments to pass down to the dota2 executable"""
        additional = []

        if self.dedicated:
            additional.append('-dedicated')
            additional.append('+dota_1v1_skip_strategy')
            additional.append('1')

        from sys import platform

        if platform == "linux" or platform == "linux2":
            additional.append('-gl')

        interactive = []
        if not self.interactive:
            # Make the game start with bots
            interactive.append('-fill_with_bots')

            # Start the game right away
            interactive.append('+map')
            interactive.append('start')
            interactive.append('gamemode')
            interactive.append(f'{int(self.game_mode)}')

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
            # This is not what is sounds like
            # this is just to watch bot games in the client
            # not to auto join as spectator
            # '-dota_spectator_auto_spectate_bot_games', 1,
            # Relates to steam client.
            '+clientport', '{}'.format(self.client_port),
            # Close dota when the game is over
            '+dota_surrender_on_disconnect', '0',
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

        ] + interactive