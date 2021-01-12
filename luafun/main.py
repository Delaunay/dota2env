import asyncio
import logging
import os
import subprocess
import uuid

from luafun.config import DotaPaths
from luafun.args import dota2_aguments, PORT_TEAM_RADIANT, PORT_TEAM_DIRE
from luafun.modes import DOTA_GameMode
from luafun.ipc_recv import ipc_recv
from luafun.ipc_send import ipc_send
from luafun.states import worldstate_listener


log = logging.getLogger(__name__)

class State:
    def __init__(self):
        self.running = True


class Dota2Game:
    """Simple interface to listen and send messages to a running dota2 game instance
    
    Notes
    -----
    Type  ``jointeam spec`` in the dota2 console to observe the game
    """
    def __init__(self, path=None, dedicated=True):
        self.paths = DotaPaths(path)
        self.game_id = str(uuid.uuid1())
        self.game_mode = int(DOTA_GameMode.DOTA_GAMEMODE_AP)
        self.game_time_scale = 2

        self.dota_args = [self.paths.executable_path] + dota2_aguments(
            self.paths,
            self.game_id,
            self.game_mode,
            self.game_time_scale,
            dedicated
        )

        self.loop = asyncio.get_event_loop()
        self.async_tasks = None
        self.state = State()

    def launch_dota(self):
        # make sure the log is empty so we do not get garbage from the previous run
        if os.path.exists(self.paths.ipc_recv_handle):
            os.remove(self.paths.ipc_recv_handle)

        if os.path.exists(self.paths.ipc_send_handle):
            os.remove(self.paths.ipc_send_handle)

        # this is suffucient on windows
        # TODO check on linux
        subprocess.Popen(self.dota_args)

    def start_ipc(self):
        self.async_tasks = asyncio.gather(
            # State Capture
            worldstate_listener(PORT_TEAM_RADIANT, self.radiant_state, self.state),
            worldstate_listener(PORT_TEAM_DIRE, self.dire_state, self.state),

            # IPC receive
            ipc_recv(self.paths.ipc_recv_handle, self.receive_message, self.state)
        )

    def stop():
        self.state.running = False

    def wait(self):
        self.loop.run_until_complete(self.async_tasks)

    def receive_message(self, faction, player_id, message):
        """Receive a message directly from the bot"""
        print(f'{faction} {player_id} {message}')

    def dire_state(self, messsage):
        """Receive a state diff from the game for dire"""
        pass

    def radiant_state(self, message):
        """Receive a state diff from the game for radiant"""
        pass

    def send_message(self, data):
        """Send a message to the bots"""
        ipc_send(self.paths.ipc_send_handle, data)

    def cleanup(self):
        """Cleanup needed by the environment"""
        pass

    def __enter__(self):
        self.launch_dota()
        self.start_ipc()
        log.debug("Game has started")
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.cleanup()
        log.debug("Game has finished")


def main():
    from luafun.ipc_send import new_ipc_message
    logging.basicConfig(level=logging.DEBUG)

    game = Dota2Game('F:/SteamLibrary/steamapps/common/dota 2 beta/', False)

    with game:
        game.send_message(new_ipc_message())
    
        game.wait()

    print('Done')


if __name__ == '__main__':
    main()
