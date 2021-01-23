import asyncio
from collections import defaultdict
import logging
import os
import subprocess
import uuid

from luafun.game.config import DotaPaths
from luafun.game.args import DotaOptions
from luafun.game.modes import DOTA_GameMode
from luafun.game.http_inspect import http_inspect
from luafun.game.ipc_recv import ipc_recv
from luafun.game.ipc_send import ipc_send, TEAM_RADIANT, TEAM_DIRE
import luafun.game.dota2.state_types as msg
from luafun.game.states import worldstate_listener


log = logging.getLogger(__name__)


class State:
    """Simple Object used to propagate game state through the infra to stop components from running forever
    when the game finishes
    """
    def __init__(self):
        self.running = True


SECONDS_PER_TICK = 1 / 30

class Dota2Game:
    """Simple interface to listen and send messages to a running dota2 game instance
    This class only stich the different components together to provide a unified API over them
    You should subclass this to implement the desired behaviour

    Components
    ----------

    * world state listenner: receive state update about dire/radiant from the game itself
    * ipc_recv: receive message from each bot (through the console log)
    * ipc_send: send message to each bot (through a generated lua file)

    Notes
    -----
    Type  ``jointeam spec`` in the dota2 console to observe the game
    """
    def __init__(self, path=None, dedicated=True):
        self.paths = DotaPaths(path)
        self.options = DotaOptions(dedicated=dedicated)
        self.http_server = None
        self.args = None
        self.loop = asyncio.get_event_loop()
        # self.loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(self.loop)
        self.async_tasks = None
        self.state = State()
        self.process = None
        self.reply_count = defaultdict(int)
        self.bot_count = 10
        self.players = {
            TEAM_RADIANT: 0,
            TEAM_DIRE: 0
        }

    @property
    def deadline(self):
        """Return the inference time limit"""
        return SECONDS_PER_TICK * self.options.ticks_per_observation

    @property
    def running(self):
        return self.state.running

    def is_game_ready(self):
        return self.players[TEAM_RADIANT] + self.players[TEAM_DIRE] == self.bot_count

    def launch_dota(self):
        # make sure the log is empty so we do not get garbage from the previous run
        try:
            if os.path.exists(self.paths.ipc_recv_handle):
                os.remove(self.paths.ipc_recv_handle)
        except Exception as e:
            log.error(f'Error when removing file {e}')

        try:
            if os.path.exists(self.paths.ipc_send_handle):
                os.remove(self.paths.ipc_send_handle)
        except Exception as e:
            log.error(f'Error when removing file {e}')

        # save the arguments of the current game for visibility
        self.args = [self.paths.executable_path] + self.options.args(self.paths)
        self.process = subprocess.Popen(self.args )

    def start_ipc(self):
        self.async_tasks = asyncio.gather(
            # State Capture
            worldstate_listener(self.options.port_radiant, self.update_radiant_state, self),
            worldstate_listener(self.options.port_dire, self.update_dire_state, self),

            # IPC receive
            ipc_recv(self.paths.ipc_recv_handle, self._receive_message, self.state),

            # Debug HTTP server, so state can be inspected at runtime
            http_inspect(self)
        )

    def stop(self):
        """Stop the game in progress
        
        Notes
        -----
        On windows the dota2 game is not stopped but the underlying python processes are
        """
        self.state.running = False

        # Stop HTTP server
        if self.http_server is not None:
            log.debug('Stopping HTTP server')
        
            if hasattr(self.http_server, 'cancel'):
                self.http_server.cancel()
            else:
                self.http_server.close()

    def wait(self):
        """Wait for the asyncio coroutine to finish"""
        try:
            self.loop.run_until_complete(self.async_tasks)
        except Exception as e:
            if self.state.running:
                log.error(f'Error happened while game was running {e}')
            else:
                log.debug(f'Error happened on shutting down {e}')
        

    def _receive_message(self, faction: int, player_id: int, message: dict):
        # error processing
        error = message.get('E')
        if error is not None:
            log.error(f'recv {team_name(faction)} {player_id} {error}')
            return

        # init message
        info = message.get('P')
        if info is not None:
            self.players[int(faction)] += 1
            if self.is_game_ready():
                log.debug('All bots accounted for, Game is ready')
            return

        # Message ack
        ack = message.get('A')
        if ack is not None:
            self.reply_count[ack] += 1
            if self.reply_count[ack] == self.bot_count:
                log.debug(f'(uid: {ack}) message received by all {self.bot_count} bots')
                self.reply_count.pop(ack)
            return

        self.receive_message(faction, player_id, message)

    def receive_message(self, faction: int, player_id: int, message: dict):
        """Receive a message directly from the bot"""
        print(f'{faction} {player_id} {message}')

    async def update_dire_state(self, messsage: msg.CMsgBotWorldState):
        """Receive a state diff from the game for dire"""
        pass

    async def update_radiant_state(self, message: msg.CMsgBotWorldState):
        """Receive a state diff from the game for radiant"""
        pass

    def send_message(self, data: dict):
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
        if self.process.poll() is None:
            self.process.terminate()


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
