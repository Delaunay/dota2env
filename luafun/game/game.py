from collections import defaultdict
from dataclasses import dataclass, field
import logging
import os
import json
import multiprocessing as mp
import subprocess
import time
import traceback
import uuid

from luafun.game.config import DotaPaths
from luafun.game.args import DotaOptions
from luafun.game.modes import DOTA_GameMode
from luafun.game.http_inspect import http_inspect
from luafun.game.ipc_recv import ipc_recv
from luafun.game.ipc_send import ipc_send, TEAM_RADIANT, TEAM_DIRE
import luafun.game.dota2.state_types as msg
from luafun.game.states import world_listener_process


log = logging.getLogger(__name__)


@dataclass
class WorldConnectionStats:
    message_size: int = 0
    success: int = 0
    error: int = 0
    reconnect: int = 0
    double_read: int = 0


@dataclass
class Stats:
    radiant: WorldConnectionStats = field(default_factory=WorldConnectionStats)
    dire: WorldConnectionStats = field(default_factory=WorldConnectionStats)


SECONDS_PER_TICK = 1 / 30

TEAM_NAMES = {
    TEAM_RADIANT: 'Radiant',
    str(TEAM_RADIANT): 'Radiant',
    TEAM_DIRE: 'Dire',
    str(TEAM_DIRE): 'Dire',
}


def team_name(v):
    return str(TEAM_NAMES.get(v))


class Dota2Game:
    """Simple interface to listen and send messages to a running dota2 game instance
    This class only stich the different components together to provide a unified API over them
    You should subclass this to implement the desired behaviour
    No ML related feature there

    Components
    ----------

    * world state listenner: receive state update about dire/radiant from the game itself
    * ipc_recv: receive message from each bot (through the console log)
    * ipc_send: send message to each bot (through a generated lua file)
    * http server: used to inspect the game in realitime

    6 Processes are created when launching the environment

    .. code-block::

        1) Main Process             : 29824 | 1% CPU | stich state together
        2) WorldListener-Dire       : 26272 | 4% CPU | retrieve game state
        3) WorldListener-Radiant    : 33228 | 4% CPU | retrieve game state
        4) IPC-recv                 : 28848 | 0% CPU | Read Game logs for bot errors
        5) HTTP-server              : 30424 | 0% CPU | Debug Process
        6) Multiprocess Manager

    Notes
    -----
    Type  ``jointeam spec`` in the dota2 console to observe the game

    We use multiprocess, asyncio was not given the required performance.
    A huge part of performance is used to receive messages from the game itself
    """
    def __init__(self, path=None, dedicated=True):
        self.paths = DotaPaths(path)
        self.options = DotaOptions(dedicated=dedicated)
        self.args = None

        self.state = None
        self.process = None
        self.reply_count = defaultdict(int)
        self.manager = None

        self.dire_state_process = None
        self.radiant_state_process = None

        self.dire_state_delta_queue = None
        self.radiant_state_delta_queue = None

        self.ipc_recv_process = None
        self.ipc_recv_queue = None

        self.http_server = None
        self.http_rpc_send = None
        self.http_rpc_recv = None

        self.bot_count = 10
        self.stats = Stats()
        self.players = {
            TEAM_RADIANT: 0,
            TEAM_DIRE: 0
        }

        self.cache = dict()
        self.store = open('exporting_trees.json', 'w')

        log.debug(f'Main Process: {os.getpid()}')

    @property
    def deadline(self):
        """Return the inference time limit"""
        return SECONDS_PER_TICK * self.options.ticks_per_observation

    @property
    def running(self):
        return self.state['running']

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
        self.process = subprocess.Popen(self.args)

    def dire_state_delta(self):
        if not self.dire_state_delta_queue.empty():
            delta = self.dire_state_delta_queue.get()
            return delta

        return None

    def radiant_state_delta(self):
        if not self.radiant_state_delta_queue.empty():
            delta = self.radiant_state_delta_queue.get()
            return delta

        return None

    def start_ipc(self):
        self.manager = mp.Manager()
        self.state = self.manager.dict()
        self.state['running'] = True
        level = logging.DEBUG

        # Dire State
        self.dire_state_delta_queue = self.manager.Queue()
        self.dire_state_process = world_listener_process(
            '127.0.0.1',
            self.options.port_dire,
            self.dire_state_delta_queue,
            self.state,
            None,
            'Dire',
            level
        )

        # Radiant State
        self.radiant_state_delta_queue = self.manager.Queue()
        self.radiant_state_process = world_listener_process(
            '127.0.0.1',
            self.options.port_radiant,
            self.radiant_state_delta_queue,
            self.state,
            None,
            'Radiant',
            level
        )

        # IPC receive
        self.ipc_recv_queue = self.manager.Queue()
        self.ipc_recv_process = ipc_recv(
            self.paths.ipc_recv_handle,
            self.ipc_recv_queue,
            self.state,
            level
        )

        # HTTP server
        self.http_rpc_recv = self.manager.Queue()
        self.http_rpc_send = self.manager.Queue()
        self.http_server = http_inspect(
            self.state,
            self.http_rpc_send,
            self.http_rpc_recv,
            level
        )

    def stop(self):
        """Stop the game in progress

        Notes
        -----
        On windows the dota2 game is not stopped but the underlying python processes are
        """
        self.state['running'] = False

        if self.process.poll() is None:
            self.process.terminate()

    def _handle_http_rpc(self):
        # handle debug HTTP request
        if self.http_rpc_recv.empty():
            return

        msg = self.http_rpc_recv.get()
        attr = msg.get('attr')
        args = msg.get('args', [])
        kwargs = msg.get('kwargs', dict())

        result = dict(error=f'Object does not have attribute {attr}')

        if hasattr(self, attr):
            result = getattr(self, attr)(*args, **kwargs)

        if result is None:
            result = dict(msg='none')

        self.http_rpc_send.put(result)

    def _handle_ipc(self):
        if self.ipc_recv_queue.empty():
            return

        msg = self.ipc_recv_queue.get()
        self._receive_message(*msg)

    def _handle_state(self):
        dire_delta = self.dire_state_delta()
        rad_delta = self.radiant_state_delta()

        if dire_delta is not None:
            self.update_dire_state(dire_delta)

        if rad_delta is not None:
            self.update_radiant_state(rad_delta)

    def wait(self):
        """Wait for the game to finish, this is used for debugging exclusively"""
        try:
            while self.process.poll() is None:
                time.sleep(0.01)

                # Process event
                if not self.running:
                    self.stop()

                winner = self.state.get('win')
                if winner is not None:
                    log.debug(f'{winner} won')
                    self.stop()
                # ---

                self._handle_http_rpc()
                self._handle_ipc()
                self._handle_state()

        except KeyboardInterrupt:
            pass

        self.stop()

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

        # Message Info
        info = message.get('I')
        if info is not None:
            tid = info[0]

            if tid not in self.cache:
                self.store.write(json.dumps(info))
                self.store.write('\n')
                self.cache[tid] = True
                log.debug(info)

            return

        self.receive_message(faction, player_id, message)

    def receive_message(self, faction: int, player_id: int, message: dict):
        """Receive a message directly from the bot"""
        print(f'{faction} {player_id} {message}')

    def update_dire_state(self, messsage: msg.CMsgBotWorldState):
        """Receive a state diff from the game for dire"""
        pass

    def update_radiant_state(self, message: msg.CMsgBotWorldState):
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
        self.stop()

        self.dire_state_process.join()
        self.radiant_state_process.join()
        self.ipc_recv_process.join()
        self.http_server.join()

        # ---
        self.store.close()
        # --

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
