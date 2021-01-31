from datetime import datetime
import logging
import select
import socket
import time
import multiprocessing as mp
from struct import unpack
import traceback

from google.protobuf.json_format import MessageToJson

from luafun.game.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

log = logging.getLogger(__name__)


class SyncWorldListener:
    """Connect to the dota2 game and save the messages in a queue to be read"""

    def __init__(self, host, port, queue, state, stats, name):
        self.host = host
        self.port = port
        self.queue = queue
        self.sock = None
        self.state = state
        self.stats = stats
        self.name = name
        self.namespace = f'word-{self.name}'
        self.reason = None

    @property
    def running(self):
        return self.state['running']

    def connect(self, retries=20):
        pending = None

        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # this does not do anything
                # s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)     # Number of Keep alive probes
                # s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)    # Iddle time before keep alive proves are sent
                # s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)   # interval between keep alive probes
                # s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)     # disables the Nagle algorithm for TCP sockets

                s.setblocking(True)
                s.connect((self.host, self.port))
                log.debug(f'Connection established after {i} retries')
                return s

            except ConnectionRefusedError as err:
                if not self.running:
                    log.debug('Giving up; game has stopped')
                    return None

                pending = err
                time.sleep(1)
        else:
            log.debug('Could not establish connection')

            if pending is not None:
                raise pending

        return None

    def read_message(self, read: socket.socket):
        chunks = []
        bytes_recv = 0

        msg_size = b''
        retries = 0
        while msg_size == b'' and retries < 10:
            msg_size = read.recv(4)

            # means that the socket was closed
            # This mainly happens on windows every 1000 messages or so
            # this also means we are MISSING messages, windows
            # it critically flawed because of those disconnects
            if msg_size == b'':
                self.sock.close()
                self.sock = self.connect(10)
                self.reason = f'Reconnected to server'
                return None
            # ---
            retries += 1

        if msg_size == b'':
            self.reason = f'Message size is empty {msg_size} after {retries} retries'
            return None

        msg_len = int(unpack("@I", msg_size)[0])

        while bytes_recv < msg_len:
            chunk = read.recv(min(msg_len - bytes_recv, 8192))

            if chunk == b'':
                self.reason = f'Could not read rest of message (received: {bytes_recv}) (length: {msg_len})'
                return None

            chunks.append(chunk)
            bytes_recv = bytes_recv + len(chunk)

        if bytes_recv > msg_len:
            self.reason = f'Read more than necessary breaking communication (received: {bytes_recv}) (length: {msg_len})'
            return None

        msg = b''.join(chunks)
        world_state = CMsgBotWorldState()
        world_state.ParseFromString(msg)
        return world_state

    def insert_message(self, msg, s):
        json_msg = MessageToJson(
            msg,
            preserving_proto_field_name=True,
            use_integers_for_enums=True)

        self.queue.put(json_msg)
        self.state[self.namespace] = (datetime.utcnow() - s).total_seconds()

    def _run(self):
        # this needs to be lower than the game.deadline so in case of disconnect we can reconnect fast
        readable, _, error = select.select([self.sock], [], [self.sock], 0.05)

        for read in readable:
            s = datetime.utcnow()
            msg = self.read_message(read)

            if msg is not None:
                self.insert_message(msg, s)
            else:
                log.debug(f'Could not read message because: {self.reason}')
                self.reason = None

        for err in error:
            err.close()
            log.debug(f'socket error')
            self.sock = self.connect()

    def run(self):
        self.sock = self.connect()

        if self.sock is None:
            raise RuntimeError('Impossible to connect to the game')

        while self.running:
            try:
                self._run()

            except ConnectionResetError:
                # dota2 proabaly shutdown
                self.state['running'] = False
                log.error('Dota2 shutdown')

            except Exception as err:
                if self.running:
                    log.error(traceback.format_exc())

        if self.sock:
            self.sock.close()
        log.debug('World state listener shutting down')


def sync_world_listener(host, port, queue, state, stats, level, name):
    logging.basicConfig(level=level)
    wl = SyncWorldListener(host, port, queue, state, stats, name)
    wl.run()


def world_listener_process(host, port, queue, state, stats, name, level):
    p = mp.Process(
        name=f'WorldListener-{name}',
        target=sync_world_listener,
        args=(host, port, queue, state, stats, level, name)
    )
    p.start()
    log.debug(f'WorldListener-{name}: {p.pid}')
    return p
