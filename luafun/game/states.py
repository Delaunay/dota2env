import logging
import select
import socket
import multiprocessing as mp
from struct import unpack
import traceback

from google.protobuf.json_format import MessageToJson

from luafun.game.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

log = logging.getLogger(__name__)


class SyncWorldListener:
    """Connect to the dota2 game and save the messages in a queue to be read"""

    def __init__(self, host, port, queue, state, stats):
        self.host = host
        self.port = port
        self.queue = queue
        self.sock = None
        self.state = state
        self.stats = stats

    @property
    def running(self):
        return self.state['running']

    def connect(self, retries=10):
        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setblocking(True)
                s.connect((self.host, self.port))
                s.setblocking(True)

                log.debug(f'Connection established after {i} retries')
                return s

            except ConnectionRefusedError:
                if not self.running:
                    return None

        else:
            log.debug('Could not establish connection')

        return None

    def read_message(self, read):
        chunks = []
        bytes_recv = 0

        msg_size = read.recv(4)
        if msg_size == b'':
            return None

        msg_len = int(unpack("@I", msg_size)[0])

        while bytes_recv < msg_len:
            chunk = read.recv(min(msg_len - bytes_recv, 8192))

            if chunk == b'':
                log.debug('Could not read message')
                return None

            chunks.append(chunk)
            bytes_recv = bytes_recv + len(chunk)

        msg = b''.join(chunks)
        world_state = CMsgBotWorldState()
        world_state.ParseFromString(msg)
        return world_state

    def _run(self):
        readable, _, error = select.select([self.sock], [], [self.sock], 0.250)

        for read in readable:
            msg = self.read_message(read)

            if msg is not None:
                json_msg = MessageToJson(
                    msg, 
                    preserving_proto_field_name=True, 
                    use_integers_for_enums=True)

                self.queue.put(json_msg)

        for err in error:
            err.close()
            s = self.connect()

    def run(self):
        self.sock = self.connect()

        while self.running:
            try:
                self._run()

            except ConnectionResetError:
                # dota2 proabaly shutdown
                self.state['running'] = False

            except Exception as err:
                if self.running:
                    log.error(traceback.format_exc())

        self.sock.close()
        log.debug('World state listener shutting down')


def sync_world_listener(host, port, queue, state, stats, level):
    logging.basicConfig(level=level)
    wl = SyncWorldListener(host, port, queue, state, stats)
    wl.run()


def world_listener_process(host, port, queue, state, stats, name, level):
    p = mp.Process(
        name=f'WorldListener-{name}',
        target=sync_world_listener,
        args=(host, port, queue, state, stats, level)
    )
    p.start()
    return p
