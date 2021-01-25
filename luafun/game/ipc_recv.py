import asyncio
import json
import logging
import os
import multiprocessing as mp
import re
import time
import traceback

from pygtail import Pygtail

log = logging.getLogger(__name__)

IPC_RECV = re.compile(r'\[IPC\](?P<faction>[0-9])\.(?P<player>[0-9])\t(?P<message>.*)')
DIRE_WIN = re.compile(r'Building: npc_dota_goodguys_fort destroyed at')
RADIANT_WIN = re.compile(r'Building: npc_dota_badguys_fort destroyed at')


class IPCRecv:
    def __init__(self, logfilename, queue, state):
        self.logfilename = logfilename
        self.queue = queue
        self.state = state
        self.got_messages = False

        # remove offset from previous game
        try:
            os.remove(f'{logfilename}.offset')
        except:
            pass
        # ---

    @property
    def running(self):
        return self.state['running']

    def connect(self, retries=10):
        # wait for the file to be created
        for i in range(retries):
            if os.path.exists(self.logfilename):
                log.debug(f'Starting IPC recv after {i} retries')
                break

            time.sleep(0.05)
        else:
            msg = 'Impossible to initialize IPC recv'
            log.error(msg)
            raise RuntimeError(msg)

    def _run(self):
        self.got_messages = False
        win = None

        for line in Pygtail(self.logfilename):
            result = IPC_RECV.search(line)

            # this happens 99.99% of times so we do it first
            if result:
                self.got_messages = True
                msg = result.groupdict()
                self.queue.put((msg.get('faction'), msg.get('player'), json.loads(msg.get('message'))))
                continue

            # this only happen once per game
            result = DIRE_WIN.search(line)
            if result:
                self.state['win'] = 'DIRE'
                continue

            result = RADIANT_WIN.search(line)
            if result:
                self.state['win'] = 'RADIANT'
                continue

    def run(self):
        while self.running:
            try:
                self._run()

                if not self.got_messages:
                    time.sleep(0.13)

            except Exception as e:
                time.sleep(0.01)
                log.debug(f'IPC error {e}')
                log.error(traceback.format_exc())

        log.debug('IPC recv finished')


def _ipc_recv(logfilename, queue, state, level, retries=10):
    """The only way for bots to send message back is through the log file
    we have standardized our log lines so we know which bot is sending us a message
    """
    logging.basicConfig(level=level)
    recv = IPCRecv(logfilename, queue, state)
    recv.connect()
    recv.run()


def ipc_recv(logfilename, queue, state, level, retries=10):
    p = mp.Process(
        name='IPC-recv',
        target=_ipc_recv,
        args=(logfilename, queue, state, level, retries)
    )

    p.start()
    log.debug(f'IPC-recv: {p.pid}')
    return p
