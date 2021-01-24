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


class IPCRecv:
    def __init__(self, logfilename, queue, state):
        self.logfilename = logfilename
        self.queue = queue
        self.state = state

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
        for line in Pygtail(self.logfilename):
            result = IPC_RECV.search(line)

            if result:
                msg = result.groupdict()
                self.queue.put((msg.get('faction'), msg.get('player'), json.loads(msg.get('message'))))

    def run(self):
        while self.running:
            try:
                self._run()
            
            except Exception as e:
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
    return p
