from luafun.utils.options import option

import logging
import select
import socket

import traceback

log = logging.getLogger(__name__)


class ModelServer:
    def __init__(self):
        self.sock = None
        self.state = None

    @property
    def running(self):
        if self.state:
            return self.state['running']

        return True

    def connect(self):
        port = option('debug.port', 8080, int)
        host = "127.0.0.1"

        self.sock = socket.create_server((host, port))
        log.debug(f'Listening to {host}:{port}')

    def process_request(self, client):
        pass

    def _run(self):
        readable, _, error = select.select([self.sock], [], [self.sock], 0.250)

        for s in readable:
            if s is self.sock:
                client, client_address = s.accept()
                client.setblocking(1)
                self.process_request(client)

    def run(self):
        self.connect()

        while self.running:
            try:
                self._run()
            except Exception:
                log.error(traceback.format_exc())

        log.debug('http server shuting down')
        self.sock.close()


class ModelQueueClient:
    def __init__(self, queue):
        self.queue = queue

    def push(self, uid, state, reward, done, info):
        self.queue.push(('state', uid, state, reward, done, info))

    def infer(self, uid):
        return


class ModelQueueServer:
    def __init__(self, queue):
        self.queue = queue

    def run(self):
        while True:
            uid, state, reward, done, info = self.queue.get()



