from dataclasses import dataclass
import logging
import multiprocessing as mp
from struct import unpack
import traceback
import json

from luafun.httpserver.server import HttpServer
from luafun.utils.python_fix import asdict


log = logging.getLogger(__name__)


class GameInspector(HttpServer):
    def __init__(self, state, rpc_recv, rpc_send):
        super(GameInspector, self).__init__(state)
        self.rpc_recv = rpc_recv
        self.rpc_send = rpc_send

    def fetch(self):
        while self.running:
            try:
                return self.rpc.recv.get(timeout=0.250)
            except queue.Empty:
                pass
        
        return None

    def show_dire_state(self):
        self.rpc.send(dict(attr='dire_state'))
        reply = self.fetch()

        if reply is None:
            return self.html('could not fetch reply')

        json_state = json.dumps(reply, indent=2)
        return self.html(f'<pre>{json_state}</pre>')


def _http_inspect(state, rpc_recv, rpc_send, level):
    logging.basicConfig(level=level)
    s = GameInspector(state, rpc_recv, rpc_send)
    s.run()


def http_inspect(state, rpc_recv, rpc_send, level):
    p = mp.Process(
        target=_http_inspect, 
        args=(state, rpc_recv, rpc_send, level)
    )
    p.start()
    return p
