from dataclasses import dataclass
import logging
import multiprocessing as mp
from struct import unpack
import traceback
import json

from luafun.httpserver.server import HttpServer
from luafun.utils.python_fix import asdict
from luafun.game.action import IPCMessageBuilder


log = logging.getLogger(__name__)


class GameInspector(HttpServer):
    def __init__(self, state, rpc_recv, rpc_send):
        super(GameInspector, self).__init__(state)
        self.rpc_recv = rpc_recv
        self.rpc_send = rpc_send

        self.routes['/dire_state_delta'] = self.getattr('dire_state_delta')
        self.routes['/radiant_state_delta'] = self.getattr('radiant_state_delta')
        self.routes['/dire_state'] = self.getattr('dire_state', 'state.html')
        self.routes['/radiant_state'] = self.getattr('dire_state', 'state.html')
        self.routes['/send_move_action'] = self.send_move_action

    def send_move_action(self, request):
        # here
        b = IPCMessageBuilder()
        p = b.player(0)
        p.MoveToLocation([0, 0])
        m = b.build()
        # done

        log.debug(f'Sending message {m}')
        self.rpc_send.put(dict(attr='send_message', args=[m]))
        obj = self.fetch()
        # --

        return self.default_route(request)

    def fetch(self):
        while self.running:
            try:
                return self.rpc_recv.get(timeout=0.250)
            except Exception as err:
                log.error(f'Error {err}')

        return None

    def getattr(self, attr_name, template=None):
        def rpc_attr(_):
            self.rpc_send.put(dict(attr=attr_name))
            obj = self.fetch()

            if obj is None:
                return self.html('could not fetch reply')

            if not isinstance(obj, (str, dict)):
                reply = json.dumps(asdict(obj), indent=2)

            if template is None:
                return self.html(f'<pre>{reply}</pre>')
            else:
                t = self.env.get_template(template)
                return t.render(obj=obj, code=reply)

        return rpc_attr


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
    log.debug(f'HTTP-server: {p.pid}')
    return p
