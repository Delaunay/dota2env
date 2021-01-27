import logging

from luafun.game.inspect.base import BasePage

from flask import request
import rpcjs.elements as html

log = logging.getLogger(__name__)


class Actions(BasePage):
    def routes(self):
        return [
            '/action/<string:action>'
        ]

    def __init__(self, env, state, rpc_recv, rpc_send):
        super(Actions, self).__init__(rpc_recv, rpc_send)
        self.title = 'State'
        self.state = state
        self.env = env

    def main(self, action):
        if action == 'stop':
            self.state['running'] = False

        if action == 'get_info':
            self.send_get_info()

        if action == 'send_move_action':
            self.send_get_info()

        page = self.env.get_template('state.html')
        return page.render(code='Stopped')

    def send_get_info(self):
        # here
        b = IPCMessageBuilder()
        p = b.player(0)
        p.MoveToLocation([0, 0])
        p.act[0] = 31

        m = b.build()
        # done

        log.debug(f'Sending message {m}')
        self.rpc_send.put(dict(attr='send_message', args=[m]))
        obj = self.fetch()
        # --

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


