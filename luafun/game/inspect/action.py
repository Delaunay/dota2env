import logging
import multiprocessing as mp

from luafun.game.inspect.base import BasePage

from flask import request
import rpcjs.elements as html

log = logging.getLogger(__name__)


class Config:
    def __init__(self, app):
        # this is supposed to be inverted, not a bug
        self.rpc_recv = app.rpc_send
        self.rpc_send = app.rpc_recv
        self.state = app.state

class Actions(BasePage):
    def routes(self):
        return [
            '/action/<string:action>'
        ]

    def __init__(self, app):
        super(Actions, self).__init__(app)
        self.title = 'State'
        self.process = None

    def main(self, action):
        if action == 'stop':
            self.state['running'] = False

        if action == 'get_info':
            return self.send_get_info()

        if action == 'send_move_action':
            return self.send_get_info()

        if action == 'play':
            return self.start()

        page = self.env.get_template('state.html')
        return page.render(code='Stopped', state=self.state)

    def start(self):
        """Start a new environment running"""
        from luafun.dotaenv import main

        config = Config(self.app)
        self.state['running'] = True
        self.process = mp.Process(
            target=main,
            args=(config,)
        )
        self.process.start()
        page = self.env.get_template('state.html')
        return page.render(code='Starting', state=self.state)

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
