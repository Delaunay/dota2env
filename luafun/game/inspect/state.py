import logging

from luafun.game.inspect.base import BasePage

import rpcjs.elements as html


log = logging.getLogger(__name__)

class GameInspector(BasePage):
    def routes(self):
        return [
            '/state/<string:faction>'
        ]

    def __init__(self, env, rpc_recv, rpc_send):
        self.env = env
        self.rpc_recv = rpc_recv
        self.rpc_send = rpc_send
        self.title = 'State'

    def main(self, faction):
        if faction.lower() in ('rad', 'radiant'):
            self.title = 'Radiant State'
            body = html.pre(self.getattr('radiant_state'))
        else:
            self.title = 'Dire State'
            body = html.pre(self.getattr('dire_state'))

        page = self.env.get_template('state.html')
        return page.render(code=body)
