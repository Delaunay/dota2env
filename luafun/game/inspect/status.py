from datetime import datetime
import logging
import multiprocessing as mp

from luafun.game.inspect.base import BasePage

import rpcjs.elements as html

log = logging.getLogger(__name__)


class Status(BasePage):
    def routes(self):
        return [
            '/status'
        ]

    def __init__(self, app):
        super(Status, self).__init__(app)
        self.title = 'Status'

    def main(self):
        ipc_recv = self.state.get('ipc_recv')
        dire_state = self.state.get('word-Dire')
        rad_state = self.state.get('word-Radiant')
        now = datetime.utcnow()

        kwargs = {
            'ipc_recv': 'NA',
            'dire_state': 'NA',
            'rad_state': 'NA'
        }
        if ipc_recv:
            kwargs['ipc_recv'] = now - ipc_recv

        if dire_state:
            kwargs['dire_state'] = now - dire_state

        if rad_state:
            kwargs['rad_state'] = now - rad_state

        page = self.env.get_template('status.html')
        return page.render(**kwargs, state=self.state)

