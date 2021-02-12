import logging

from rpcjs.page import Page

log = logging.getLogger(__name__)


class BasePage(Page):
    def __init__(self, app):
        self.app = app
        self.rpc_recv = app.rpc_recv
        self.rpc_send = app.rpc_send
        self.state = app.state
        self.env = app.env

    def fetch(self):
        """Get the result of the rpc call"""
        while self.state['running']:
            try:
                return self.rpc_recv.get(timeout=0.250)
            except Exception as err:
                log.debug(f'RCV timed-out')
        return None

    def getattr(self, attr_name):
        """Fetch a game attribute using the queues"""
        self.rpc_send.put(dict(attr=attr_name))
        return self.fetch()
