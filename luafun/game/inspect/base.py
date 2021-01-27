import json

from luafun.utils.python_fix import asdict

from rpcjs.page import Page


class BasePage(Page):
    def __init__(self, rpc_recv, rpc_send):
        self.rpc_recv = rpc_recv
        self.rpc_send = rpc_send

    def fetch(self):
        """Get the result of the rpc call"""
        try:
            return self.rpc_recv.get(timeout=0.250)
        except Exception as err:
            log.error(f'Error {err}')
        return None

    def getattr(self, attr_name):
        """Fetch a gamr attribute using the queues"""
        self.rpc_send.put(dict(attr=attr_name))
        obj = self.fetch()

        if obj is None:
            return 'None'

        if not isinstance(obj, (str, dict)):
            reply = json.dumps(asdict(obj), indent=2)

        return reply
