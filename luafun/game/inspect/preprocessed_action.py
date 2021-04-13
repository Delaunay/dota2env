import logging
import json
import multiprocessing as mp

from luafun.game.inspect.base import BasePage
from luafun.game.action import ARG, new_ipc_message, Action, AbilitySlot


log = logging.getLogger(__name__)


class Config:
    def __init__(self, app):
        # this is supposed to be inverted, not a bug
        self.rpc_recv = app.rpc_send
        self.rpc_send = app.rpc_recv
        self.state = app.state


class PreprocessedActions(BasePage):
    """Used to check that the (x, y) mapper, item mapper, ability mapper work"""
    def routes(self):
        return [
            '/paction',
            '/paction/<int:player>/<int:action>/<float:x>/<float:y>/<int:item>/<int:nslot>/<int:ix2>',
        ]

    TEAM_RAD = [0, 1, 2, 3, 4]
    TEAM_DIRE = [5, 6, 7, 8, 9]
    FACTION = {
        0: 2, 1: 2, 2: 2, 3: 2, 4: 2,
        5: 3, 6: 3, 7: 3, 8: 3, 9: 3,
    }

    def main(self, player=None, action=None, x=None, y=None, item=None, nslot=None, ix2=None):
        page_args = dict(
            AbilitySlot=AbilitySlot,
            Action=Action
        )
        if action is not None:
            ipc_msg = new_ipc_message()
            ipc_msg[self.FACTION[player]][player] = {
                ARG.action: action,
                ARG.vLoc: (x, y),
                ARG.sItem: item,
                ARG.nSlot: nslot,
                ARG.ix2: ix2,
            }

            self.rpc_send.put(dict(attr='preprocessed_send', args=[ipc_msg]))
            preprocessed = self.fetch()

            page = self.env.get_template('preprocessed.html')

            body = list()
            body.append(f'<h2>Original</h2>')
            body.append(f'<pre>{json.dumps(ipc_msg, indent=2)}</pre>')
            body.append(f'<h2>Preprocessed</h2>')
            body.append(f'<pre>{json.dumps(preprocessed, indent=2)}</pre>')

            return page.render(body=''.join(body), state=self.state, player=player, **page_args)

        page = self.env.get_template('preprocessed.html')
        return page.render(body='', state=self.state, player=0, **page_args)

    def __init__(self, app):
        super(PreprocessedActions, self).__init__(app)
        self.title = 'Action'
        self.process = None
