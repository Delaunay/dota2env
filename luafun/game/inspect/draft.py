import logging

from luafun.game.inspect.base import BasePage
from luafun.game.action import IPCMessageBuilder


log = logging.getLogger(__name__)


class Draft(BasePage):
    def routes(self):
        return [
            '/draft',
            '/draft/<int:faction>/<string:action>/<string:hero>',
        ]

    def __init__(self, app):
        super(Draft, self).__init__(app)
        self.title = 'State'
        self.latest = 0

    def send_action(self, action):
        self.rpc_send.put(dict(attr='send_message', args=[action]))
        _ = self.fetch()

    def main(self, faction=None, action=None, hero=None):
        if faction and action and hero:
            b = IPCMessageBuilder()
            draft = b.hero_selection(faction)
            if action == 'select':
                draft.select(hero, 0)
            else:
                draft.ban(hero)
            self.send_action(b.build())

        page = self.env.get_template('draft.html')
        return page.render(code='Drafting', state=self.state)
