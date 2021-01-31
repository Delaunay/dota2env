import logging
import json
import multiprocessing as mp

from luafun.game.inspect.base import BasePage
from luafun.game.action import IPCMessageBuilder, Action

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
            '/action/<string:action>',
            '/action/<string:action>/<int:player>',
            '/action/<string:action>/<int:player>/<string:item>',
            '/action/<string:action>/<int:player>/<float:x>x<float:y>',
            '/action/<string:action>/<int:player>/<int:slot>',
            '/action/<string:action>/<int:player>/<int:slot>/<float:x>x<float:y>',
            '/action/<string:action>/<int:player>/<int:slot>/<int:slot2>',
        ]

    def __init__(self, app):
        super(Actions, self).__init__(app)
        self.title = 'Action'
        self.process = None
        self.base_actions = {
            # vloc actions
            'MoveToLocation': self.make_vloc_action('MoveToLocation'),
            'MoveDirectly': self.make_vloc_action('MoveDirectly'),
            'AttackMove': self.make_vloc_action('AttackMove'),

            # No argument action
            'CourierTransfer': self.make_noarg_action('CourierTransfer'),
            'CourierTakeStash': self.make_noarg_action('CourierTakeStash'),
            'CourierSecret': self.make_noarg_action('CourierSecret'),
            'CourierReturn': self.make_noarg_action('CourierReturn'),
            'Stop': self.make_noarg_action('Stop'),

            # Ability is hidden
            # 'CourierEnemySecret': self.make_noarg_action('CourierEnemySecret'),
            'CourierBurst': self.make_noarg_action('CourierBurst'),
            'TakeOutpost': self.make_noarg_action('TakeOutpost'),
            'Glyph': self.make_noarg_action('Glyph'),
            'Buyback': self.make_noarg_action('Buyback'),

            # Ability / slot
            'UseAbility': self.make_item_action('UseAbility'),
            'UseAbilityOnLocation': self.location_ability,
            'DropItem': self.drop_item,
            'SellItem': self.make_item_action('SellItem'),
            'DisassembleItem': self.make_item_action('DisassembleItem'),
            'SetItemCombineLock': self.make_item_action('SetItemCombineLock'),
            'LevelAbility': self.make_item_action('LevelAbility'),
            # use rune id
            'PickUpRune': self.make_item_action('PickUpRune'),
            'SwapItems': self.swap_item,

            # Entity Handle
            # -------------
            'MoveToUnit': self.make_unit_action('MoveToUnit'),
            'AttackUnit': self.make_unit_action('AttackUnit'),
            'PickUpItem': self.make_unit_action('PickUpItem'),
            'UseAbilityOnEntity': self.ability_on_entity,

            # Trees
            'UseAbilityOnTree': self.ability_on_tree,

            # Item name
            'PurchaseItem': self.purchase_item,

            # No ops
            'NotUsed1': self.noop,
            'NotUsed2': self.noop,
            'NotUsed3': self.noop,
            'NotUsed4': self.noop,
            'NotUsed5': self.noop,
        }

        if len(self.base_actions) != len(Action):
            print(f'Missing action implementations: {len(self.base_actions)} < {len(Action)}')

    def purchase_item(self, player, item, **kwargs):
        b = IPCMessageBuilder()
        p = b.player(player)
        p.PurchaseItem(item)
        m = b.build()
        return self.send_action(m)

    def noop(self, *args, **kwargs):
        page = self.env.get_template('state.html')
        return page.render(
            code=f'No Op',
            state=self.state)

    def ability_on_entity(self, player, slot=0, slot2=0, **kwargs):
        b = IPCMessageBuilder()
        p = b.player(player)
        p.UseAbilityOnEntity(slot, slot2)
        m = b.build()
        return self.send_action(m)

    def ability_on_tree(self, player, slot=0, slot2=0, **kwargs):
        b = IPCMessageBuilder()
        p = b.player(player)
        p.UseAbilityOnTree(slot, slot2)
        m = b.build()
        return self.send_action(m)

    def make_unit_action(self, name):
        def send_item_action(player, slot=0, **kwargs):
            b = IPCMessageBuilder()
            p = b.player(player)
            getattr(p, name)(slot)
            m = b.build()
            return self.send_action(m)

        return send_item_action

    def swap_item(self, player, slot=0, slot2=0, **kwargs):
        b = IPCMessageBuilder()
        p = b.player(player)
        p.SwapItems(slot, slot2)
        m = b.build()
        return self.send_action(m)

    def location_ability(self, player, slot=0, x=0, y=0, **kwargs):
        b = IPCMessageBuilder()
        p = b.player(player)
        p.UseAbilityOnLocation(hAbility=slot, vLoc=[x, y])
        m = b.build()
        return self.send_action(m)

    def drop_item(self, player, slot=0, x=0, y=0, **kwargs):
        b = IPCMessageBuilder()
        p = b.player(player)
        p.DropItem(slot, [x, y])
        m = b.build()
        return self.send_action(m)

    def make_item_action(self, name):
        def send_item_action(player, slot=0, **kwargs):
            b = IPCMessageBuilder()
            p = b.player(player)
            getattr(p, name)(slot)
            m = b.build()
            return self.send_action(m)

        return send_item_action

    def sell_item(self, player, slot=0, **kwargs):
        b = IPCMessageBuilder()
        p = b.player(player)
        p.SellItem(slot)
        m = b.build()
        return self.send_action(m)

    # this shows the minimal number of arguments for the action table in lua +tree
    def main(self, action, player=0, x=0, y=0, slot=0, slot2=0, item='item_gauntlets'):
        if action == 'stop':
            self.state['running'] = False

        if action == 'info':
            return self.send_get_info()

        if action in self.base_actions:
            return self.base_actions[action](player, x=x, y=y, slot=slot, slot2=slot2, item=item)

        if action == 'play':
            return self.start()

        page = self.env.get_template('state.html')
        return page.render(
            code=f'Route did not match anything {player}'
                 f'(player: {player}) (x: {x}) (y: {y}) (slot: {slot})',
            state=self.state)

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

    def send_action(self, action):
        self.rpc_send.put(dict(attr='send_message', args=[action]))
        _ = self.fetch()

        page = self.env.get_template('base.html')
        return page.render(body=f'<pre>{json.dumps(action, indent=2)}</pre>', state=self.state)

    def send_get_info(self):
        b = IPCMessageBuilder()
        p = b.player(0)
        p.MoveToLocation([0, 0])
        p.act[0] = 31
        m = b.build()
        return self.send_action(m)

    def make_vloc_action(self, name):
        def send_vloc_action(player, x, y, **kwargs):
            b = IPCMessageBuilder()
            p = b.player(player)
            getattr(p, name)([x, y])
            m = b.build()
            return self.send_action(m)

        return send_vloc_action

    def make_noarg_action(self, name):
        def send_noarg_action(player, **kwargs):
            b = IPCMessageBuilder()
            p = b.player(player)
            getattr(p, name)()
            m = b.build()
            return self.send_action(m)
        return send_noarg_action

