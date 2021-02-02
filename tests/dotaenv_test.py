import logging
import time

import pytest

from luafun.game.modes import DOTA_GameMode
from luafun.dotaenv import Dota2Env, guess_path
from luafun.utils.options import option
from luafun.game.action import IPCMessageBuilder, SpellSlot

logging.basicConfig(level=logging.DEBUG)


class DotaTestEnvironment(Dota2Env):
    def __init__(self, send_actions, check_action, action_wait=0.26, timeout=1, **kwargs):
        super(DotaTestEnvironment, self).__init__(option('dota.path', guess_path()), True)

        self.options.game_mode = int(DOTA_GameMode.DOTA_GAMEMODE_AP)
        self.options.ticks_per_observation = 4
        self.options.host_timescale = 2

        self.err = None
        self.action_sent = False
        self.check_action = check_action
        self.send_actions = send_actions
        self.action_checked = False
        self.kwargs = kwargs
        self.timeout = 0
        self.timeout_deadline = timeout
        self.action_wait = action_wait

    def send_all_actions(self):
        for send_action in self.send_actions:
            ds = self.dire_state()
            rs = self.radiant_state()

            msg = send_action(ds, rs, **self.kwargs)
            self.send_message(msg)

            time.sleep(self.action_wait)
            self._tick()

        self.action_sent = True

    def check_all_actions(self):
        ds = self.dire_state()
        rs = self.radiant_state()

        self.timeout = 0
        while (not self.check_action(ds, rs, **self.kwargs)) and self.timeout < self.timeout_deadline :
            time.sleep(0.01)
            self._tick()

            ds = self.dire_state()
            rs = self.radiant_state()
            self.timeout += 0.01

            # during testing we might run in UI mode
            if self.process.poll() is not None:
                break

        self.action_checked = True

    def exec(self):
        try:
            while self.process.poll() is None:
                stop = False
                time.sleep(0.01)

                # Send
                # ---
                if self.ready:
                    if not self.options.dedicated:
                        time.sleep(10)

                    try:
                        self.send_all_actions()
                    except Exception as err:
                        stop = True
                        self.err = err
                else:
                    stop = self._tick()
                # ---

                # Check
                # --
                if self.action_sent and self.ready:
                    try:
                        self.check_all_actions()
                    except Exception as err:
                        self.err = err
                    stop = True
                # --

                if stop:
                    self.stop()
                    break
        except KeyboardInterrupt:
            pass


def run_env(send_actions, check_action, timeout=1, action_wait=0.26, **kwargs):
    game = DotaTestEnvironment(send_actions, check_action, action_wait, timeout, **kwargs)

    with game:
        game.exec()
        game.wait()

    if game.err is not None:
        raise game.err

    if game.timeout >= game.timeout_deadline:
        raise RuntimeError('Action was not registered on time')

    if not game.action_checked:
        raise RuntimeError('Environment closed before test could be done')


def get_player(pid, ds, rs):
    if pid < 5:
        return rs._players.get(int(pid))
    return ds._players.get(int(pid))


def get_courier(pid, ds, rs):
    if pid < 5:
        return rs._couriers.get(int(pid))
    return ds._couriers.get(int(pid))


def get_distance(p, point):
    x = p['location']['x'] - point[0]
    y = p['location']['y'] - point[1]

    return x * x + y * y


TEAM_RADIANT = 2
TEAM_DIRE = 3


def get_team(pid):
    if pid < 5:
        return TEAM_RADIANT
    return TEAM_DIRE


def get_friendly(pid, ds, rs):
    s = ds
    if pid < 5:
        s = rs

    for handle, building in s._buildings.items():
        if building.get('team_id', 0) == get_team(pid):
            p = building.get('location')
            return handle, (p['x'], p['y'])


@pytest.mark.parametrize('function', ['MoveToLocation', 'MoveDirectly', 'AttackMove'])
def test_move_to_loc(function, pid=0, pos=(0, 0)):
    def send_action(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        getattr(p, function)(pos)

        s = get_player(pid, ds, rs)
        state['d'] = get_distance(s, pos)
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)
        distance = get_distance(s, pos)
        return distance < state['d']

    state = dict()
    run_env([send_action], was_success, state=state)


# AttackUnit needs an actual enemy this only happens after 1:30 minutes of game
@pytest.mark.parametrize('function', ['MoveToUnit'])
def test_move_to_unit(function, pid=0):
    def send_action(ds, rs, state):
        unit, pos = get_friendly(pid, ds, rs)

        b = IPCMessageBuilder()
        p = b.player(pid)
        getattr(p, function)(unit)

        s = get_player(pid, ds, rs)
        state['d'] = get_distance(s, pos)
        state['p'] = pos

        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)
        distance = get_distance(s, state['p'])
        return distance < state['d']

    state = dict()
    run_env([send_action], was_success, state=state)


def test_learn_ability(pid=0):
    def learn_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.LevelAbility(SpellSlot.Q)
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)
        ability = None

        for a in s.get('abilities'):
            if a['slot'] == 0:
                ability = a
                break

        return ability is not None and ability['level'] == 1

    state = dict()
    run_env([learn_ability], was_success, state=state)


def test_use_ability(pid=0):
    # Player 1 is antimage, using counter spell
    def learn_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.LevelAbility(SpellSlot.W)
        return b.build()

    def use_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.UseAbility(SpellSlot.W)
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)

        ability = None
        for a in s.get('abilities'):
            if a['slot'] == 1:
                ability = a
                break

        return ability is not None and ability['cooldown_remaining'] > 0

    state = dict()
    run_env([learn_ability, use_ability], was_success, state=state)


# This tries to stop a cast mid animation
@pytest.mark.skip
def test_stop_ability(pid=0):
    # Player 1 is antimage, using counter spell
    def learn_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.LevelAbility(SpellSlot.E)
        return b.build()

    def use_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.UseAbility(SpellSlot.E)
        return b.build()

    def stop(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.Stop()
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)

        ability = None
        for a in s.get('abilities'):
            if a['slot'] == 2:
                ability = a
                break

        print(ability['cooldown_remaining'])
        return ability is not None and ability['cooldown_remaining'] <= 0.001

    state = dict()
    run_env([learn_ability, use_ability, stop], was_success, action_wait=0.1, state=state)


def test_use_ability_loc(pid=0, pos=(0, 0)):
    # Player 1 is antimage, using counter spell
    def learn_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.LevelAbility(SpellSlot.W)
        return b.build()

    def use_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.UseAbilityOnLocation(SpellSlot.W, pos)
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)

        ability = None
        for a in s.get('abilities'):
            if a['slot'] == 1:
                ability = a
                break

        return ability is not None and ability['cooldown_remaining'] > 0

    state = dict()
    run_env([learn_ability, use_ability], was_success, state=state)


def test_no_strategy_buy(pid=0):
    # our heroes should have all the control of the world
    # to limit the amount of logic buying happens on game start
    # rather than during strategy time
    def was_success(ds, rs, state):
        # we have TPs at the start
        s = get_player(pid, ds, rs)
        items = s.get('items', [])
        return len(items) == 1

    state = dict()
    run_env([], was_success, state=state)


def test_purchase_item(pid=0):
    def purchase_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.PurchaseItem('item_tango')
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)

        # We have TP in inventory at the beginning
        items = s.get('items', [])

        tango = None
        for item in items:
            if item['slot'] == 0:
                tango = item
                break

        return tango is not None and tango['charges'] == 3

    state = dict()
    run_env([purchase_item], was_success, state=state)


def test_use_ability_on_unit():
    # Player 1 is antimage, using counter spell
    def learn_ability(ds, rs, state):
        b = IPCMessageBuilder()
        # 3 is bloodseeker
        p = b.player(3)
        p.LevelAbility(SpellSlot.Q)
        return b.build()

    def use_ability(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(3)

        s = get_player(0, ds, rs)
        p.UseAbilityOnEntity(SpellSlot.Q, s['handle'])
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(0, ds, rs)

        # We have TP in inventory at the beginning
        modifiers = s.get('modifiers', [])
        for mod in modifiers:
            if mod.get('name', '') == 'modifier_bloodseeker_bloodrage':
                return True

        return False

    state = dict()
    # increase timeout to give time to the hero to reach a tree
    run_env([learn_ability, use_ability], was_success, state=state)


def test_use_ability_on_tree(pid=0):
    # Player 1 is antimage, using counter spell
    def purchase_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.PurchaseItem('item_tango')
        return b.build()

    def use_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.UseAbilityOnTree(0, 10)
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)

        # We have TP in inventory at the beginning
        items = s.get('items', [])

        tango = None
        for item in items:
            if item['slot'] == 0:
                tango = item
                break

        return tango is not None and tango['charges'] == 2

    state = dict()
    # increase timeout to give time to the hero to reach a tree
    run_env([purchase_item, use_item], was_success, timeout=10, state=state)


def test_sell_item(pid=0):
    def purchase_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.PurchaseItem('item_tango')
        return b.build()

    def sell_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.SellItem(0)
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)
        items = s.get('items')
        return items is not None and len(items) == 1

    state = dict()
    run_env([purchase_item, sell_item], was_success, state=state)


def test_swap_item(pid=0):
    def purchase_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.PurchaseItem('item_tango')
        return b.build()

    def swap_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.SwapItems(0, 6)
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)
        items = s.get('items')

        tango = None
        for item in items:
            if item['slot'] == 6:
                tango = item
                break

        return tango is not None

    state = dict()
    run_env([purchase_item, swap_item], was_success, state=state)


def test_drop_item(pid=0):
    def purchase_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.PurchaseItem('item_tango')
        return b.build()

    def drop_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.DropItem(0, (-6700.0, -6700.0))
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)
        items = s.get('items')
        return items is not None and len(items) == 1

    state = dict()
    run_env([purchase_item, drop_item], was_success, state=state)


def test_pick_item(pid=0):
    def purchase_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.PurchaseItem('item_tango')
        return b.build()

    def drop_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.DropItem(0, (-6700.0, -6700.0))

        s = get_player(pid, ds, rs)
        items = s.get('items')

        tango = None
        for item in items:
            if item['slot'] == 0:
                tango = item
                break

        state['item'] = tango['handle']
        return b.build()

    def pick_item(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.PickUpItem(state['item'])
        return b.build()

    def was_success(ds, rs, state):
        s = get_player(pid, ds, rs)
        items = s.get('items')
        return items is not None and len(items) == 2

    state = dict()
    run_env([purchase_item, drop_item, pick_item], was_success, state=state)


# Courier action take no arguments and are all alike
# might be able to get away with not testing the other cases
def test_courier_secret(pid=0):
    def go_secret(ds, rs, state):
        b = IPCMessageBuilder()
        p = b.player(pid)
        p.CourierSecret()

        pos = get_courier(pid, ds, rs)['location']
        state['pos'] = (pos['x'], pos['y'])
        return b.build()

    def was_success(ds, rs, state):
        courier = get_courier(pid, ds, rs)
        distance = get_distance(courier, state['pos'])
        return distance > 0

    state = dict()
    run_env([go_secret], was_success, state=state)



#
# def use_ability_unit(ds, rs, state):
#     b = IPCMessageBuilder()
#     p = b.player(pid)
#     p.UseAbilityOnEntity(SpellSlot.Q, unit)
#     return b.build()


if __name__ == '__main__':
    # this does not work
    # test_stop_ability()
    # test_courier_return()
    test_use_ability_on_unit()
