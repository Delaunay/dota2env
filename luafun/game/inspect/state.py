import base64
import json
import os
import logging

from luafun.game.inspect.base import BasePage
from luafun.utils.python_fix import asdict
import luafun.game.constants as const

import rpcjs.elements as html


log = logging.getLogger(__name__)


class GameInspector(BasePage):
    def routes(self):
        return [
            '/state/<string:faction>/<int:player>',
            '/state/<string:faction>'
        ]

    def __init__(self, app):
        super(GameInspector, self).__init__(app)
        self.title = 'State'
        self.latest = 0

    def main(self, faction, player=None):
        radiant = faction.lower() in ('rad', 'radiant')
        page = 'state.html'

        if radiant:
            self.title = 'Radiant State'
            state = self.getattr('radiant_state')
        else:
            self.title = 'Dire State'
            state = self.getattr('dire_state')

        if state is not None:
            if state._s <= self.latest:
                log.debug('State is not being updated anymore')

            self.latest = state._s

        if player is not None and state is not None:
            pid = player
            page = 'player.html'
            state = state._players.get(int(pid), dict())

        raw = ''
        if state and not isinstance(state, (str, dict)):
            state = asdict(state)

        if state is not None:
            if '_manager' in state:
                state['_manager'] = None
            raw = json.dumps(state, indent=2)

        page = self.env.get_template(page)
        return page.render(code=html.pre(raw), state=self.state, player=player)


class DrawMap(BasePage):
    def routes(self):
        return [
            '/draw/<string:faction>/<int:player>',
            '/draw/<string:faction>'
        ]

    def __init__(self, app):
        super(DrawMap, self).__init__(app)
        self.title = 'Draw'

        dir = os.path.dirname(os.path.abspath(__file__))
        resource = os.path.join(dir, '..', 'resources')
        self.resource_folder = os.path.join(resource, 'minimap')

        minimap = os.path.join(self.resource_folder, 'minimap_7.23.png')
        minimap = os.path.realpath(minimap)

        self.background = self.load_image(minimap)
        self.colors = {
            0: '#3074F9', # Blue
            1: '#66FFC0', # Aquamarine
            2: '#BD00B7', # Purple
            3: '#F8F50A', # Yellow
            4: '#FF6901', # Orange
            5: '#FF88C5', # Pink
            6: '#A2B349', # Olive
            7: '#63DAFA', # Sky Blue
            8: '#01831F', # Green
            9: '#9F6B00', # Brown
        }
        self.color_building = {
            2: '#006400',
            3: '#B20000'
        }

        self.color_unit = {
            2: '#00FF00',
            3: '#FF0000'
        }

        self.trees = {
            'ignored': '#6bff33',
            'duplicated': '#fff633',
        }

    def load_image(self, name):
        """Load an image"""
        with open(name, 'br') as f:
            raw = f.read()

        data = base64.b64encode(raw).decode('utf-8')
        return f'data:image/png;base64,{data}'

    def show_units(self, item, type):
        units = []
        # Heroes
        for k, v in item.items():
            loc = v.get('location')

            if loc is None:
                continue

            x = (loc['x'] + const.BOUNDS[1][0]) * 1024 / const.SIZE[0]
            y = (const.BOUNDS[1][1] - loc['y']) * 1024 / const.SIZE[0]

            c = None
            if type == 'hero':
                c = self.colors.get(k)

            if type == 'building':
                c = self.color_building.get(v.get('team_id'))

            if type == 'unit':
                c = self.color_unit.get(v.get('team_id'))

            if c is None:
                c = 'white'

            # can use `url(#default_icon)` to fill using an image
            h = f'<circle id="{k}" cx="{x}" cy="{y}" stroke="black" r="8" fill="{c}"/>'
            units.append(h)
        return units

    def show_bad_trees(self):
        bad_trees = []

        for k, loc in const.IGNORED_TREES.items():
            x = (loc[0] + const.BOUNDS[1][0]) * 1024 / const.SIZE[0]
            y = (const.BOUNDS[1][1] - loc[1]) * 1024 / const.SIZE[0]
            c = self.trees['ignored']
            h = f'<circle id="{k}" cx="{x}" cy="{y}" stroke="black" r="8" fill="{c}"/>'
            bad_trees.append(h)

        for k, loc in const.DUP_TREES.items():
            x = (loc[0] + const.BOUNDS[1][0]) * 1024 / const.SIZE[0]
            y = (const.BOUNDS[1][1] - loc[1]) * 1024 / const.SIZE[0]
            c = self.trees['duplicated']
            h = f'<circle id="{k}" cx="{x}" cy="{y}" stroke="black" r="8" fill="{c}"/>'
            bad_trees.append(h)

        return bad_trees

    def main(self, faction, player=None):
        radiant = faction.lower() in ('rad', 'radiant')

        if radiant:
            self.title = 'Radiant Map'
            state = self.getattr('radiant_state')
        else:
            self.title = 'Dire Map'
            state = self.getattr('dire_state')

        default_icon = self.load_image(os.path.join(self.resource_folder, '30_005.png'))

        heroes = []
        buildings = []
        units = []
        trees = self.show_bad_trees()

        if state:
            heroes = self.show_units(state._players, 'hero')
            buildings = self.show_units(state._buildings, 'building')
            units = self.show_units(state._units, 'unit')

        units = '\n'.join(heroes + buildings + units + trees)

        svg = f"""
        <svg height="1024px" width="1024px" style="background-image: url({self.background})">
            <defs>
                <pattern id="default_icon" x="16" y="16" patternUnits="userSpaceOnUse" height="32" width="32">
                    <image x="0" y="0" xlink:href="{default_icon}"></image>
                </pattern>
            </defs>

            {units}
        <svg>
        """

        page = self.env.get_template('map.html')
        # image = f'<img src="data:image/png;base64,{self.background}">'
        return page.render(image=svg, state=self.state)
