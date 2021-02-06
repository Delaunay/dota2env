
import luafun.game.constants as const
from luafun.entity import EntityManager


class ProximityMapper:
    """Find an entity close to a given location (few pixel away)

    Notes
    -----
    The standard attack range for melee units is 150.
    and hero collision has a size of 24

    Examples
    --------
    >>> proxymapper = ProximityMapper()

    Exact match is found

    >>> proxymapper.entities(-6016, -6784)
    (None, None, 10)

    Units have the smallest margin of error

    >>> proxymapper.add_unit(1233, -6016, -6784)
    >>> (min_x, max_x), (min_y, max_y)= proxymapper.get_range(-6016, -6784, kind='unit')
    >>> (min_x, max_x), max_x - min_x
    ((-3, 3), 6)
    >>> (min_y, max_y), max_y - min_y
    ((-5, 1), 6)

    Tree have a larger margin of error

    >>> (min_x, max_x), (min_y, max_y)= proxymapper.get_range(-6016, -6784, kind='tree')
    >>> (min_x, max_x), max_x - min_x
    ((-14, 22), 36)
    >>> (min_y, max_y), max_y - min_y
    ((-23, 13), 36)

    Runes have the largest margin of error because they rarely can be close together.
    And even if they were you should probably get all of them because they are probably bounties

    >>> proxymapper.add_rune(1233, -6016, -6784)
    >>> (min_x, max_x), (min_y, max_y)= proxymapper.get_range(-6016, -6784, kind='rune')
    >>> (min_x, max_x), max_x - min_x
    ((-35, 53), 88)
    >>> (min_y, max_y), max_y - min_y
    ((-68, 20), 88)
    """

    def __init__(self):
        # Range Creep have a collision box of 8
        self.manager = EntityManager(7)

        # Trees are more and more dynamic
        # so we need to track them as well
        # Trees have a collision box of 128 units
        self.tree = EntityManager(37)
        self.tid_pos = dict()
        for tid, x, y, z in const.TREES:
            if not self.tree.add_entity(tid, x, y):
                print('duplicated tree')

            self.tid_pos[tid] = x, y

        # Runes are not that static either with
        # neutral items spawning them on the fly
        self.runes = EntityManager(89)
        for rid, x, y, z in const.RUNES:
            if not self.runes.add_entity(rid, x, y):
                print('duplicated rune')

    def add_unit(self, handle, x, y):
        self.manager.add_entity(handle, x, y)

    def add_rune(self, handle, x, y):
        self.runes.add_entity(handle, x, y)

    def update(self, delta):
        self._update_trees(delta)
        self._update_units(delta)

    def _update_units(self, delta):
        for event in delta.get('units', []):
            handle = event['handle']
            x, y, z = event['location']
            self.manager.update_position(handle, x, y)

    def _update_trees(self, delta):
        for event in delta.get('tree_events', []):
            if event.get('destroyed', False):
                self.tree.pop_entity(event['tree_id'])
                continue

            if event.get('respawned', False):
                x, y = self.tid_pos[event['tree_id']]
                self.tree.add_entity(event['tree_id'], x, y)

    def entities(self, x, y):
        tree = self.tree.get_entity(x, y)
        rune = self.runes.get_entity(x, y)
        unit = self.manager.get_entity(x, y)
        return unit, rune, tree

    def get_range(self, x, y, kind='tree', size=100):
        """Returns the accepted range for a given position"""
        max_i = float('-inf')
        min_i = float('+inf')
        min_j = float('+inf')
        max_j = float('-inf')

        if kind == 'unit':
            o = 0
        if kind == 'rune':
            o = 1
        if kind == 'tree':
            o = 2

        target = self.entities(x, y)[o]

        for i in range(-size, size):
            for j in range(-size, size):

                tree = self.entities(x + i, y + j)[o]

                if tree == target:
                    max_i = max(max_i, i)
                    min_i = min(min_i, i)
                    max_j = max(max_j, j)
                    min_j = min(min_j, j)

        return (min_i, max_i), (min_j, max_j)
