
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

    >>> max_i = 0
    >>> min_i = 0
    >>> min_j = 0
    >>> max_j = 0
    >>> for i in range(-100, 100):
    ...     for j in range(-100, 100):
    ...         tree = proxymapper.entities(-6016 + i, -6784 + j)[2]
    ...         if tree:
    ...             max_i = max(max_i, i)
    ...             min_i = min(min_i, i)
    ...             max_j = max(max_j, j)
    ...             min_j = min(min_j, j)
    >>> (min_i, max_i), max_i - min_i
    ((-100, 16), 116)
    >>> (min_j, max_j), max_j - min_j
    ((-15, 99), 114)

    """
    def __init__(self):
        self.manager = EntityManager()

        # Trees are more and more dynamic
        # so we need to track them as well
        self.tree = EntityManager()
        self.tid_pos = dict()
        for _, tree in const.TREES.items():
            x, y, _ = tree['loc']
            self.tree.add_entity(tree['id'], x, y)
            self.tid_pos[tree['id']] = x, y
            # print(x, y)

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
        rune = const.get_rune(x, y)
        unit = self.manager.get_entity(x, y)
        return unit, rune, tree
