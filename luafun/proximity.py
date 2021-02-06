
import luafun.game.constants as const
from luafun.entity import EntityManager


class ProximityMapper:
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
