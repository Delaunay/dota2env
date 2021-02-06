import logging

from luafun.game.constants import position_to_key

log = logging.getLogger(__name__)


class EntityManager:
    """Manages all the entity in the game"""
    def __init__(self, div):
        self.handle_to_key = {}
        self.key_to_handle = {}
        self.div = div

    def dump(self):
        for k, v in self.key_to_handle.items():
            print(k, v)

    def get_range(self, x, y):
        rx = int(x) % self.div
        ry = int(y) % self.div

        return (x - rx) + self.div / 2, (y - ry) + self.div / 2

    def get_entity(self, x, y):
        """Get the entity at that position

        Returns
        -------
        the stored entity or None

        Examples
        --------
        >>> em = EntityManager(10)
        >>> em.add_entity('123', 100, 100)
        True
        >>> em.get_entity(100, 100)
        '123'
        >>> em.get_entity(101, 101)
        '123'
        >>> em.get_entity(120, 120)

        """
        k = position_to_key(x, y, div=self.div)
        return self.key_to_handle.get(k)

    def update_position(self, handle, x, y):
        """Update the position of a given entity

        Returns
        -------
        False if the entity was not found and was inserted

        Examples
        --------
        >>> em = EntityManager(10)
        >>> em.add_entity('123', 10, 10)
        True
        >>> em.update_position('123', 12, 12)
        True
        >>> em.update_position('124', 12, 12)
        False
        """
        return self._update_position(handle, position_to_key(x, y, div=self.div))

    def _update_position(self, handle, key):
        good = False
        oldkey = self.handle_to_key.get(handle)
        self.handle_to_key[handle] = key

        if oldkey:
            self.key_to_handle.pop(oldkey, None)
            good = True

        self.key_to_handle[key] = handle
        return good

    def add_entity(self, handle, x, y):
        """Add a new entity at a given position.

        Returns
        -------
        False if a key was overridden and True otherwise

        Examples
        --------
        >>> em = EntityManager(10)
        >>> em.add_entity('123', 10, 10)
        True
        >>> em.add_entity('123', 10, 10)
        False
        """
        dup = False
        key = position_to_key(x, y, div=self.div)

        if key in self.key_to_handle:
            log.debug('Duplicate key for entities')
            dup = True

        self._update_position(handle, key)
        return not dup

    def pop_entity(self, handle):
        """Remove a given entity from the manager

        Returns
        -------
        False if the handle was not found, True otherwise

        Examples
        --------
        >>> em = EntityManager(10)
        >>> em.add_entity('123', 10, 10)
        True
        >>> em.pop_entity('123')
        True
        >>> em.pop_entity('123')
        False
        """
        oldkey = self.handle_to_key.pop(handle, None)
        if oldkey:
            self.key_to_handle.pop(oldkey, None)
            return True

        return False

    def __len__(self):
        return len(self.handle_to_key)
