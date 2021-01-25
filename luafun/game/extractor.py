
from collections import defaultdict
import json
import logging

log = logging.getLogger(__name__)


# Recognized DataType
# BOUNDS
# TREE
# RUNE
# SHOP


class Extractor:
    """Use Lua to extract semi-constant from the game (runes, trees, etc..)"""
    def __init__(self):
        self.cache = defaultdict(dict)
        self.store = open('exported.json', 'w')

    def save(self, message):
        dtype = message.get('T')
        if dtype is None:
            return

        data = message.get('I')

        # uid = None
        # if isinstance(data, list):
        #     uid = data[0]

        # if uid is not None and uid in self.cache[dtype]:
        #     return

        # self.cache[dtype][uid] = True

        data.append(dtype)
        self.store.write(json.dumps(data))
        self.store.write('\n')
        log.debug(message)

    def close(self):
        self.store.close()

