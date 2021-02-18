from collections import defaultdict
import json
import logging

from luafun.utils.options import option


log = logging.getLogger(__name__)


EXTRACT_ENABLED = option('extract', False, bool)
REPLAY_ENABLED = option('replay', True, bool)


class SaveReplay:
    """Save Proto messages we receive to replay the match for debugging purposes"""
    def __init__(self, filename):
        self.store = open(filename, 'w')

    def save(self, radiant, dire):
        if not REPLAY_ENABLED:
            return

        radiant.pop('perf', None)
        dire.pop('perf', None)

        self.store.write(f'RAD, {json.dumps(radiant)}\nDIRE,  {json.dumps(dire)}\n')

    def close(self):
        self.store.close()


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
        if not EXTRACT_ENABLED:
            return

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

