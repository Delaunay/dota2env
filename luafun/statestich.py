import asyncio
import copy

from luafun.game.states import WorldStateDelta

class FactionState:
    def __init__(self):
        # make sure we do not access partial states
        self.lock = asyncio.Lock()
        self.s = 0
        self.e = 0
        self.r = 0

    def copy():
        return copy.deepcopy(self)


async def apply_diff(state, delta: WorldStateDelta):
    async with self.lock:
        state.s += 1
        # check s == e to know if the apply has finished
        print(delta)
        state.e += 1
