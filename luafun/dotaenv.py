import asyncio
import logging

from luafun.game.game import Dota2Game
from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE
from luafun.statestich import FactionState, apply_diff


log = logging.getLogger(__name__)


def team_name(faction):
    if faction == TEAM_RADIANT:
        return 'Radiant'
    return 'Dire'


async def _acquire_faction(state):
    # wait for latest diff to be applied
    async while state.lock:
        state.r += 1
        return state.copy()

def acquire_state(state):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_acquire_faction(state))


class Dota2Env(Dota2Game):
    """Currently the state is simply built as we go, but we should probably do something
    with a bit more guarantees, in particular the state should be for a particular frame
    so the ML can work on a consistent dataset

    Although leaving it inconsistent could be an interesting experiment
    """
    def __init__(self, path):
        super.__init__(self, path)
        self.radiant_state = FactionState()
        self.dire_state = FactionState()

    def dire_state(self, messsage: WorldStateDelta):
        """Receive a state diff from the game for dire"""
        apply_diff(self.dire_state, message)

    def radiant_state(self, message: WorldStateDelta):
        """Receive a state diff from the game for radiant"""
        apply_diff(self.radiant_state, message)

    def receive_message(self, faction: int, player_id: int, message: dict):
        """We only use log to get errors back if any"""
        error = message.get('E')
        if error is not None:
            log.error(f'recv {team_name(faction)} {player_id} {error}')

    # Training data
    def generate_bot_state(self):
        """Generate the states of our bots. The state is the faction state augmented with
        player specific information
        
        In a standard self-play Game this would return 10 states
        """
        dire = acquire_state(self.dire_state)
        radiant = acquire_state(self.radiant_state)
        pass

    # Action
    def send_bot_actions(self, action):
        pass


def main():
    pass


if __name__ == '__main__':
    pass
