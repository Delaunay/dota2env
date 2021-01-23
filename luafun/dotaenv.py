"""This module encapsulate the basic Dota2Game API and make it a RL gym environment
that is suitable for machine learning
"""
import asyncio
import logging
import os
import traceback

from luafun.game.game import Dota2Game
import luafun.game.dota2.state_types as msg
import luafun.game.dota2.shared as enums

from luafun.openai import models, stichers, states

log = logging.getLogger(__name__)


registered_states = states
registered_models = models
registered_stich = stichers


def team_name(faction):
    if faction == TEAM_RADIANT:
        return 'Radiant'
    return 'Dire'


async def _acquire_faction(state):
    # wait for latest diff to be applied
    async with state._lock:
        state._r += 1
        return state.copy()

def acquire_state(state):
    return asyncio.run(_acquire_faction(state))


class Dota2Env(Dota2Game):
    """Currently the state is simply built as we go, but we should probably do something
    with a bit more guarantees, in particular the state should be for a particular frame
    so the ML can work on a consistent dataset

    Although leaving it inconsistent could be an interesting experiment
    """
    def __init__(self, path, dedicated=True):
        super(Dota2Env, self).__init__(path, dedicated)
        # For debugging only
        self.radiant_message = open(self.paths.bot_file('out_radiant.txt'), 'w')
        self.dire_message = open(self.paths.bot_file('out_dire.txt'), 'w')

        # Get factories necessary for our model
        self.setup_name = 'openai-mid'

        # Function to stich state together
        self.sticher = stichers[self.setup_name]

        # State per faction
        self._radiant_state = states[self.setup_name]()
        self._dire_state = states[self.setup_name]()

    def cleanup(self):
        self.radiant_message.close()
        self.dire_message.close()

    # Random Access state getter
    # waits for the stiching to be finished and fetch the observation
    @property
    async def dire_state_async(self):
        return await _acquire_faction(self._dire_state)

    @property
    async def radiant_state_async(self):
        return await _acquire_faction(self._radiant_state)

    @property
    def dire_state(self):
        return acquire_state(self._dire_state)

    @property
    def radiant_state(self):
        return acquire_state(self._radiant_state)

    # For states we should have a queue of state to observe
    async def update_dire_state(self, message: msg.CMsgBotWorldState):
        """Receive a state diff from the game for dire"""
        try:
            await self.sticher(self._dire_state, message)
        except Exception as e:
            log.error(f'Error happened during state stiching {e}')
            log.error(traceback.format_exc())

        self.dire_message.write(str(type(message)) + '\n')
        self.dire_message.write(str(message))
        self.dire_message.write('-------\n')

    async def update_radiant_state(self, message: msg.CMsgBotWorldState):
        """Receive a state diff from the game for radiant"""
        try:
            await self.sticher(self._radiant_state, message)
        except Exception as e:
            log.error(f'Error happened during state stiching {e}')
            log.error(traceback.format_exc())

        self.radiant_message.write(str(message))    

    def receive_message(self, faction: int, player_id: int, message: dict):
        """We only use log to get errors back if any"""
        pass
    
    # Training data
    def generate_bot_state(self):
        """Generate the states of our bots. The state is the faction state augmented with
        player specific information
        
        In a standard self-play Game this would return 10 states
        """
        dire = acquire_state(self._dire_state)
        radiant = acquire_state(self._radiant_state)
        pass

    # Action
    def send_bot_actions(self, action):
        pass


def main():
    from luafun.game.modes import DOTA_GameMode
    from luafun.game.ipc_send import new_ipc_message
    logging.basicConfig(level=logging.DEBUG)

    game = Dota2Env('F:/SteamLibrary/steamapps/common/dota 2 beta/', False)
    game.options.game_mode = int(DOTA_GameMode.DOTA_GAMEMODE_1V1MID)

    with game:
        game.send_message(new_ipc_message())
    
        game.wait()

    print('Done')


if __name__ == '__main__':
    main()
