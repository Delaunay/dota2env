import asyncio
import logging

from luafun.game.game import Dota2Game
from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE
from luafun.statestich import FactionState, apply_diff
import luafun.game.dota2.state_types as msg


log = logging.getLogger(__name__)


def team_name(faction):
    if faction == TEAM_RADIANT:
        return 'Radiant'
    return 'Dire'


async def _acquire_faction(state):
    # wait for latest diff to be applied
    async with state.lock:
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
    def __init__(self, path, dedicated=True):
        super(Dota2Env, self).__init__(path, dedicated)
        self.radiant_state = FactionState()
        self.dire_state = FactionState()
        self.players = {
            TEAM_RADIANT: 0,
            TEAM_DIRE: 0
        }
        self.radiant_message = open(self.paths.bot_file('out_radiant.txt'), 'w')
        self.dire_message = open(self.paths.bot_file('out_dire.txt'), 'w')

    def cleanup(self):
        self.radiant_message.close()
        self.dire_message.close()

    def is_game_ready(self):
        return self.players[TEAM_RADIANT] + self.players[TEAM_DIRE] == 10

    async def update_dire_state(self, message: msg.CMsgBotWorldState):
        """Receive a state diff from the game for dire"""
        await apply_diff(self.dire_state, message)
        self.dire_message.write(str(message))

    async def update_radiant_state(self, message: msg.CMsgBotWorldState):
        """Receive a state diff from the game for radiant"""
        await apply_diff(self.radiant_state, message)
        self.radiant_message.write(str(message))    

    def receive_message(self, faction: int, player_id: int, message: dict):
        """We only use log to get errors back if any"""
        # error processing
        error = message.get('E')
        if error is not None:
            log.error(f'recv {team_name(faction)} {player_id} {error}')
            return

        # init message
        info = message.get('P')
        if info is not None:
            self.players[int(faction)] += 1
            if self.is_game_ready():
                log.info('All bots accounted for, Game is ready')
            return

        ack = message.get('A')
        if ack is not None:
            log.info(f'(Team: {faction}, Player {player_id}) Message received')
    
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
    from luafun.game.ipc_send import new_ipc_message
    logging.basicConfig(level=logging.DEBUG)

    game = Dota2Env('F:/SteamLibrary/steamapps/common/dota 2 beta/', False)

    with game:
        game.send_message(new_ipc_message())
    
        game.wait()

    print('Done')


if __name__ == '__main__':
    main()
