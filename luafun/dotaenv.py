"""This module encapsulate the basic Dota2Game API and make it a RL gym environment
that is suitable for machine learning
"""
import asyncio
from itertools import chain
import logging
import traceback

from luafun.game.game import Dota2Game
from luafun.game.modes import DOTA_GameMode
import luafun.game.dota2.state_types as msg
from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE
from luafun.game.action import action_space
import luafun.game.constants as const
import luafun.game.action as actions

from luafun.stitcher import Stitcher
from luafun.reward import Reward
from luafun.draft import DraftTracker

log = logging.getLogger(__name__)


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

    Parameters
    ----------
    path: str
        Path to the game folder `.../dota 2 beta`

    dedicated: bool
        Server only

    stitcher: Stitcher
        Stitch game state together

    reward: Reward
        Used to compute reward after every step

    _config:
        Internal argument used when the HTTP server controls the environment
    """
    def __init__(self, path, dedicated=True, draft=0, stitcher=None, reward=None, _config=None):
        super(Dota2Env, self).__init__(path, dedicated, draft, config=_config)
        # For debugging only
        self.radiant_message = open(self.paths.bot_file('out_radiant.txt'), 'w')
        self.dire_message = open(self.paths.bot_file('out_dire.txt'), 'w')

        self._action_space = action_space()

        # Function to stich state together
        if stitcher is None:
            stitcher = Stitcher()

        self.sticher = stitcher

        # State per faction
        self._radiant_state = self.sticher.initial_state()
        self._dire_state = self.sticher.initial_state()
        # ---

        # Reward function
        if reward is None:
            reward = Reward()

        self.reward = reward

        # Draft tracker for the drafting AI
        self.draft_tracker = DraftTracker()
        self._radiant_state.draft = self.draft_tracker.draft
        self._dire_state.draft = self.draft_tracker.draft

    def cleanup(self):
        self.radiant_message.close()
        self.dire_message.close()

    def dire_state(self):
        return self._dire_state

    def radiant_state(self):
        return self._radiant_state

    # For states we should have a queue of state to observe
    def update_dire_state(self, message: msg.CMsgBotWorldState):
        """Receive a state diff from the game for dire"""
        try:
            self.sticher.apply_diff(self._dire_state, message)
        except Exception as e:
            log.error(f'Error happened during state stiching {e}')
            log.error(traceback.format_exc())

        self.dire_message.write(str(type(message)) + '\n')
        self.dire_message.write(str(message))
        self.dire_message.write('-------\n')

    def update_radiant_state(self, message: msg.CMsgBotWorldState):
        """Receive a state diff from the game for radiant"""
        try:
            self.sticher.apply_diff(self._radiant_state, message)
        except Exception as e:
            log.error(f'Error happened during state stiching {e}')
            log.error(traceback.format_exc())

        self.radiant_message.write(str(type(message)) + '\n')
        self.radiant_message.write(str(message))
        self.radiant_message.write('-------\n')

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

    # Gym Environment API
    # -------------------
    def render(self):
        self.options.dedicated = False

    def reset(self):
        """

        Returns
        -------
        observation: Tuple[FactionState, FactionState]
            state observation (radiant, dire)
        """
        if self.running:
            self.__exit__(None, None, None)
            self._radiant_state = self.sticher.initial_state()
            self._dire_state = self.sticher.initial_state()

        self.__enter__()
        return self._radiant_state, self._dire_state

    def close(self):
        self.__exit__(None, None, None)

    @property
    def action_space(self):
        """This comes from the action encoding work we did

        Notes
        -----

        .. code-block:: python

            {
                'Radiant': {
                    # Player 1
                    '0': {
                        action       = One of 25 base actions
                        vloc         = Location Vector x, y
                        hUnit        = Unit handle
                        abilities    = Ability/Item slot to use (40)
                        tree         = Tree ID
                        runes        = Rune ID
                        items        = Which item to buy
                        ix2          = Inventory Item slot for swapping
                    },
                    # Player 2
                    '1' : {
                        ...
                    },

                    ...

                    # Hero Selection command
                    'HS'            = {
                        select = spaces.Discrete(const.HERO_COUNT),
                        ban    = spaces.Discrete(const.HERO_COUNT)
                    }
                }
                'Dire': [
                    ...
                ]
            }

        """
        return self._action_space

    @property
    def observation_space(self):
        return self.sticher.observation_space

    def step(self, action):
        """Make an action and return the resulting state

        Returns
        -------
        observation: Tuple[FactionState, FactionState]
            state observation (radiant, dire)

        reward: float
            amount of reward that state

        done: bool
            is the game is done

        info: Optional[dict]
            returns nothing
        """
        # 1. send action
        # 1.1 Preprocess the actions (remapping)
        preprocessed = self._action_preprocessor(action)

        # 1.2 Send the action
        self.send_message(preprocessed)

        # 2. Wait for the new stitched state
        rs = self.radiant_state()
        ds = self.dire_state()

        obs = (rs, ds)

        # 3. Compute the reward
        reward = self.reward(*obs)

        done = self.state.get('win', None) is not None

        info = dict()

        return obs, reward, done, info

    # Helpers
    # -------
    def _action_preprocessor(self, message):
        self.draft_tracker.update(
            message[TEAM_RADIANT].get('HS', dict()),
            message[TEAM_DIRE].get('HS', dict())
        )

        players = chain(message[TEAM_RADIANT].items(), message[TEAM_DIRE].item())

        for pid, action in players:
            if pid == 'HS':
                continue

            # slots needs to be remapped from our unified slot
            # to the game internal slot
            hid = self.heroes[pid]['hid']
            slot = action[actions.ARG.nSlot]
            slot = const.HERO_LOOKUP.ability_from_id(hid, slot)
            action[actions.ARG.nSlot] = slot

            # Remap Trees

            # Remap Entity Handles

            # Remap Item Name

            # Remap vloc to be across al map

        return message


def _default_game(path, dedicated=True, config=None):
    game = Dota2Env(path, dedicated=dedicated, _config=config)
    game.options.ticks_per_observation = 4
    game.options.host_timescale = 2
    return game


def mid1v1(path, config=None):
    game = _default_game(path, config=config)
    game.options.game_mode = int(DOTA_GameMode.DOTA_GAMEMODE_1V1MID)
    return game


def allpick_nobans(path, config=None):
    game = _default_game(path, config=config)
    game.options.game_mode = int(DOTA_GameMode.DOTA_GAMEMODE_AP)
    return game


def ranked_allpick(path, config=None):
    game = _default_game(path, config=config)
    game.options.game_mode = int(DOTA_GameMode.DOTA_GAMEMODE_ALL_DRAFT)
    return game


def allrandom(path, config=None):
    game = _default_game(path, config=config)
    game.options.game_mode = int(DOTA_GameMode.DOTA_GAMEMODE_AR)
    return game


_environments = {
    'mid1v1': mid1v1,
    'allpick_nobans': allpick_nobans,
    'ranked_allpick': ranked_allpick,
    'allrandom': allrandom
}


def main(path=None, config=None):
    """This simply runs the environment forever with NO MODELS
    It means bots will not do anything, if drafting is enabled nothing will be drafted

    This function is used for testing purposes
    """
    from argparse import ArgumentParser
    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument('--draft', action='store_true', default=False,
                        help='Enable bot drafting')

    parser.add_argument('--mode', type=str, default='allpick_nobans',
                        help='Game mode')

    parser.add_argument('--path', type=str, default=path,
                        help='Custom Dota2 game location')

    args = parser.parse_args()
    factory = _environments.get(args.mode)
    if factory is None:
        return

    game = factory(args.path, config=config)
    game.options.dedicated = False
    game.draft = int(args.draft)

    with game:
        game.wait()

    print('Done')


if __name__ == '__main__':
    main()
