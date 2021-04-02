from luafun.dotaenv import Dota2Env
from luafun.game.ipc_send import new_ipc_message


class InferenceEngine:
    """InferenceEngine is used to produce more observations while the model is undergoing training
    The newest model is pulled from time to time to make sure it keeps improving
    """

    def __init__(self, model, train):
        self.bots = None
        self.model = None
        self.state_space = None
        self.sampler = None
        self.filter = None
        self.action_space = None
        self.passive = model == 'passive'
        self.random = model == 'random'
        self.trainer = train

    def init_draft(self):
        pass

    def close_draft(self):
        pass

    def init_play(self, game: Dota2Env):
        """Initialize using environment config"""
        self.bots = game.bot_ids
        self.state_space = game.observation_space
        self.action_space = game.action_space

        # self.model = HeroModel(len(self.bots), input_size, 16)
        # self.filter = ActionFilter()
        # self.sampler = ActionSampler()
        # self.filter = lambda *args: lambda x: x

    def load_model(self):
        pass

    def action(self, state):
        """Build the observation batch and the action to take"""
        # batch = generate_game_batch(state, self.bots)

        if self.passive:
            return None

        if self.random:
            return self.action_space.sample()

        # reload model
        if self.trainer.new_inference_model:
            self.load_model()

        # msg = self.model(state)
        # filter = self.filter(state, unit, rune, tree)
        # action = self.sampler.sampled(msg, filter)
        return None


class TrainEngine:
    def __init__(self, train):
        pass

    @property
    def new_inference_model(self):
        return False

    def push(self, *args):
        pass
