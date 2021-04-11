from luafun.dotaenv import Dota2Env
from luafun.utils.ring import RingBuffer
from luafun.model.training import TrainEngine
from luafun.game.ipc_send import new_ipc_message


import torch
import torch.distributions as distributions


class InferenceEngine:
    """InferenceEngine is used to produce more observations while the model is undergoing training
    The newest model is pulled from time to time to make sure it keeps improving
    """

    def __init__(self, model, train: TrainEngine):
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

    def load_model(self, weights):
        pass

    def action(self, uid, state) -> (torch.Tensor, torch.Tensor):
        """Build the observation batch and the action to take"""
        # batch = generate_game_batch(state, self.bots)

        if self.passive:
            return None, None, None

        if self.random or state is None:
            return self.action_space.sample(), None, None

        # reload model
        new_weights = self.trainer.weights
        if new_weights is not None:
            self.load_model(new_weights)

        # Local model
        if self.trainer:
            probs = self.trainer.engine.actor_critic.infer(state)
            # Filter actions here
            filter = None
            #
            dist = distributions.Categorical(probs)
            action = dist.sample()

            return action, dist.log_prob(action), filter

        # msg = self.model(state)
        # filter = self.filter(state, unit, rune, tree)
        # action = self.sampler.sampled(msg, filter)
        return None


class LocalInference(InferenceEngine):
    def __init__(self):
        self.time_steps = 16
        self.states = RingBuffer(self.time_steps, None)

    def load_model(self, weights):
        pass

    def init_play(self, game: Dota2Env):
        pass

    def init_draft(self):
        pass

    def close_draft(self):
        pass

    def action(self, uid, state):
        self.states.append(state)
        pass
