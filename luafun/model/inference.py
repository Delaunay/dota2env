from luafun.dotaenv import Dota2Env
from luafun.utils.ring import RingBuffer
from luafun.model.training import TrainEngine
from luafun.model.filter import ActionFilter
from luafun.model.actor_critic import ActionSampler
from luafun.game.ipc_send import new_ipc_message

from luafun.game.action import ARG


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
        self.action_sampler = ActionSampler()
        self.passive = model == 'passive'
        self.random = model == 'random'
        self.trainer = train
        self.obs = []
        # self.filter = ActionFilter()

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
        #  lambda *args: lambda x: x

    def load_model(self, weights):
        pass

    def make_batch(self, state):
        self.obs.append(state)

        if len(self.obs) == 16:
            self.obs = self.obs[-16:]

        return torch.stack(self.obs, dim=1)

    def make_ipc_message(self, action):
        msg = new_ipc_message()

        for i, pid in enumerate(self.bots):
            f = 2
            if pid > 4:
                f = 3

            msg[f][pid] = {
                ARG.action: action[ARG.action][i].item(),
                ARG.vLoc: action[ARG.vLoc][i].tolist(),
                ARG.sItem: action[ARG.sItem][i].item(),
                ARG.nSlot: action[ARG.nSlot][i].item(),
                ARG.ix2: action[ARG.ix2][i].item(),
            }

        return msg

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
            states = self.make_batch(state)

            action_probs = self.trainer.engine.actor_critic.infer(states)

            # filter, action, log_probs = self.filter(action_probs)
            filter = lambda x: x

            action, log_probs, entropy = self.action_sampler.sampled(action_probs, filter)

            return self.make_ipc_message(action), log_probs, filter

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
