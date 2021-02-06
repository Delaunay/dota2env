
from luafun.model.filter import ActionFilter
from luafun.model.actor_critic import ActionSampler, HeroModel
from luafun.proximity import ProximityMapper
from luafun.game.action import ARG
from luafun.observations import StateBuilder


class InferenceEngine:
    """InferenceEngine is used to produce more observations while the model is undergoing training
    The newest model is pulled from time to time to make sure it keeps improving
    """

    def __init__(self, nbots, initial_state):
        s = StateBuilder()
        input_size = s.total_size

        self.filter = ActionFilter()
        self.sampler = ActionSampler()

        self.model = HeroModel(nbots, input_size, 16)

        self.filter = lambda *args: lambda x: x

    def __call__(self, state):
        msg = self.model(state)

        filter = self.filter(state, unit, rune, tree)

        action = self.sampler.sampled(msg, filter)

        return action
