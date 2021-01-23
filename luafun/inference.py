



class InferenceEngine:
    """InferenceEngine is used to produce more observations while the model is undergoing training
    The newest model is pulled from time to time to make sure it keeps improving
    """

    def __init__(self, storage, model):
        self.storage = storage
        # This is the old policy
        self.model = models

    def infer(self, obs):
        """Execute the current model"""
        out = self.model(obs)
        return self.to_actions(out)

    def to_actions(self, x):
        """Transform a NNet output into a understandable set of commands by the Lua bots"""
        return x