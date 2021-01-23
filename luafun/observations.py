from dataclasses import dataclass

from luafun.utils.option import option


class ObservationSampler:
    def __init__(self, timestep):
        pass


@dataclass
class Entry:
    action: float
    state: None
    logprob: float
    reward: float
    done: bool
    newstate: None


class GameDataset:
    """Store all the states of a given game and allow states to be sampled for a given timestep"""
    def __init__(self, timestep=option('timestep', 16)):
        self._actions = []
        self._states = []
        self._logprobs = []
        self._rewards = []
        self._dones = []
        self._newstate = []
        self.timestep = timestep

    def rewards(self, item, timestep, device, gamma):
        rewards = []
        discounted_reward = 0

        # TODO: we should discount by over all time
        for reward, done in zip(reversed(self._rewards), reversed(self._dones)):
            if done:
                discounted_reward = 0

            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return rewards
        
    def __len__(self):
        return len(self._states) - self.timestep

    def states(self, item, timestep, device):
        return self._states[item - self.timestep:item]

    def actions(self, item, timestep, device):
        return self._actions[item - self.timestep:item]

    def logprobs(self, item, timestep, device):
        return self._logprobs[item - self.timestep:item]

    def __getitem__(self, item) -> Entry:
        return Entry(
            self._actions[item],
            self._states[item],
            self._logprobs[item],
            self._rewards[item],
            self._dones[item],
            self._newstate[item])

    def append(self, action, state, logprob, reward, newstate, done):
        self._actions.append(action)
        self._states.append(state)
        self._logprobs.append(logprob)
        self._rewards.append(reward)
        self._dones.append(done)
        self._newstate.append(newstate)
