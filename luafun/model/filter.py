import torch

from luafun.game.action import Action, AbilitySlot
import luafun.game.constants as const


class ActionFilter:
    """To speedup learning we compute every ability that are truly available instead
    of making the network learn it implicitly

    The filter could be applied before the softmax layer so the action probabilities would be split
    on the possible actions.

    Notes
    -----
    OpenAI approach was to make the network select action from the list of filtered abilities.

    Our approach is totally different as the network still outputs for all the actions
    but can only sample from possible action, i.e instead of wasting timesteps trying actions doomed to fail
    we force the network to do the next best thing and learn from that right away instead of spending more time
    learning when the action is possible or not.

    We expect this to speedup training a lot since it will help the model learn the underlying conditional
    distribution of actions more explicitly.

    Additionally we like this approach better because this layer is truely a training wheel that could be removed
    in the future with minimal impact on our network; whereas OpenAI network was built around the filtering.
    """
    def __init__(self):
        self.action = torch.zeros((len(Action),))
        # Action that are mostly always possible
        self.action[Action.Stop] = 1
        self.action[Action.MoveDirectly] = 1
        self.action[Action.MoveToLocation] = 1
        self.action[Action.CourierSecret] = 1
        self.action[Action.CourierReturn] = 1
        self.action[Action.CourierTakeStash] = 1
        self.action[Action.CourierTransfer] = 1
        self.action[Action.SwapItems] = 1
        self.action[Action.DropItem] = 1

        # swappable item = only item in inventory
        self.swap = torch.zeros((len(const.ItemSlot),))

        # check cooldown, mana, level for learning
        self.abilities = torch.zeros((len(AbilitySlot),))
        for i in range(AbilitySlot.Item0, AbilitySlot.R + 1):
            self.abilities[i] = 1

        # check gold & store proximity (do not forget courier)
        self.items = torch.zeros((len(const.ITEM_COUNT),))

    def get_filter(self, state, unit, rune, tree):
        pass
