from argparse import ArgumentParser
from dataclasses import dataclass
import logging

import torch

from luafun.draft import DraftTracker
from luafun.game.ipc_send import TEAM_RADIANT
from luafun.model.drafter import SimpleDrafter
import luafun.game.constants as const


@dataclass
class HeroSuggestion:
    name: str
    prob: float
    id: int


class StandAloneDrafter:
    """Returns the pick/ban suggestion for a single game

    Example
    -------
    >>> _ = torch.manual_seed(0)
    >>> drafter = StandAloneDrafter(TEAM_RADIANT, top=5)
    >>> picks, bans = drafter.suggestions()

    >>> for p in picks:
    ...     print(f'{p.name:>20} {p.prob}')
            Shadow Fiend 0.009010703302919865
            Storm Spirit 0.008973498828709126
           Shadow Shaman 0.008970417082309723
            Earth Spirit 0.00895723607391119
                 Warlock 0.008921736851334572

    >>> for b in bans:
    ...     print(f'{b.name:>20} {b.prob}')
            Witch Doctor 0.00909948069602251
                  Huskar 0.009042616002261639
                 Invoker 0.009035504423081875
                  Mirana 0.009014634415507317
     Keeper of the Light 0.008976804092526436

    """
    def __init__(self, team, top=20):
        self.tracker = DraftTracker()
        self.model = SimpleDrafter()
        self.team = team
        self.probs = None
        self.suggestion_count = top

    def suggestions(self):
        x = self.tracker.as_tensor(self.team)
        x = x.unsqueeze(0)
        picks, bans = self.model(x)
        picks = self.decode_suggestion(picks[0])[:self.suggestion_count]
        bans = self.decode_suggestion(bans[0])[:self.suggestion_count]
        return picks, bans

    def decode_suggestion(self, probs):
        probs = probs.detach()
        probs = probs.tolist()
        suggestions = []

        for id, p in enumerate(probs):
            hero = const.HERO_LOOKUP.from_offset(id)

            suggestions.append(HeroSuggestion(
                hero.get('pretty_name', hero.get('name')),
                p,
                id
            ))

        suggestions.sort(key=lambda x: x.prob, reverse=True)
        return suggestions


# Add a webserver here to add bans/picks & display suggestions
#   * Use dota2 API to get draft of a given username in realtime
# Add training facility
#   * parse Dota2 draft using the dota2 API
#   * train on a given dataset (herald, crusader, archon, legend, divine, Immortal, tournament)
def main():
    """Utility to train & use the drafter alone"""
    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument('--steam-id', type=str, help='Your steam id to retrieve your match')


