from argparse import ArgumentParser
import logging

from luafun.draft import DraftTracker
from luafun.model.actor_critic import SimpleDrafter


class StandAloneDrafter:
    def __init__(self, team):
        self.tracker = DraftTracker()
        self.model = SimpleDrafter()
        self.team = team

    def suggestions(self):
        x = self.tracker.as_tensor(self.team)
        y = self.model(x)

        return self.decode_suggestion(y)

    def decode_suggestion(self, y):
        pass


# Add a webserver here to add bans/picks & display suggestions
#   * Use dota2 API to get draft of a given username in realtime
# Add training facility
#   * parse Dota2 draft using the dota2 API
#   * train on a given dataset (herald, crusader, archon, legend, divine, Immortal, tournament)
def main():
    """Utility to train & use the drafter alone"""
    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser()


