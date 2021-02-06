from argparse import ArgumentParser
import logging

from luafun.dotaenv import dota2_environment
from luafun.utils.options import option


class InferenceEngine:
    def __init__(self, factory):
        from luafun.game.action import action_space
        self.action_space = action_space()

    def action(self, state):
        return self.action_space.sample()


def main(config=None):
    """This simply runs the environment forever, default to RandomActor (actions are random)
    It means bots will not do anything game winning, if drafting is enabled nothing will be drafted
    """

    parser = ArgumentParser()
    parser.add_argument('--draft', action='store_true', default=False,
                        help='Enable bot drafting')

    parser.add_argument('--mode', type=str, default='allpick_nobans',
                        help='Game mode')

    parser.add_argument('--path', type=str, default=option('dota.path', None),
                        help='Custom Dota2 game location')

    parser.add_argument('--render', action='store_true', default=False,
                        help='Custom Dota2 game location')

    parser.add_argument('--speed', type=float, default=4,
                        help='Speed multiplier')

    parser.add_argument('--interactive', action='store_true', default=False,
                        help='Make a human create the lobby')

    parser.add_argument('--model', type=str, default=None,
                        help='Model name factory, defaults to a random action sampler')

    args = parser.parse_args()
    game = dota2_environment(args.mode, args.path, config=config)

    if game is None:
        return

    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    game.options.dedicated = not args.render
    game.options.interactive = args.interactive
    game.options.host_timescale = args.speed
    game.options.draft = int(args.draft)

    model = InferenceEngine(args.model)

    with game:
        state = game.initial()

        # Draft here if enabled
        while game.running:

            if game.options.draft:
                pass

            break

        game.wait_end_draft()

        for pid in game.bot_ids:
            print(f'Player {pid} is a bot')

        # Play the game
        while game.running:
            # Start issuing orders here
            action = model.action(state)

            # take a random action
            state, reward, done, info = game.step(action)

            if game.cnt > 0 and game.cnt % 100 == 0:
                print(f'Step time {game.avg / game.cnt:.4f}')

        print('Game Finished')

    print('Done')


if __name__ == '__main__':
    main()
