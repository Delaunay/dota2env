from argparse import ArgumentParser
import logging

from luafun.dotaenv import dota2_environment
from luafun.utils.options import option
from luafun.model.inference import InferenceEngine


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

    parser.add_argument('--model', type=str, default='random',
                        help='Model name factory, defaults to a random action sampler')

    args = parser.parse_args()
    game = dota2_environment(args.mode, args.path, config=config)

    if game is None:
        return

    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    # Game options
    game.options.dedicated = not args.render
    game.options.interactive = args.interactive
    game.options.host_timescale = args.speed
    game.options.draft = int(args.draft)

    # Training environment Options
    # game.env_options.GodReward = False

    model = InferenceEngine(args.model)

    with game:
        state = game.initial()

        # Initialize Drafter & Encoders
        if game.options.draft:
            model.init_draft()
        # ---

        # Draft here if enabled
        while game.running:

            if game.options.draft:
                pass

            break

        game.wait_end_draft()
        model.close_draft()
        model.init_play(game)

        # Play the game
        while game.running:
            # Start issuing orders here
            action = model.action(state)

            # take a random action
            state, reward, done, info = game.step(action)

            if state is None:
                break

            if game.cnt > 0 and game.cnt % 100 == 0:
                print(f'Step time {game.avg / game.cnt:.4f}')

            print(reward[0], reward[-1])

        print('Game Finished')

    print('Done')


if __name__ == '__main__':
    main()
