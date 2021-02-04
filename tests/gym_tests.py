from luafun.dotaenv import dota2_environment
from luafun.scratch import guess_path


def base_gym_api_tests(path=None):
    env = dota2_environment('mid1v1', path)
    env.render()

    state = env.reset()

    # Draft here if enabled
    while env.running:
        break

    env.wait_end_draft()

    # Play the game
    while env.running:
        action = env.fix_sampled_actions(
            env.action_space.sample()
        )

        # take a random action
        obs, reward, done, info = env.step(action)

    env.close()


def nice_gym_api_tests(path=None):
    env = dota2_environment('mid1v1', path)
    # env.render()

    with env:
        state = env.initial()

        # Draft here if enabled
        while env.running:
            break

        env.wait_end_draft()

        # Play the game
        while env.running:
            # Start issuing orders here
            action = env.fix_sampled_actions(
                env.action_space.sample()
            )

            # take a random action
            obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    nice_gym_api_tests(guess_path())

    # from luafun.game.action import action_space
    # print(action_space())


