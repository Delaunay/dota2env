Dota2 Gym Environment
=====================

Straight forward API with full bot take over

.. code-block:: python

    from luafun.dotaenv import dota2_environment

    env = dota2_environment('mid1v1')
    env.render()

    with env:
        state = env.initial()

        # Draft here if enabled
        while env.running:
            # env.step(select_hero)
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


.. image:: _static/sfmid.gif


.. toctree::
   :caption: Getting Started

   pages/installation.rst


.. toctree::
   :caption: API
   :maxdepth: 1

   pages/dotaenv
   pages/action
   pages/observation
   pages/reward
   pages/draft
   pages/model

