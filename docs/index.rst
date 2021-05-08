Dota2 Gym Environment
=====================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4514210.svg
   :target: https://doi.org/10.5281/zenodo.4514210


.. image:: https://codecov.io/gh/Delaunay/LuaFun/branch/master/graph/badge.svg?token=ILL29DPOE3
   :target: https://codecov.io/gh/Delaunay/LuaFun


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
            action = env.action_space.sample()

            # take a random action
            obs, reward, done, info = env.step(action)


.. image:: _static/sfmid.gif
    :width: 100%


You can call that code by simply executing the command below

.. code-block:: bash

    luafun --render --speed 1



Dota2 is the ultimate RL environment

* Long Time Horizons (40+ minutes)
* Partially observed state
* High dimensional action and observation spaces
* Multiple strategies to solve the game
* Multiple challenge in a single game (drafting + playing)
* Agents needs to cooperate to win
* trade off between multiple objectives to achieve victory

Luafun comes with a full training pipeline even basic models are available,
so you can focus on your research and not writing code.

* Drafting: select 5 and ban 7 heroes among 120+ heroes to counter you enemy and give you the advantage in lane
* Playing chose between 25 base base actions to take, 40 abilities to use, 100+ items to buy.


Features
~~~~~~~~

* Full bot take over
* Full drafting capabilities
* Integration tested
* No scripted logic
* Humans can play with bots (the game have to be launched from python)


.. toctree::
   :caption: Getting Started

   pages/installation
   pages/interaction


.. toctree::
   :caption: API
   :maxdepth: 1

   pages/drafting
   pages/options
   pages/dotaenv
   pages/action
   pages/observation
   pages/reward
   pages/draft
   pages/model
   pages/constants
   pages/training
   pages/performance



.. code-block:: text

    @software{pierre_delaunay_2021_4514210,
      author       = {Pierre Delaunay},
      title        = {{LuaFun a complete RL training environment for
                       dota2}},
      month        = feb,
      year         = 2021,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.4514210},
      url          = {https://doi.org/10.5281/zenodo.4514210}
    }