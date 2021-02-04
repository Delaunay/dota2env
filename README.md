Dota 2 ML Bots
==============

![Dota2 RL Env](docs/_static_/sfmid.gif)

Random SF doing random actions

```python
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
```


Dota2 is the ultimate RL environment

* Long Time Horizons
* Partially Observed State
* High dimensional action and observation spaces
* Multiple strategies to solve the game
* Agents needs to cooperate to win

Like in real life you will need to
1. Choose the relevant data to solve the problem and discard unnecessary data
2. Choose the correct rewards balance early rewards and late rewards
3. Encode a variety of actions for efficient processing

# Assets

Most assets located in `luafun/game/resources` are owned by Valve.

# Platforms

* works best on linux
* there is a bug in windows where state are being dropped randomly (unknown reason)

# Install

For installation procedure see [Installation][2]

[2]:

# Features

* Full draft control
* Full Hero take over
* No scripted logic

# Known Issues

* Entering spectator mode after drafting phase can cause crashes
* Bots do not control their illusions/minions

# Steam Web API

Steam Web API is only used to bootstrap some components to gain some time when training.
You might be able to skip these steps

* register a key https://steamcommunity.com/dev/registerkey
* set it inside `config.json`
* Prevent git from picking up the change
    `git update-index --assume-unchanged config.json`

