Dota 2 ML Bots
==============

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4514210.svg)](https://doi.org/10.5281/zenodo.4514210)

[![codecov](https://codecov.io/gh/Delaunay/LuaFun/branch/master/graph/badge.svg?token=ILL29DPOE3)](https://codecov.io/gh/Delaunay/LuaFun)


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
* Multiple agents
* Agents needs to cooperate to win

Like in real life you will need to
1. Choose the relevant data to solve the problem and discard unnecessary data
2. Choose the correct rewards balance early rewards and late rewards
3. Encode a variety of actions for efficient processing


# Area of research

* Will an untrained agent trained faster if paired with trained agents ?
    * i.e is the untrained agent able to learn from allies & enemies

Hypothesis: Might be able to learn better from enemies has they punish bad plays strongly
while allies cannot punish nor reward an action directly
Might lead to untrained agent learning to do safer plays instead and lack the risk taking needed to
get back into the game

* Unit Agents
    * One agent controlling all 10

* Split Agents
    * n in \[2-10\] agent controlling 10


# Assets

Most assets located in `luafun/game/resources` are owned by Valve.

# Platforms

* Windows & Linux
* works best on linux


NB: Windows might drop states after a few 1000th steps.
This is a known issue, unfortunately the connection seems to drop on
dota side and there is nothing we can do. Thankfully the issue seems to be
mostly transient

# Install

For installation procedure see [Installation][2]

[2]:

# Features

* Full draft control
* Full Hero take over
* No scripted logic

# Dota2 Major Update Check list

* Update Map: regenerate the map image
* Add Abilities
* Add Hero
* Add Items
* Add Actions
* Update Rune location
* Update Outpost location
* Update tree location

# Known Issues

* Entering spectator mode after drafting phase can cause crashes
* Bots do not control their illusions/minions
* Game can start firing dev warning `Overflow in GetSerializedWorldState!\n`

# Steam Web API

Steam Web API is only used to bootstrap some components to gain some time when training.
You might be able to skip these steps

* register a key https://steamcommunity.com/dev/registerkey
* set it inside `config.json`
* Prevent git from picking up the change
    `git update-index --assume-unchanged config.json`

# Color Palette

https://colorhunt.co/palette/252860

75cfb8
bbdfc8
f0e5d8
ffc478

Font: Comfortaa

