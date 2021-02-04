Dota 2 ML Bots
==============

![Dota2 RL Env](doc/sfmid.gif)

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
        action = env.fix_sampled_actions(
            env.action_space.sample()
        )

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

# Platform

* works best on linux
* there is a bug in windows where state are being dropped randomly (unknown reason)

#  Install

0. Install Dota2
1. Download [Anaconda 3][1]
2. Start > Anaconda Prompt (anaconda)
3. git clone https://github.com/Delaunay/LuaFun.git
4. cd LuaFun
4. Install Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
pip install -e .
```

5. change the bot folder

```
DOTA_PATH=/media/setepenre/local/SteamLibraryLinux/steamapps/common/dota\ 2\ beta
cd $DOTA_PATH/game/dota/scripts/vscripts
ln -s ~/work/LuaFun/bots/ bots
```

6. Run the simulation
7. Use you browser to see how it is going
    http://localhost:5000/draw/radiant

[1]: https://www.anaconda.com/products/individual

# Features

* Full draft control
* Full Hero take over

# Known Issues

* Entering spectator mode after drafting phase can cause crashes


# Steam Web API

Steam Web API is only used to bootstrap some components to gain some time when training.
You might be able to skip these steps

* register a key https://steamcommunity.com/dev/registerkey
* set it inside `config.json`
* Prevent git from picking up the change
    git update-index --assume-unchanged config.json

