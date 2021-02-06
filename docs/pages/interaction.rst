Player Interaction
==================

Humans can create lobby filled with bots that they will play with by using the command below.
It will launch a Dota2 game instance that look standard but it will have all the necessary setup done
for bots to send and receive updates from the model.

1. Launch a game with the command below
2. click on Play Dota
3. Select Custom Lobby + Create
4. Edit Lobby Setting
5. Bot Settings > Radiant Bots > Local dev Script
6. Bot Settings > Dire Bots > Local dev Script
7. Fill empty slots with bots
8. OK
9. Start Game

.. code-block:: python

    luafun --render --speed 1 --interactive



.. image:: ../_static/interactive_play.gif
    :width: 100%

Additionally, humans can spectate full bot match by opening the Dota2 console and typing ``jointeam spec``,
to join the game as a spectator.
