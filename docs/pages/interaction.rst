Player Interaction
==================

Humans can create lobby filled with bots that they will play with by using the command below.
It will launch a Dota2 game instance that look standard but it will have all the necessary setup done
for bots to send and receive updates from the model.


.. code-block::python

    python luafun/dotaenv.py --render --speed 1 --interactive


Additionally, humans can spectate full bot match by opening the Dota2 console and typing ``jointeam spec``,
to join the game as a spectator.
