import asyncio
import json 

from rpcjs.dashboard import Dashboard
from rpcjs.page import Page
import rpcjs.elements as html

from luafun.dotaenv import Dota2Env


class Dota2EnvInspector(Page):
    def routes(self):
        return ['/']

    def __init__(self, game: Dota2Env):
        self.game: Dota2Env = game

    def main(self):
        state = self.game.dire_state
        return '<pre>' + json.dumps(state, indent=2) + '<\pre>'


def main():
    import logging
    from luafun.game.ipc_send import new_ipc_message
    logging.basicConfig(level=logging.DEBUG)

    asyncio.set_event_loop(asyncio.SelectorEventLoop())

    with Dashboard(__name__) as dash:
        # Game
        game = Dota2Env('F:/SteamLibrary/steamapps/common/dota 2 beta/', False)

        # Setup
        dash.add_page(Dota2EnvInspector(game))

        with game:
            game.send_message(new_ipc_message())


            # -- 
            dash.run()
            game.wait()


if __name__ == '__main__':
    main()
