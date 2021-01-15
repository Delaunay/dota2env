import asyncio
from dataclasses import dataclass
import logging
from struct import unpack

from luafun.game.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

log = logging.getLogger(__name__)


async def worldstate_listener(port, message_handler, state, retries=10):
    """Dota2 send us struct with what changed in the world
    we need to stitch them together to get the whole picture
    """
    reader = None

    for i in range(retries):
        try:
            await asyncio.sleep(0.5)
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            log.debug(f'Connection established after {i} retries')
        except ConnectionRefusedError:
            pass
        else:
            break

    if reader is None:
        log.debug('Failed to connect to the game')
        return

    while True:
        data = await reader.read(4)

        if len(data) != 4:
            log.debug('Could not read message length')
            state.running = False
            break

        n_bytes = unpack("@I", data)[0]
        data = await reader.read(n_bytes)

        world_state = CMsgBotWorldState()
        world_state.ParseFromString(data)

        # wait finishing processing the state before
        # getting a new one
        await message_handler(world_state)

    log.debug('World state listener shutting down')
