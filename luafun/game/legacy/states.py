
# Legacy asyncio implementation
# Python GIL single process impl it too slow
import asyncio
from dataclasses import dataclass
import logging
import select
import socket
import multiprocessing as mp
from struct import unpack
import traceback

from luafun.game.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


LIMIT = 2 ** 25   # 32 Mo


async def connect(port, retries):
    reader = None
    writer = None

    for i in range(retries):
        try:
            await asyncio.sleep(0.5)
            reader, writer = await asyncio.open_connection('127.0.0.1', port, limit=LIMIT)
            log.debug(f'Connection established after {i} retries')
        except ConnectionRefusedError:
            pass
        else:
            break

    return reader, writer

# need to move away from asyncio and use multiprocess instead I think
async def worldstate_listener(port, message_handler, game, stats, retries=10):
    """Dota2 send us struct with what changed in the world
    we need to stitch them together to get the whole picture

    Notes
    -----
    Deprecated, this is not good enough when working with high traffic
    we need to move to a multiprocess approach
    """

    reader, writer = await connect(port, retries)

    if reader is None:
        log.debug('Failed to connect to the game')
        return

    error_state = False

    while True:
        msg_size = await reader.read(4)

        # len(msg_size) != 4
        if msg_size == b'':
            # why are we getting disconnected 
            # Re-open the connection
            await asyncio.sleep(game.deadline/2)
            stats.reconnect += 1
            log.debug('Reconnecting')
            writer.close()
            reader, writer = await connect(port, retries)
            continue

        if error_state:
            error_state = False
            log.debug('Recovered from error')

        try:
            n_bytes = int(unpack("@I", msg_size)[0])

            # this blocks the thread too much
            # read_bytes = b''
            # while len(read_bytes) < n_bytes:
            #     next_piece = await reader.read(n_bytes - len(read_bytes))
            #     read_bytes += next_piece

            read_bytes = await reader.read(n_bytes)

            # this is a very strange behaviour when ticks_per_observation=4
            # I think asyncio is probably too slow for somereason
            if len(read_bytes) != n_bytes:
                rb = await reader.read(n_bytes - len(read_bytes))
                read_bytes += rb
                stats.double_read += 1
                
                if len(read_bytes) != n_bytes:
                    log.debug(f'Could not read the full message {len(read_bytes)} != {n_bytes}')
                    stats.error += 1
                    continue

            data = read_bytes
            world_state = CMsgBotWorldState()
            world_state.ParseFromString(data)

            stats.success += 1
            stats.message_size = max(stats.message_size, n_bytes)

        except Exception as e:
            log.debug(f'Error when reading world state: {e} after {stats.success} success {stats.error} errors')
            log.debug(f'Max {stats.message_size} | Size {msg_size} {n_bytes} | Message: {data[:10]}')
            log.error(traceback.format_exc())
            stats.error += 1
            raise

        # wait finishing processing the state before
        # getting a new one
        await message_handler(world_state)

    log.debug('World state listener shutting down')


