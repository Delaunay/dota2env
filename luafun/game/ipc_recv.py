import asyncio
import json
import logging
import os
import re
import traceback

from pygtail import Pygtail

log = logging.getLogger(__name__)

IPC_RECV = re.compile(r'\[IPC\](?P<faction>[0-9])\.(?P<player>[0-9])\t(?P<message>.*)')


async def ipc_recv(logfilename, handler, state, retries=10):
    """The only way for bots to send message back is through the log file
    we have standardized our log lines so we know which bot is sending us a message
    """
    try:
        os.remove(f'{logfilename}.offset')
    except:
        pass

    # wait for the file to be created
    for i in range(retries):
        if os.path.exists(logfilename):
            log.debug(f'Starting IPC recv after {i} retries')
            break
    
        await asyncio.sleep(0.5)
    else:
        msg = 'Impossible to initialize IPC recv'
        log.error(msg)
        raise RuntimeError(msg)
    
    while state.running:
        try:
            for line in Pygtail(logfilename):
                result = IPC_RECV.search(line)

                if result:
                    msg = result.groupdict()
                    handler(msg.get('faction'), msg.get('player'), json.loads(msg.get('message')))
        except Exception as e:
            log.debug(f'IPC error {e}') 
            log.error(traceback.format_exc())

        # when we are at the end of the file restart
        await asyncio.sleep(0.01)

    log.debug('IPC recv finished')
