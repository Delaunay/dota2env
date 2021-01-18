import asyncio
from dataclasses import dataclass
import logging
from struct import unpack
import traceback
import json


from luafun.utils.python_fix import asdict
from luafun.game.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState

log = logging.getLogger(__name__)


@dataclass 
class HTTPRequest:
    method: str
    version: str
    uri: str
    headers: dict
    body: str

def parse_request(message):
    top, body = message.split('\r\n\r\n')

    top = top.split('\r\n')
    status = top[0]
    headers = top[1:]

    header_dict = dict()
    for header in headers:
        result = header.split(':', maxsplit=1)

        if len(result) == 2:
            k, v = result
            header_dict[k] = v.strip()
        else:
            log.error(f'Cannot parse {header}')

    method, uri, version = status.split(' ')
    return HTTPRequest(
        method,
        version,
        uri,
        header_dict,
        body)


class GameInspector:
    def __init__(self, game, server_ref):
        self.game = game
        self.server = server_ref
        self.routes = {
            '/stop': self.stop_server
        }

    async def stop_server(self):
        self.server[0].close()
        return f'<html><head></head><body><pre>Stopping the server</pre></body></html>'

    async def show_dire_state(self):
        state = await self.game.dire_state_async

        if hasattr(state, '_lock'):
            state._lock = None

        obj = asdict(state)
        reply = (
            '<html><head></head><body><pre>' + 
                json.dumps(obj, indent=2) + 
            '</pre></body></html>'
        )
    
        return reply

    async def handle_request(self, reader, writer):
        try:
            log.debug('Read data')
            data = await reader.read(1024)
            message = data.decode()

            request = parse_request(message)
        
            print('>>> Request')
            print(request)
            print('Request <<<')

            route_handler = self.routes.get(request.uri, self.show_dire_state)
            reply = await route_handler()

            header = f"""HTTP/1.1 200 OK\r
            |Content-Length: {len(reply.encode("utf-8"))}\r
            |Content-Type: text/html;charset=utf-8\r
            |Expires: Wed, 21 Oct 2015 07:28:00 GMT\r
            |\r
            |""".replace('            |', '')

            reply = f'{header}{reply}'.encode('utf-8')

            writer.write(reply)
            await writer.drain()
            log.debug('Reply written')

        except Exception as e:
            log.error(f'When processing inspect request {e}')
            log.error(traceback.format_exc())

        writer.close()
        await writer.wait_closed()
    

async def http_inspect(game, port=5000):
    """Offers a simple asyncio server to show game states in realtime for debugging"""
    server_ref = [None]

    async def async_inpsect(reader, writer):
        inspect = GameInspector(game, server_ref)
        log.debug('Process request')
        await inspect.handle_request(reader, writer)
        return 

    server = await asyncio.start_server(async_inpsect, '127.0.0.1', port)
    addr = server.sockets[0].getsockname()

    log.info(f'Serving on {addr}')
    game.http_server = server
    server_ref[0] = server

    async with server:
         # save the server future so we can stop it
        game.http_server = await server.serve_forever()

    log.info(f'Closing HTPP server')


def test_main():
    import sys
    logging.basicConfig(level=logging.DEBUG)

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    @dataclass
    class State:
        x: int = 0

    class GameMock:
        def __init__(self):
            self.state = State()
        
        @property
        async def dire_state_async(self):
            return self.state
  
    try:
        asyncio.run(http_inspect(GameMock()))
    except Exception as e:
        log.error(f'Error {str(e)}')
        log.error(traceback.format_exc())


if __name__ == '__main__':
    test_main()
