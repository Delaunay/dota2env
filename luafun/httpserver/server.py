import asyncio
from dataclasses import dataclass
import logging
from struct import unpack
import select
import socket
import traceback
import json

from luafun.utils.python_fix import asdict
from luafun.utils.options import option


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


class HttpServer:
    def __init__(self, state):
        self.state = state
        self.routes = {
            '/stop': self.stop_server
        }
        # self.inputs = []

    def connect(self):
        port = option('debug.port', 8080, int)
        host = "127.0.0.1"

        self.sock = socket.create_server((host, port))
        log.debug(f'Listening to {host}:{port}')
        # self.inputs.append(self.sock)
        
    @property
    def running(self):
        return self.state['running']

    def stop_server(self, _):
        self.state['running'] = False
        return self.html('<pre>Stopping the server</pre>')

    def html(self, body):
        return f'<html><head></head><body>{body}</body></html>'

    def default_route(self, request):
        routes = []
        for k in self.routes:
            routes.append(f'<li><a href="{k}">{k}</a></li>')
        routes = ''.join(routes)
        return self.html(f'<ul>Routes {routes}</ul><pre>{json.dumps(asdict(request), indent=2)}</pre>')

    def process_request(self, client):
        log.debug('Read data')
        data = client.recv(1024)
        message = data.decode()

        request = parse_request(message)
        log.debug('>>> Request')
        log.debug(request)
        log.debug('Request <<<')

        route_handler = self.routes.get(request.uri, self.default_route)
        reply = route_handler(request)

        header = f"""HTTP/1.1 200 OK\r
        |Content-Length: {len(reply.encode("utf-8"))}\r
        |Content-Type: text/html;charset=utf-8\r
        |Expires: Wed, 21 Oct 2015 07:28:00 GMT\r
        |\r
        |""".replace('        |', '')

        reply = f'{header}{reply}'.encode('utf-8')
        client.send(reply)
        log.debug('Reply written')
        client.close()

    def _run(self):
        readable, _, error = select.select([self.sock], [], [self.sock], 0.250)

        for s in readable:
            if s is self.sock:
                client, client_address = s.accept()
                client.setblocking(0)

                self.process_request(client)

    def run(self):
        self.connect()
    
        while self.running:
            try:
                self._run()
            except Exception as e:
                log.error(traceback.format_exc())

        log.debug('http server shuting down')
        self.sock.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    s = HttpServer(dict(running=True))
    s.run()
