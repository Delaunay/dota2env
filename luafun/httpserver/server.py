from dataclasses import dataclass, field
import logging
from struct import unpack
import select
import os
import socket
import traceback
import json

from jinja2 import Environment, PackageLoader, select_autoescape

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


@dataclass
class HTTPResponse:
    status: int = 200
    headers: dict = field(default_factory=dict)
    body: bytes = b''

    def setbody(self, data, content_type):
        if isinstance(data, str):
            data = data.encode('utf-8')

        self.body = data
        self.headers['Content-Type'] = content_type
        self.headers['Content-Length'] = len(data)
        self.headers['Expires'] = 'Wed, 21 Oct 2015 07:28:00 GMT'

    def set_html(self, data):
        self.setbody(data, 'text/html;charset=utf-8')

    def set_css(self, data):
        self.setbody(data, 'text/css;charset=utf-8')

    def set_js(self, data):
        self.setbody(data, 'text/javascript;charset=utf-8')

    def set_img(self, data):
        self.setbody(data, 'image/png')

    def tobytes(self):
        data = [
            f'HTTP/1.1 {self.status} {HTTP_STATUS[self.status]}',
        ]

        for k, v in self.headers.items():
            data.append(f'{k}: {v}')

        data.append('')
        data.append(self.body.decode('utf-8'))
        return '\r\n'.join(data).encode('utf-8')


class HttpServer:
    def __init__(self, state):
        self.state = state
        self.routes = {
            '/stop': self.stop_server
        }
        # self.inputs = []
        self.env = Environment(
            loader=PackageLoader('luafun.httpserver', 'templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )

    #     self.static_files = dict()
    #     self.load_static_files()
    #     self.folder = None

    # def load_static_files(self):
    #     self.folder = os.path.join(os.path.dirname(__file__), 'static')

    #     for f in os.listdir(os.path.join(self.folder, 'css')):
    #         self.static_files.add(os.path.hoin('css', f))

    #     for f in os.listdir(os.path.join(self.folder, 'js')):
    #         self.static_files.add(os.path.hoin('js', f))

    def connect(self):
        port = option('debug.port', 8080, int)
        host = "127.0.0.1"

        self.sock = socket.create_server((host, port))
        log.debug(f'Listening to {host}:{port}')
        # self.inputs.append(self.sock)

    @property
    def running(self):
        return self.state['running']

    def stop_server(self, request):
        self.state['running'] = False
        return self.default_route(request)

    def html(self, body):
        return f'<html><head></head><body>{body}</body></html>'

    def default_route(self, request):
        page = self.env.get_template('routes.html')
        return page.render(
            title='Routes',
            routes=self.routes,
            headers=json.dumps(asdict(request), indent=2)
        )

    def process_request(self, client):
        data = client.recv(1024)
        message = data.decode()

        if message == '':
            return

        try:
            request = parse_request(message)
        except Exception as err:
            log.debug(f'Error {err}')
            log.debug(message)
            return

        route_handler = self.routes.get(request.uri, self.default_route)
        reply = route_handler(request)

        response = HTTPResponse()
        response.set_html(reply)

        client.send(response.tobytes())
        client.close()

    def _run(self):
        readable, _, error = select.select([self.sock], [], [self.sock], 0.250)

        for s in readable:
            if s is self.sock:
                client, client_address = s.accept()
                client.setblocking(1)
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


HTTP_STATUS = {
    # 1×× Informational
    100: 'Continue',
    101: 'Switching Protocols',
    102: 'Processing',

    # 2×× Success
    200: 'OK',
    201: 'Created',
    202: 'Accepted',
    203: 'Non-authoritative Information',
    204: 'No Content',
    205: 'Reset Content',
    206: 'Partial Content',
    207: 'Multi-Status',
    208: 'Already Reported',
    226: 'IM Used',

    # 3×× Redirection
    300: 'Multiple Choices',
    301: 'Moved Permanently',
    302: 'Found',
    303: 'See Other',
    304: 'Not Modified',
    305: 'Use Proxy',
    307: 'Temporary Redirect',
    308: 'Permanent Redirect',

    # 4×× Client Error
    400: 'Bad Request',
    401: 'Unauthorized',
    402: 'Payment Required',
    403: 'Forbidden',
    404: 'Not Found',
    405: 'Method Not Allowed',
    406: 'Not Acceptable',
    407: 'Proxy Authentication Required',
    408: 'Request Timeout',
    409: 'Conflict',
    410: 'Gone',
    411: 'Length Required',
    412: 'Precondition Failed',
    413: 'Payload Too Large',
    414: 'Request-URI Too Long',
    415: 'Unsupported Media Type',
    416: 'Requested Range Not Satisfiable',
    417: 'Expectation Failed',
    # 418 I'm a teapot
    421: 'Misdirected Request',
    422: 'Unprocessable Entity',
    423: 'Locked',
    424: 'Failed Dependency',
    426: 'Upgrade Required',
    428: 'Precondition Required',
    429: 'Too Many Requests',
    431: 'Request Header Fields Too Large',
    444: 'Connection Closed Without Response',
    451: 'Unavailable For Legal Reasons',
    499: 'Client Closed Request',

    # 5×× Server Error
    500: 'Internal Server Error',
    501: 'Not Implemented',
    502: 'Bad Gateway',
    503: 'Service Unavailable',
    504: 'Gateway Timeout',
    505: 'HTTP Version Not Supported',
    506: 'Variant Also Negotiates',
    507: 'Insufficient Storage',
    508: 'Loop Detected',
    510: 'Not Extended',
    511: 'Network Authentication Required',
    599: 'Network Connect Timeout Error',
}


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    s = HttpServer(dict(running=True))
    s.run()
