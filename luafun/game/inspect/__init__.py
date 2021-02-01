import os
import logging
import multiprocessing as mp
import weakref

from rpcjs.dashboard import Dashboard
import rpcjs.elements as html
from rpcjs.page import Page

from werkzeug.routing import FloatConverter as BaseFloatConverter
from jinja2 import Environment, PackageLoader, select_autoescape

from luafun.game.inspect.state import GameInspector, DrawMap
from luafun.game.inspect.action import Actions
from luafun.game.inspect.status import Status


log = logging.getLogger(__name__)


class FloatConverter(BaseFloatConverter):
    regex = r'-?\d+(\.\d+)?'


class BoolConverter(BaseFloatConverter):
    regex = r'[01]'


def to_bool(value):
    return str(value).lower()


def select(value, format):
    if value:
        return format
    return ''


class AppState:
    def __init__(self, state, dash, rpc_recv, rpc_send):
        self.state = state
        self.dash = dash
        self.rpc_recv = rpc_recv
        self.rpc_send = rpc_send
        self.env = Environment(
            loader=PackageLoader('luafun.game.inspect', 'templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )

        self.env.filters['bool'] = to_bool
        self.env.filters['select'] = select


class ShowRoutes(Page):
    """Display available routes the user can take"""
    @staticmethod
    def routes():
        return '/'

    def __init__(self, app):
        super(ShowRoutes, self).__init__()
        self.app = app
        self.dash = app.dash
        self.title = 'Routes'
        self.env = app.env

    def main(self):
        routes = [
            html.link(html.chain(html.span(name), ':', html.code(spec)), spec)
                for spec, name in self.dash.routes
        ]
        page = self.env.get_template('routes.html')
        return page.render(
            routes=html.div(
                html.header('Routes', level=4),
                html.ul(routes)),
            state=self.app.state
        )


def add_static_path(dash, path):
    self_ref = weakref.ref( dash.app)
    dash.app.add_url_rule(
        f"/{path}/<path:filename>",
        endpoint="static2",
        host=None,
        view_func=lambda **kw: self_ref().send_static_file(**kw),
    )


def _http_inspect(state, rpc_recv, rpc_send, level, debug=False):
    logging.basicConfig(level=level)

    with Dashboard(__name__) as dash:
        state = AppState(
            state,
            dash,
            rpc_recv,
            rpc_send
        )

        dash.app.config['DEBUG'] = debug
        dash.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        dash.app.url_map.converters['float'] = FloatConverter
        dash.app.url_map.converters['bool'] = BoolConverter

        dash.add_page(ShowRoutes(state))
        dash.add_page(GameInspector(state))
        dash.add_page(Actions(state))
        dash.add_page(DrawMap(state))
        dash.add_page(Status(state))
        dash.run()


def http_inspect(state, rpc_recv, rpc_send, level):
    p = mp.Process(
        target=_http_inspect,
        args=(state, rpc_recv, rpc_send, level)
    )
    p.start()
    log.debug(f'HTTP-server: {p.pid}')
    return p


def http_monitor():
    level = logging.DEBUG

    manager = mp.Manager()
    state = manager.dict()
    state['running'] = False
    rpc_recv = manager.Queue()
    rpc_send = manager.Queue()

    _http_inspect(
        state,
        rpc_recv,
        rpc_send,
        level,
        False
    )


if __name__ == '__main__':
    http_monitor()
