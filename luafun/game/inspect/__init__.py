import os
import logging
import multiprocessing as mp
import weakref

from rpcjs.dashboard import Dashboard
import rpcjs.elements as html
from rpcjs.page import Page

from jinja2 import Environment, PackageLoader, select_autoescape

from luafun.game.inspect.state import GameInspector
from luafun.game.inspect.action import Actions

log = logging.getLogger(__name__)


class ShowRoutes(Page):
    """Display available routes the user can take"""
    @staticmethod
    def routes():
        return '/'

    def __init__(self, dash, env):
        super(ShowRoutes, self).__init__()
        self.dash = dash
        self.title = 'Routes'
        self.env = env

    def main(self):
        routes = [
            html.link(html.chain(html.span(name), ':', html.code(spec)), spec)
                for spec, name in self.dash.routes
        ]
        page = self.env.get_template('routes.html')
        return page.render(
            routes=html.div(
                html.header('Routes', level=4),
                html.ul(routes))
        )


def add_static_path(dash, path):
    self_ref = weakref.ref( dash.app)
    dash.app.add_url_rule(
        f"/{path}/<path:filename>",
        endpoint="static2",
        host=None,
        view_func=lambda **kw: self_ref().send_static_file(**kw),
    )


def _http_inspect(state, rpc_recv, rpc_send, level):
    logging.basicConfig(level=level)

    with Dashboard(__name__) as dash:
        env = Environment(
            loader=PackageLoader('luafun.game.inspect', 'templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )

        dash.app.config['DEBUG'] = False
        add_static_path(dash, os.path.join(os.path.dirname(__file__), 'static'))

        dash.add_page(ShowRoutes(dash, env))
        dash.add_page(GameInspector(env, rpc_recv, rpc_send))
        dash.add_page(Actions(env, state, rpc_recv, rpc_send))
        dash.run()



def http_inspect(state, rpc_recv, rpc_send, level):
    p = mp.Process(
        target=_http_inspect,
        args=(state, rpc_recv, rpc_send, level)
    )
    p.start()
    log.debug(f'HTTP-server: {p.pid}')
    return p
