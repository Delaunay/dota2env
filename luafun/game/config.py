import os

EXECUTABLE_PATH_WINDOWS = '/game/bin/win64/dota2.exe'
EXECUTABLE_PATH_LINUX = '/game/dota.sh'
EXECUTABLE_PATH_LINUX = '/game/bin/linuxsteamrt64/dota2'

BOT_PATH = '/game/dota/scripts/vscripts/bots/'
CONSOLE_LOG = '/game/dota/scripts/vscripts/bots/console.log'
SEND_MSG = '/game/dota/scripts/vscripts/bots/IPC_recv.lua'
CONFIG_MSG = '/game/dota/scripts/vscripts/bots/IPC_config.lua'

LINUX_APP_PATH = "~/Steam/steamapps/common/dota 2 beta"
OSX_APP_PATH = "~/Library/Application Support/Steam/SteamApps/common/dota 2 beta"
WINDOWS_APP_PATH = "C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta"

# <steam path>/ubuntu12_32/steam-runtime/run.sh


class DotaPaths:
    """Class to hold system specific configuration"""
    def __init__(self, path=None):
        if path is None:
            path = self.guess()

        self.path = path

    def guess(self):
        from sys import platform

        if platform == "linux" or platform == "linux2":
            return os.path.expanduser(LINUX_APP_PATH)

        elif platform == "darwin":
            return os.path.expanduser(OSX_APP_PATH)

        return WINDOWS_APP_PATH

    @property
    def executable_path(self):
        from sys import platform

        if platform == "linux" or platform == "linux2":
            return self.path + '/' + EXECUTABLE_PATH_LINUX

        return self.path + '/' + EXECUTABLE_PATH_WINDOWS

    @property
    def ipc_recv_handle(self):
        return self.path + '/' + CONSOLE_LOG

    @property
    def console_log(self):
        return self.ipc_recv_handle

    @property
    def ipc_send_handle(self):
        return self.path + '/' + SEND_MSG

    @property
    def ipc_config_handle(self):
        return self.path + '/' + CONFIG_MSG

    def bot_file(self, filename):
        """Return a file path that is located in the bot folder"""
        return self.path + '/' + BOT_PATH + filename
