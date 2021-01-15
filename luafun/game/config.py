import os

EXECUTABLE_PATH = '/game/bin/win64/dota2.exe'
BOT_PATH = '/game/dota/scripts/vscripts/bots/'
CONSOLE_LOG = '/game/dota/scripts/vscripts/bots/console.log'
SEND_MSG = '/game/dota/scripts/vscripts/bots/IPC_recv.lua'

LINUX_APP_PATH = "~/Steam/steamapps/common/dota 2 beta"
OSX_APP_PATH = "~/Library/Application Support/Steam/SteamApps/common/dota 2 beta"
WINDOWS_APP_PATH = "C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta"


class DotaPaths:
    """Class to hold system specific configuration"""
    def __init__(self, path=None):
        self.path = path

        if self.path is None:
            self.path = self.guess()

    def guess(self):
        from sys import platform

        if platform == "linux" or platform == "linux2":
            return os.path.expanduser(LINUX_APP_PATH)
       
        elif platform == "darwin":
            return os.path.expanduser(OSX_APP_PATH)
    
        return WINDOWS_APP_PATH

    @property
    def executable_path(self):
        return self.path + '/' + EXECUTABLE_PATH

    @property
    def ipc_recv_handle(self):
        return self.path + '/' + CONSOLE_LOG

    @property
    def console_log(self):
        return self.ipc_recv_handle

    @property
    def ipc_send_handle(self):
        return self.path + '/' + SEND_MSG

    def bot_file(self, filename):
        """Return a file path that is located in the bot folder"""
        return self.path + '/' + BOT_PATH + filename
