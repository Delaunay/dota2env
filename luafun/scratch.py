"""You can ignore that file"""

from luafun.dotaenv import main


def guess_path():
    from sys import platform

    s = 'F:/SteamLibrary/steamapps/common/dota 2 beta/'

    if platform == "linux" or platform == "linux2":
        s = '/media/setepenre/local/SteamLibraryLinux/steamapps/common/dota2/'

    return s


if __name__ == '__main__':
    import sys

    from luafun.utils.options import option
    sys.stderr = sys.stdout

    p = option('dota.path', guess_path())
    main(p)
