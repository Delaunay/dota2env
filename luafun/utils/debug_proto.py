import base64

from luafun.game.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from google.protobuf.message import DecodeError


def main():

    with open('/home/setepenre/work/LuaFun/decode_error/summ.txt', 'w') as out:
        stop = False
        partial = None

        for i in range(0, 182):
            try:
                with open(f'/home/setepenre/work/LuaFun/decode_error/decode_{i}.txt', 'rb') as f:
                    data = f.read()

                bytes = base64.b64decode(data)

                if partial:
                    partial = partial + bytes
                    stop = True
                    print('Try to parse partial')

                msg = CMsgBotWorldState()
                size_read = msg.ParseFromString(bytes)
                remaining = len(bytes) - size_read

                if remaining > 0:
                    partial = bytes[size_read:]
                    print('')

                if stop:
                    return

            except DecodeError:
                print(i, 'bad message')


if __name__ == '__main__':
    import sys
    sys.stderr = sys.stdout
    main()
