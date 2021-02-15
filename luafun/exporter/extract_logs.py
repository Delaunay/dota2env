import json
from luafun.game.ipc_recv import IPC_RECV


def read_logs(filename):
    messages = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            result = IPC_RECV.search(line)

            if result:
                msg = result.groupdict()
                messages.append(json.loads(msg.get('message')))

    with open('extracted_logs.json', 'w') as f:
        json.dump(messages, f)


if __name__ == '__main__':
    read_logs('/home/setepenre/work/LuaFun/bots/console.log')

