import json

from luafun.game.ipc_recv import IPC_RECV
import luafun.game.constants as const
from luafun.observation.minimap import new_origin, show_tensor, to_image

import torch


def read_logs(filename, s=16576):
    img = torch.zeros(3, s, s)
    scale = s / const.SIZE[0]
    origin = const.ORIGIN

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            result = IPC_RECV.search(line)

            if result:
                try:
                    msg = result.groupdict()
                    msg = json.loads(msg.get('message'))
                    mtype = msg.get('T')

                    offset = None
                    div = 1

                    if mtype == 'PASSSABLE':
                        offset = 1

                    if mtype == 'HEIGHT':
                        offset = 2
                        div = 5

                    if offset:
                        x, y, value = msg.get('I')

                        ix, iy = new_origin((x, y), origin, scale)
                        img[offset, ix, iy] = value / div

                        if i % 1000:
                            print(f'\r ({x}x{y})', end='')

                except:
                    pass

        print('DOne')
        return img


if __name__ == '__main__':
    img = read_logs('/home/setepenre/work/LuaFun/bots/console.log')

    show_tensor(img)
    torch.save(img, 'minimap.pt')
    to_image(img, "passable")

