import json

from luafun.game.ipc_recv import IPC_RECV
import luafun.game.constants as const
from luafun.observation.minimap import new_origin, show_tensor, to_image

import torch


def read_logs(filename, img):
    # this tensor is 3.3Go
    s = img.shape[1] - 1
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

                    if mtype is not None and mtype == 'P':
                        x, y, passable, height = msg.get('I')

                        ix, iy = new_origin((x, y), origin, scale)
                        img[1, int(iy), int(ix)] = passable
                        img[2, int(iy), int(ix)] = height / 5

                        if y == 0:
                            print(f'\r ({x}x{y})', end='')

                except Exception as err:
                    print('\n', x, y, ix, iy, err)

        return img


def process_topology_logs():
    """

    Notes
    -----
    Extract base topology information about the map

    .. images:: _static/topology.png

    """
    # To extract topology map
    # run the game and make one bot execute _get_passable
    # parse the game log
    # because of the size of the logs the full map is generated from
    # multiple runs
    print('Start')
    img = torch.zeros((3, 16576 + 1, 16576 + 1))  # torch.load('/media/setepenre/local/dotaextracted/minimap_2.pt')
    print('Loaded tensor')

    read_logs('/home/setepenre/work/LuaFun/bots/console.log', img)

    print('\nSaving result')
    torch.save(img, '/media/setepenre/local/dotaextracted/minimap_512.pt')

    print('Saving Image')
    to_image(img, "/media/setepenre/local/dotaextracted/passable_512.png")

    print('Done')


def add_trees_to_map():
    from luafun.observation.minimap import add_trees

    img = torch.load('/media/setepenre/local/dotaextracted/minimap.pt')

    add_trees(img)

    torch.save(img, '/media/setepenre/local/dotaextracted/minimap_3.pt')
    to_image(img, "/media/setepenre/local/dotaextracted/passable_3.png")


if __name__ == '__main__':
    process_topology_logs()

    # add_trees_to_map()

    # from luafun.game.constants import load_map
    #
    # m = load_map()
    # print(m.shape)
    #
    # import time
    #
    # time.sleep(60)

