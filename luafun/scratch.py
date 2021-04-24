"""You can ignore that file"""


def guess_path():
    from sys import platform

    s = 'F:/SteamLibrary/steamapps/common/dota 2 beta/'

    if platform == "linux" or platform == "linux2":
        s = '/media/setepenre/local/SteamLibraryLinux/steamapps/common/dota2/'

    return s


def gym_main():
    import sys

    from luafun.utils.options import option
    sys.stderr = sys.stdout

    p = option('dota.path', guess_path())
    main(p)


from luafun.game.constants import position_to_key


def make_pos_v1(x, y, div=10):
    return f'{int(x / div)}{int(y / div)}'


def make_pos_v3(x, y, div=10):
    import math
    ox = (x - int(x / div) * div)
    oy = (y - int(y / div) * div)

    xx = int((x + ox / 4) / div)
    yy = int((y + oy / 4) / div)

    r = math.sqrt(x * x + y * y)
    a = math.atan2(y, x)

    # return f'{xx}{yy}{int(a * 10)}'
    return f'{xx}{yy}'


def position_viz(div=100, hash=position_to_key):
    from PIL import Image
    import numpy as np

    w, h = 512 + 256, 512 + 256
    m = 32

    black = [0, 0, 0]
    red = [255, 0, 0]
    white = [125, 125, 125]

    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[:, :, :] = white

    for i in range(-w // 2 + 1, w // 2 - 1):
        if i % div == 0:
            data[:, i] = black

    for i in range(-h // 2 + 1, h // 2 - 1):
        if i % div == 0:
            data[i, :] = black

    data[h // 2, :] = black
    data[:, w // 2] = black

    def draw(px, py):
        # floor(px / div) * div
        origin = hash(py, px, div=div)

        for i in range(-w // 2 + 1, w // 2 - 1):
            for j in range(-h // 2 + 1, h // 2 - 1):
                k = hash(j, i, div=div)

                if origin == k:
                    data[h // 2 - j, w // 2 + i, :] = red

                if px - m < i < px + m:
                    if py - m < j < py + m:
                        data[h // 2 - j, w // 2 + i, :] = [0, 0, 0]

                if px - 2 < i < px + 2:
                    if py - 2 < j < py + 2:
                        data[h // 2 - j, w // 2 + i, :] = [255, 255, 255]

    config = [
        (-div * 2.5, div * 2.5),
        (         0, div * 2.5),
        (   div * 2, div * 2.5),

        (-div * 2.5  , 0),
        (       0  , 0),
        ( div * 2, 0),

        (-div * 2, -div * 2.5),
        (0, -div * 2.5),
        (div * 2.5, -div * 2.5),
    ]

    for args in config:
        draw(*args)

    img = Image.fromarray(data, 'RGB')
    img.save('position_mapping_v3.png')
    # img.show()


def check_tensor_view():
    import torch

    from luafun.model.actor_critic import LSTMDrafter

    draft = torch.randn(2, 12, 24, 122)

    drafter = LSTMDrafter()

    e1 = drafter.encode_draft_fast(draft).cpu()

    e2 = drafter.encode_draft(draft).cpu()

    print((e1 - e2).abs().mean())

    import json
    with open('a.txt', 'w') as a:
        a.write(json.dumps(e1.tolist(), indent=2))

    with open('b.txt', 'w') as b:
        b.write(json.dumps(e2.tolist(), indent=2))



if __name__ == '__main__':
    import torch

    # draft = torch.randn(2, 12, 24, 122)

    # position_viz(hash=make_pos_v3)
    check_tensor_view()
