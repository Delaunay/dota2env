from enum import IntEnum, auto

import torch

import luafun.game.constants as const


class MiniMapItem(IntEnum):
    Me = 0
    AlliedHero = auto()
    AlliedMelee = auto()
    AlliedRanged = auto()
    EnemyHero = auto()
    EnemyMelee = auto()
    EnemyRanged = auto()
    Tree = auto()
    Building = auto()


# We can generate a minimap from the state of the game we see
# the map size is (16576, 16576) and the smallest collision is 2
#   16576 /  2       = 8288
#   16576 /  8       = 2072
#   16576 / 16       = 1036
#   16576 / 32       =  518   # 24 = Collision of hero
#   16576 / 64       =  259   # 80 = Power Cog / 96 = Filler (resnet like)
#   16576 / (16 * 7) =  148
#   16576 / (32 * 7)     = 74
#   16576 / (37 * 7)     = 64
#   16576 / (37 * 7 * 2) = 32   # CIFAR 32x32
#                               # MNIST 28x28
#
#
#   The idea here is to find a divisor that allows us to keep the most information
#   while dividing the map into a dimension that is tractable
#

def new_origin(p, origin, scale):
    x = p[0] + origin[0]
    y = origin[1] - p[1]

    return x * scale, y * scale


def tree_minimap(s=259):
    """

    Notes
    -----
    The main issue with minimap is extracting info from the game.
    A few lua function are defined that could help us.
    but the terrain information is mostly not available.

    .. code-block:: bash
        bool IsLocationPassable( vLocation )
        bool IsLocationVisible( vLocation )
        int GetHeightLevel( vLocation )

        { hUnit, ... } GetNearbyHeroes( nRadius, bEnemies, nMode)
        { hUnit, ... } GetNearbyCreeps( nRadius, bEnemies )
        { hUnit, ... } GetNearbyLaneCreeps( nRadius, bEnemies )
        { hUnit, ... } GetNearbyNeutralCreeps( nRadius )
        { hUnit, ... } GetNearbyTowers( nRadius, bEnemies )
        { hUnit, ... } GetNearbyBarracks( nRadius, bEnemies )
        { hUnit, ... } GetNearbyShrines( nRadius, bEnemies )
        { int, ...   } GetNearbyTrees ( nRadius )

    Examples
    --------

    .. images:: ../_static/minmap_trees.png

    """

    img = torch.zeros(3, s, s)
    add_trees(img)
    return img


def add_trees(img):
    s = img.shape[1]
    scale = s / const.SIZE[0]

    tree_size = 256
    half = tree_size / 2 * scale

    origin = const.ORIGIN

    for tid, x, y, z in const.TREES:
        ix, iy = new_origin((x, y), origin, scale)
        img[0, int(iy - half):int(iy + half), int(ix - half):int(ix + half)] = 1

    return img


def to_image(tensor, filename=None):
    from torchvision import transforms
    im = transforms.ToPILImage()(tensor).convert("RGB")

    if filename:
        im.save(filename, "PNG")

    return im


def show_tensor(tensor):
    """Show a tensor as image"""
    import matplotlib.pyplot as plt

    plt.imshow(tensor.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    img = tree_minimap(16576)
    to_image(img, "tree_minimap.png")
