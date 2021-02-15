from enum import IntEnum, auto


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

    return int(x * scale), int(y * scale)


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
    import matplotlib.pyplot as plt
    import torch
    import luafun.game.constants as const

    img = torch.zeros(3, s, s)

    scale = s / const.SIZE[0]
    origin = const.ORIGIN

    for tid, x, y, z in const.TREES:
        ix, iy = new_origin((x, y), origin, scale)
        img[1, ix, iy] = 1

    plt.imshow(img.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    tree_minimap()
