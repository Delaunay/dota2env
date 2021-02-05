import os
import json
from enum import IntEnum

# Map constant
# Extracted using Lua
# might be automated so it never gets out of date


def load_source_file(name):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, name), 'r') as f:
        return json.load(f)


# World Bound
BOUNDS = [
    (-8288, -8288),
    (8288, 8288)
]

RANGE = (8288, 8288)

# Game Unit
# (16576, 16576)
SIZE = (
    BOUNDS[1][0] - BOUNDS[0][0],    # x_max -  x_min
    BOUNDS[1][1] - BOUNDS[0][1],    # y_max - y_min
)

#
# Max vision comes from the Dota2 Lua API some functions are limited to this range
# Max Vision = 1600
# True max vision is 1800, 800 at night


def position_to_key(x, y, div=10):
    """Generate a position key to query entities by their position

    Examples
    --------

    Red is the collision square and black is the position with an arbitrary size.
    The position (x, y) is always inside the red square but it is not at the center.

    Our goal here is not to have an accurate collision but rather
    an efficient lookup from position with a margin of error (the red square).

    The lookup is most precise when the position is a multiple of ``div`` and least
    precise when around half of ``div``.

    .. image:: ../_static/position_mapping.png

    Notes
    -----
    This essentially capture anything in a ``div`` unit square.
    The unit/entity is not at the center of the square.

    Collision in dota is standardized so there is only a few sizes we need to worry about.
    We chose ``div = 10`` because it seemed a good middle ground.

    This method makes the unit/tree selection a bit fuzzy, if entities are close together
    they could be mis-selected

    .. code-block:: python

        # Collision sizes
        DOTA_HULL_SIZE_BUILDING        = 298 #  Ancient
        DOTA_HULL_SIZE_TOWER 	       = 144 # Barracks
        TREES                          = 128
        DOTA_HULL_SIZE_FILLER 	       =  96 # Fillers / Outpost
        DOTA_HULL_SIZE_HUGE 	       =  80 # Power Cog
        DOTA_HULL_SIZE_HERO 	       =  24 # <== Mostly Heroes
        DOTA_HULL_SIZE_REGULAR 	       =  16 # <== Melee Creep
        DOTA_HULL_SIZE_SMALL 	       =   8 # <== Range Creep
        DOTA_HULL_SIZE_SMALLEST        =   2 # Zombie

    """
    # Extract the fractional part
    # so if we are close to a frontier we cover it
    ox = (x - int(x / div) * div)
    oy = (y - int(y / div) * div)

    x = x + ox / 2
    y = y + oy / 2

    return f'{int(x / div)}{int(y / div)}'


IGNORED_TREES = dict()
DUP_TREES = dict()


# Trees
def load_trees():
    trees = load_source_file('resources/trees.json')
    position_to_tree = dict()

    for tid, x, y, z in trees:
        tree = {
            'id': tid,
            'loc': (x, y, z)
        }

        key = position_to_key(x, y)

        if key in position_to_tree:
            x1, y1, z1 = position_to_tree[key]['loc']

            d1 = x1 * x1 + y1 * y1
            d = x * x + y * y

            # Favor trees that are closer to the origin (0, 0)
            if d < d1:
                position_to_tree[key] = tree

                DUP_TREES[key] = (x, y, z)
                IGNORED_TREES[key] = (x1, y1, z1)

            else:
                IGNORED_TREES[key] = (x, y, z)
                DUP_TREES[key] = (x1, y1, z1)

            print(f'Duplicate tree {key:>10}: [({x}, {y}, {z}), ({x1}, {y1}, {z1})]')
            continue

        position_to_tree[key] = tree

    if len(IGNORED_TREES):
        print('Total ignored trees:', len(IGNORED_TREES))
    return position_to_tree


def get_tree(x, y):
    tree_key = position_to_key(x, y)
    t = TREES.get(tree_key)
    if t:
        return t.get('id', -1)
    return -1


TREES = load_trees()
TREE_COUNT = len(TREES)


# Runes
def rune_lookup():
    runes = dict()
    all_runes = load_source_file('resources/runes.json')
    for rid, x, y, z in all_runes:
        key = position_to_key(x, y, div=100)

        if key in runes:
            print('Duplicate rune!')

        runes[key] = rid

    assert len(runes) == len(all_runes)
    return runes


def get_rune(x, y):
    key = position_to_key(x, y, div=100)
    return RUNES.get(key)


RUNES = rune_lookup()
RUNE_COUNT = len(RUNES)

SHOPS = load_source_file('resources/shops.json')
SHOP_COUNT = len(SHOPS)

NEUTRALS = load_source_file('resources/neutrals.json')
NEUTRAL_COUNT = len(NEUTRALS)

ABILITIES = load_source_file('resources/abilities.json')
ABILITY_COUNT = len(ABILITIES)

HEROES = load_source_file('resources/heroes.json')
HERO_COUNT = len(HEROES)


MAX_ABILITY_COUNT_PER_HEROES = 24


class HeroLookup:
    """Help bring some consistency with ability index.
    Move all the talent up to the end of the ability array.
    This makes the talent consistent with invoker ability array which is the big exception

    Examples
    --------
    >>> h = HERO_LOOKUP.from_id(112)
    >>> for a in h['abilities']:
    ...     print(a)
    winter_wyvern_arctic_burn
    winter_wyvern_splinter_blast
    winter_wyvern_cold_embrace
    generic_hidden
    generic_hidden
    winter_wyvern_winters_curse
    None
    None
    None
    special_bonus_unique_winter_wyvern_5
    special_bonus_attack_damage_50
    special_bonus_hp_275
    special_bonus_night_vision_400
    special_bonus_unique_winter_wyvern_1
    special_bonus_unique_winter_wyvern_2
    special_bonus_unique_winter_wyvern_3
    special_bonus_unique_winter_wyvern_4
    None
    None
    None
    None
    None
    None
    None

    Using the remapped ability index

    >>> h = HERO_LOOKUP.from_id(112)
    >>> for i in h['remap']:
    ...     print(h['abilities'][i])
    winter_wyvern_arctic_burn
    winter_wyvern_splinter_blast
    winter_wyvern_cold_embrace
    generic_hidden
    generic_hidden
    winter_wyvern_winters_curse
    None
    None
    None
    None
    None
    None
    None
    None
    None
    None
    special_bonus_unique_winter_wyvern_5
    special_bonus_attack_damage_50
    special_bonus_hp_275
    special_bonus_night_vision_400
    special_bonus_unique_winter_wyvern_1
    special_bonus_unique_winter_wyvern_2
    special_bonus_unique_winter_wyvern_3
    special_bonus_unique_winter_wyvern_4

    Invoker

    >>> h = HERO_LOOKUP.from_id(74)
    >>> for i in h['remap']:
    ...     print(h['abilities'][i])
    invoker_quas
    invoker_wex
    invoker_exort
    invoker_empty1
    invoker_empty2
    invoker_invoke
    invoker_cold_snap
    invoker_ghost_walk
    invoker_tornado
    invoker_emp
    invoker_alacrity
    invoker_chaos_meteor
    invoker_sun_strike
    invoker_forge_spirit
    invoker_ice_wall
    invoker_deafening_blast
    special_bonus_unique_invoker_10
    special_bonus_unique_invoker_6
    special_bonus_unique_invoker_13
    special_bonus_unique_invoker_9
    special_bonus_unique_invoker_3
    special_bonus_unique_invoker_5
    special_bonus_unique_invoker_2
    special_bonus_unique_invoker_11
    """
    def __init__(self):
        self.ability_count = 0
        self._from_id = dict()
        self._from_name = dict()
        self._ability_remap = dict()

        for hero in HEROES:
            self.ability_count = max(self.ability_count, len(hero.get('abilities', [])))
            self._from_id[hero['id']] = hero
            self._from_name[hero['name']] = hero
            hero['remap'] = self._remap_abilities(hero)

    def _remap_abilities(self, hero):
        # `special` count = 960 | 121 * 8 = 968
        # npc_dota_hero_target_dummy is not a real hero
        remapped_talents = []
        remapped_abilities = []

        for i, ability in enumerate(hero.get('abilities', [])):
            if ability and 'special' in ability:
                remapped_talents.append(i)
            else:
                remapped_abilities.append(i)

        abilites = [None] * MAX_ABILITY_COUNT_PER_HEROES

        for i in range(len(remapped_abilities)):
            abilites[i] = remapped_abilities[i]

        # insert talents at the end
        for i in range(len(remapped_talents)):
            abilites[- len(remapped_talents) + i] = remapped_talents[i]

        return abilites

    def from_id(self, id):
        """Get hero info from its id"""
        return self._from_id.get(id)

    def from_name(self, name):
        """Get hero info from its names"""
        return self._from_name.get(name)

    @staticmethod
    def remap(hero, aid):
        """Remap hero ability id to game ability id

        Examples
        --------
        >>> from luafun.game.action import AbilitySlot
        >>> am = HERO_LOOKUP.from_id(1)
        >>> HeroLookup.remap(am, AbilitySlot.Q)
        17
        >>> HeroLookup.remap(am, AbilitySlot.Talent42)
        33

        >>> invoker = HERO_LOOKUP.from_id(74)
        >>> HeroLookup.remap(invoker, AbilitySlot.Q)
        17
        >>> HeroLookup.remap(invoker, AbilitySlot.Talent42)
        40
        """
        n = len(ItemSlot)

        if n <= aid < 41:
            return hero['remap'][aid - n] + n

        return aid

    def ability_from_id(self, hid, aid):
        """Get the game ability from hero id and model ability id"""
        return HeroLookup.remap(self._from_id.get(hid), aid)

    def ability_from_name(self, name, aid):
        """Get the game ability from hero name and model ability id"""
        return HeroLookup.remap(self._from_name.get(name), aid)


HERO_LOOKUP = HeroLookup()

ITEMS = load_source_file('resources/items.json')
ITEM_COUNT = len(ITEMS)


class Lanes(IntEnum):
    Roam = 0
    Top = 1
    Mid = 2
    Bot = 3


class RuneSlot(IntEnum):
    PowerUpTop = 0
    PowerUpBottom = 1
    BountyRiverTop = 2
    BountyRadiant = 3
    BountyRiverBottom = 4
    BountyDire = 5


# indices are zero based with 0-5 corresponding to inventory, 6-8 are backpack and 9-15 are stash
class ItemSlot(IntEnum):
    # Inventory
    Item0 = 0
    Item1 = 1
    Item2 = 2
    Item3 = 3
    Item4 = 4
    Item5 = 5
    Bakcpack1 = 6
    Bakcpack2 = 7
    Bakcpack3 = 8
    Stash1 = 9
    Stash2 = 10
    Stash3 = 11
    Stash4 = 12
    Stash5 = 13
    Stash6 = 14
    Item15 = 15     # TP
    Item16 = 16     # Neutral ?


assert len(ItemSlot) == 17, '17 item slots'


# might have to normalize talent so it is easier to learn
class SpellSlot(IntEnum):
    Ablity0 = 0         # Q                 | invoker_quas
    Ablity1 = 1         # W                 | invoker_wex
    Ablity2 = 2         # E                 | invoker_exort
    Ablity3 = 3         # D generic_hidden  | invoker_empty1
    Ablity4 = 4         # F generic_hidden  | invoker_empty2
    Ablity5 = 5         # R                 | invoker_invoke
    Ablity6 = 6         # .                 | invoker_cold_snap
    Ablity7 = 7         # .                 | invoker_ghost_walk
    Ablity8 = 8         # .                 | invoker_tornado
    Ablity9 = 9         # .                 | invoker_emp
    Ablity10 = 10       # .                 | invoker_alacrity
    Ablity11 = 11       # .                 | invoker_chaos_meteor
    Ablity12 = 12       # .                 | invoker_sun_strike
    Ablity13 = 13       # .                 | invoker_forge_spirit
    Ablity14 = 14       # .                 | invoker_ice_wall
    Ablity15 = 15       # .                 | invoker_deafening_blast
    Ablity16 = 16       # Talent 1  (usually but the talent offset can be shifted)
    Ablity17 = 17       # Talent 2  example: rubick, invoker, etc..
    Ablity18 = 18       # Talent 3
    Ablity19 = 19       # Talent 4  98 heroes follow the pattern above
    Ablity20 = 20       # Talent 5
    Ablity21 = 21       # Talent 6
    Ablity22 = 22       # Talent 7
    Ablity23 = 23       # Talent 8


assert len(SpellSlot) == 24, '24 abilities'


# Could bundle the courier action as a hero action
class CourierAction(IntEnum):
    BURST               = 0
    # hidden
    # ENEMY_SECRET_SHOP   = 1
    RETURN              = 2
    SECRET_SHOP         = 3
    TAKE_STASH_ITEMS    = 4
    TRANSFER_ITEMS      = 5
    # bots will have to do 2 actions for those
    # not a big deal IMO
    # TAKE_AND_TRANSFER_ITEMS
    # COURIER_ACTION_SIDE_SHOP
    # COURIER_ACTION_SIDE_SHOP2
