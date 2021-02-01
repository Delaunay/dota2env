import os
import json

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

# Game Unit
# (16576, 16576)
SIZE = (
    BOUNDS[1][0] - BOUNDS[0][0],    # x_max -  x_min
    BOUNDS[1][1] - BOUNDS[0][1],    # y_max - y_min
)

# Trees
TREES = load_source_file('resources/trees.json')
TREE_COUNT = len(TREES)

RUNES = load_source_file('resources/runes.json')
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
        return self._from_id.get(id)

    def from_name(self, name):
        return self._from_name.get(name)


HERO_LOOKUP = HeroLookup()

ITEMS = load_source_file('resources/items.json')
ITEM_COUNT = len(ITEMS)

# Collision in dota is standardized
# so there is only a few shapes to worry about

# Max vision comes from the Dota2 Lua API some functions are limited to this range
# Max Vision = 1600
# True max vision is 1800, 800 at night


# DOTA_HULL_SIZE_BUILDING 	    298 | Ancient
# DOTA_HULL_SIZE_TOWER 	        144 | Barracks
#                TREES          128
# DOTA_HULL_SIZE_FILLER 	     96 | Fillters / Outpost
# DOTA_HULL_SIZE_HUGE 	         80 | Power Cog
# DOTA_HULL_SIZE_HERO 	         24 | <== Mostly Heroes
# DOTA_HULL_SIZE_REGULAR 	     16 | <== Melee Creep
# DOTA_HULL_SIZE_SMALL 	          8 | <== Range Creep
# DOTA_HULL_SIZE_SMALLEST 	      2 | Zombie

# print(SIZE)
