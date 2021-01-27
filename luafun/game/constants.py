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
    BOUNDS[1][0] - BOUNDS[0][0], # x_max -  x_min
    BOUNDS[1][1] - BOUNDS[0][1], # y_max - y_min
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
