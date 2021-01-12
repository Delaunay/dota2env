from enum import IntEnum


class DOTA_GameMode(IntEnum):
	DOTA_GAMEMODE_NONE = 0
	DOTA_GAMEMODE_AP = 1            #  All Pick
	DOTA_GAMEMODE_CM = 2            #  Captains Mode
	DOTA_GAMEMODE_RD = 3            #  Random Draft
	DOTA_GAMEMODE_SD = 4            #  Single Draft
	DOTA_GAMEMODE_AR = 5            #  All Random
	DOTA_GAMEMODE_INTRO = 6         #
	DOTA_GAMEMODE_HW = 7            #
	DOTA_GAMEMODE_REVERSE_CM = 8    #  Reverse Captains Mode
	DOTA_GAMEMODE_XMAS = 9          #
	DOTA_GAMEMODE_TUTORIAL = 10     #
	DOTA_GAMEMODE_MO = 11           #  Melee Only ?
	DOTA_GAMEMODE_LP = 12           #  Least Played
	DOTA_GAMEMODE_POOL1 = 13        #  Limited Heroes ?
	DOTA_GAMEMODE_FH = 14           #
	DOTA_GAMEMODE_CUSTOM = 15       #
	DOTA_GAMEMODE_CD = 16           #  Captains draft
	DOTA_GAMEMODE_BD = 17           #
	DOTA_GAMEMODE_ABILITY_DRAFT = 18#
	DOTA_GAMEMODE_EVENT = 19        #
	DOTA_GAMEMODE_ARDM = 20         #  All Random Death Match
	DOTA_GAMEMODE_1V1MID = 21       #
	DOTA_GAMEMODE_ALL_DRAFT = 22    #  Ranked All Pick
	DOTA_GAMEMODE_TURBO = 23        #
	DOTA_GAMEMODE_MUTATION = 24     #
