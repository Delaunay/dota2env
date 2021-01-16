from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List

class fixed64:
    pass


uint64 = int

# .DOTA_GameMode
class DOTA_GameMode(IntEnum):
    DOTA_GAMEMODE_NONE = 0
    DOTA_GAMEMODE_AP = 1
    DOTA_GAMEMODE_CM = 2
    DOTA_GAMEMODE_RD = 3
    DOTA_GAMEMODE_SD = 4
    DOTA_GAMEMODE_AR = 5
    DOTA_GAMEMODE_INTRO = 6
    DOTA_GAMEMODE_HW = 7
    DOTA_GAMEMODE_REVERSE_CM = 8
    DOTA_GAMEMODE_XMAS = 9
    DOTA_GAMEMODE_TUTORIAL = 10
    DOTA_GAMEMODE_MO = 11
    DOTA_GAMEMODE_LP = 12
    DOTA_GAMEMODE_POOL1 = 13
    DOTA_GAMEMODE_FH = 14
    DOTA_GAMEMODE_CUSTOM = 15
    DOTA_GAMEMODE_CD = 16
    DOTA_GAMEMODE_BD = 17
    DOTA_GAMEMODE_ABILITY_DRAFT = 18
    DOTA_GAMEMODE_EVENT = 19
    DOTA_GAMEMODE_ARDM = 20
    DOTA_GAMEMODE_1V1MID = 21
    DOTA_GAMEMODE_ALL_DRAFT = 22
    DOTA_GAMEMODE_TURBO = 23
    DOTA_GAMEMODE_MUTATION = 24

# .DOTA_GameState
class DOTA_GameState(IntEnum):
    DOTA_GAMERULES_STATE_INIT = 0
    DOTA_GAMERULES_STATE_WAIT_FOR_PLAYERS_TO_LOAD = 1
    DOTA_GAMERULES_STATE_HERO_SELECTION = 2
    DOTA_GAMERULES_STATE_STRATEGY_TIME = 3
    DOTA_GAMERULES_STATE_PRE_GAME = 4
    DOTA_GAMERULES_STATE_GAME_IN_PROGRESS = 5
    DOTA_GAMERULES_STATE_POST_GAME = 6
    DOTA_GAMERULES_STATE_DISCONNECT = 7
    DOTA_GAMERULES_STATE_TEAM_SHOWCASE = 8
    DOTA_GAMERULES_STATE_CUSTOM_GAME_SETUP = 9
    DOTA_GAMERULES_STATE_WAIT_FOR_MAP_TO_LOAD = 10
    DOTA_GAMERULES_STATE_LAST = 11

# .DOTA_GC_TEAM
class DOTA_GC_TEAM(IntEnum):
    DOTA_GC_TEAM_GOOD_GUYS = 0
    DOTA_GC_TEAM_BAD_GUYS = 1
    DOTA_GC_TEAM_BROADCASTER = 2
    DOTA_GC_TEAM_SPECTATOR = 3
    DOTA_GC_TEAM_PLAYER_POOL = 4
    DOTA_GC_TEAM_NOTEAM = 5

# .EEvent
class EEvent(IntEnum):
    EVENT_ID_NONE = 0
    EVENT_ID_DIRETIDE = 1
    EVENT_ID_SPRING_FESTIVAL = 2
    EVENT_ID_FROSTIVUS_2013 = 3
    EVENT_ID_COMPENDIUM_2014 = 4
    EVENT_ID_NEXON_PC_BANG = 5
    EVENT_ID_PWRD_DAC_2015 = 6
    EVENT_ID_NEW_BLOOM_2015 = 7
    EVENT_ID_INTERNATIONAL_2015 = 8
    EVENT_ID_FALL_MAJOR_2015 = 9
    EVENT_ID_ORACLE_PA = 10
    EVENT_ID_NEW_BLOOM_2015_PREBEAST = 11
    EVENT_ID_FROSTIVUS = 12
    EVENT_ID_WINTER_MAJOR_2016 = 13
    EVENT_ID_INTERNATIONAL_2016 = 14
    EVENT_ID_FALL_MAJOR_2016 = 15
    EVENT_ID_WINTER_MAJOR_2017 = 16
    EVENT_ID_NEW_BLOOM_2017 = 17
    EVENT_ID_INTERNATIONAL_2017 = 18
    EVENT_ID_PLUS_SUBSCRIPTION = 19
    EVENT_ID_SINGLES_DAY_2017 = 20
    EVENT_ID_FROSTIVUS_2017 = 21
    EVENT_ID_INTERNATIONAL_2018 = 22
    EVENT_ID_COUNT = 23

# .DOTALeaverStatus_t
class DOTALeaverStatus_t(IntEnum):
    DOTA_LEAVER_NONE = 0
    DOTA_LEAVER_DISCONNECTED = 1
    DOTA_LEAVER_DISCONNECTED_TOO_LONG = 2
    DOTA_LEAVER_ABANDONED = 3
    DOTA_LEAVER_AFK = 4
    DOTA_LEAVER_NEVER_CONNECTED = 5
    DOTA_LEAVER_NEVER_CONNECTED_TOO_LONG = 6
    DOTA_LEAVER_FAILED_TO_READY_UP = 7
    DOTA_LEAVER_DECLINED = 8

# .DOTAConnectionState_t
class DOTAConnectionState_t(IntEnum):
    DOTA_CONNECTION_STATE_UNKNOWN = 0
    DOTA_CONNECTION_STATE_NOT_YET_CONNECTED = 1
    DOTA_CONNECTION_STATE_CONNECTED = 2
    DOTA_CONNECTION_STATE_DISCONNECTED = 3
    DOTA_CONNECTION_STATE_ABANDONED = 4
    DOTA_CONNECTION_STATE_LOADING = 5
    DOTA_CONNECTION_STATE_FAILED = 6

# .Fantasy_Roles
class Fantasy_Roles(IntEnum):
    FANTASY_ROLE_UNDEFINED = 0
    FANTASY_ROLE_CORE = 1
    FANTASY_ROLE_SUPPORT = 2
    FANTASY_ROLE_OFFLANE = 3

# .Fantasy_Team_Slots
class Fantasy_Team_Slots(IntEnum):
    FANTASY_SLOT_NONE = 0
    FANTASY_SLOT_CORE = 1
    FANTASY_SLOT_SUPPORT = 2
    FANTASY_SLOT_ANY = 3
    FANTASY_SLOT_BENCH = 4

# .Fantasy_Selection_Mode
class Fantasy_Selection_Mode(IntEnum):
    FANTASY_SELECTION_INVALID = 0
    FANTASY_SELECTION_LOCKED = 1
    FANTASY_SELECTION_SHUFFLE = 2
    FANTASY_SELECTION_FREE_PICK = 3
    FANTASY_SELECTION_ENDED = 4
    FANTASY_SELECTION_PRE_SEASON = 5
    FANTASY_SELECTION_PRE_DRAFT = 6
    FANTASY_SELECTION_DRAFTING = 7
    FANTASY_SELECTION_REGULAR_SEASON = 8
    FANTASY_SELECTION_CARD_BASED = 9

# .DOTAChatChannelType_t
class DOTAChatChannelType_t(IntEnum):
    DOTAChannelType_Regional = 0
    DOTAChannelType_Custom = 1
    DOTAChannelType_Party = 2
    DOTAChannelType_Lobby = 3
    DOTAChannelType_Team = 4
    DOTAChannelType_Guild = 5
    DOTAChannelType_Fantasy = 6
    DOTAChannelType_Whisper = 7
    DOTAChannelType_Console = 8
    DOTAChannelType_Tab = 9
    DOTAChannelType_Invalid = 10
    DOTAChannelType_GameAll = 11
    DOTAChannelType_GameAllies = 12
    DOTAChannelType_GameSpectator = 13
    DOTAChannelType_Cafe = 15
    DOTAChannelType_CustomGame = 16
    DOTAChannelType_Private = 17
    DOTAChannelType_PostGame = 18
    DOTAChannelType_BattleCup = 19
    DOTAChannelType_HLTVSpectator = 20
    DOTAChannelType_GameEvents = 21
    DOTAChannelType_Trivia = 22

# .EProfileCardSlotType
class EProfileCardSlotType(IntEnum):
    k_EProfileCardSlotType_Empty = 0
    k_EProfileCardSlotType_Stat = 1
    k_EProfileCardSlotType_Trophy = 2
    k_EProfileCardSlotType_Item = 3
    k_EProfileCardSlotType_Hero = 4
    k_EProfileCardSlotType_Emoticon = 5
    k_EProfileCardSlotType_Team = 6

# .EMatchGroupServerStatus
class EMatchGroupServerStatus(IntEnum):
    k_EMatchGroupServerStatus_OK = 0
    k_EMatchGroupServerStatus_LimitedAvailability = 1
    k_EMatchGroupServerStatus_Offline = 2

# .DOTA_CM_PICK
class DOTA_CM_PICK(IntEnum):
    DOTA_CM_RANDOM = 0
    DOTA_CM_GOOD_GUYS = 1
    DOTA_CM_BAD_GUYS = 2

# .DOTALowPriorityBanType
class DOTALowPriorityBanType(IntEnum):
    DOTA_LOW_PRIORITY_BAN_ABANDON = 0
    DOTA_LOW_PRIORITY_BAN_REPORTS = 1
    DOTA_LOW_PRIORITY_BAN_SECONDARY_ABANDON = 2

# .DOTALobbyReadyState
class DOTALobbyReadyState(IntEnum):
    DOTALobbyReadyState_UNDECLARED = 0
    DOTALobbyReadyState_ACCEPTED = 1
    DOTALobbyReadyState_DECLINED = 2

# .DOTAGameVersion
class DOTAGameVersion(IntEnum):
    GAME_VERSION_CURRENT = 0
    GAME_VERSION_STABLE = 1

# .DOTAJoinLobbyResult
class DOTAJoinLobbyResult(IntEnum):
    DOTA_JOIN_RESULT_SUCCESS = 0
    DOTA_JOIN_RESULT_ALREADY_IN_GAME = 1
    DOTA_JOIN_RESULT_INVALID_LOBBY = 2
    DOTA_JOIN_RESULT_INCORRECT_PASSWORD = 3
    DOTA_JOIN_RESULT_ACCESS_DENIED = 4
    DOTA_JOIN_RESULT_GENERIC_ERROR = 5
    DOTA_JOIN_RESULT_INCORRECT_VERSION = 6
    DOTA_JOIN_RESULT_IN_TEAM_PARTY = 7
    DOTA_JOIN_RESULT_NO_LOBBY_FOUND = 8
    DOTA_JOIN_RESULT_LOBBY_FULL = 9
    DOTA_JOIN_RESULT_CUSTOM_GAME_INCORRECT_VERSION = 10
    DOTA_JOIN_RESULT_TIMEOUT = 11
    DOTA_JOIN_RESULT_CUSTOM_GAME_COOLDOWN = 12

# .DOTASelectionPriorityRules
class DOTASelectionPriorityRules(IntEnum):
    k_DOTASelectionPriorityRules_Manual = 0
    k_DOTASelectionPriorityRules_Automatic = 1

# .DOTASelectionPriorityChoice
class DOTASelectionPriorityChoice(IntEnum):
    k_DOTASelectionPriorityChoice_Invalid = 0
    k_DOTASelectionPriorityChoice_FirstPick = 1
    k_DOTASelectionPriorityChoice_SecondPick = 2
    k_DOTASelectionPriorityChoice_Radiant = 3
    k_DOTASelectionPriorityChoice_Dire = 4

# .DOTAMatchVote
class DOTAMatchVote(IntEnum):
    DOTAMatchVote_INVALID = 0
    DOTAMatchVote_POSITIVE = 1
    DOTAMatchVote_NEGATIVE = 2

# .DOTA_LobbyMemberXPBonus
class DOTA_LobbyMemberXPBonus(IntEnum):
    DOTA_LobbyMemberXPBonus_DEFAULT = 0
    DOTA_LobbyMemberXPBonus_BATTLE_BOOSTER = 1
    DOTA_LobbyMemberXPBonus_SHARE_BONUS = 2
    DOTA_LobbyMemberXPBonus_PARTY = 3
    DOTA_LobbyMemberXPBonus_RECRUITMENT = 4
    DOTA_LobbyMemberXPBonus_PCBANG = 5

# .DOTALobbyVisibility
class DOTALobbyVisibility(IntEnum):
    DOTALobbyVisibility_Public = 0
    DOTALobbyVisibility_Friends = 1
    DOTALobbyVisibility_Unlisted = 2

# .EDOTAPlayerMMRType
class EDOTAPlayerMMRType(IntEnum):
    k_EDOTAPlayerMMRType_Invalid = 0
    k_EDOTAPlayerMMRType_GeneralHidden = 1
    k_EDOTAPlayerMMRType_GeneralCompetitive = 3
    k_EDOTAPlayerMMRType_SoloCompetitive = 4
    k_EDOTAPlayerMMRType_1v1Competitive_UNUSED = 5
    k_EDOTAPlayerMMRType_GeneralSeasonalRanked = 6
    k_EDOTAPlayerMMRType_SoloSeasonalRanked = 7

# .MatchType
class MatchType(IntEnum):
    MATCH_TYPE_CASUAL = 0
    MATCH_TYPE_COOP_BOTS = 1
    MATCH_TYPE_TEAM_RANKED = 2
    MATCH_TYPE_LEGACY_SOLO_QUEUE = 3
    MATCH_TYPE_COMPETITIVE = 4
    MATCH_TYPE_WEEKEND_TOURNEY = 5
    MATCH_TYPE_CASUAL_1V1 = 6
    MATCH_TYPE_EVENT = 7
    MATCH_TYPE_SEASONAL_RANKED = 8
    MATCH_TYPE_LOWPRI_DEPRECATED = 9
    MATCH_TYPE_STEAM_GROUP = 10
    MATCH_TYPE_MUTATION = 11

# .DOTABotDifficulty
class DOTABotDifficulty(IntEnum):
    BOT_DIFFICULTY_PASSIVE = 0
    BOT_DIFFICULTY_EASY = 1
    BOT_DIFFICULTY_MEDIUM = 2
    BOT_DIFFICULTY_HARD = 3
    BOT_DIFFICULTY_UNFAIR = 4
    BOT_DIFFICULTY_INVALID = 5
    BOT_DIFFICULTY_EXTRA1 = 6
    BOT_DIFFICULTY_EXTRA2 = 7
    BOT_DIFFICULTY_EXTRA3 = 8

# .DOTA_BOT_MODE
class DOTA_BOT_MODE(IntEnum):
    DOTA_BOT_MODE_NONE = 0
    DOTA_BOT_MODE_LANING = 1
    DOTA_BOT_MODE_ATTACK = 2
    DOTA_BOT_MODE_ROAM = 3
    DOTA_BOT_MODE_RETREAT = 4
    DOTA_BOT_MODE_SECRET_SHOP = 5
    DOTA_BOT_MODE_SIDE_SHOP = 6
    DOTA_BOT_MODE_RUNE = 7
    DOTA_BOT_MODE_PUSH_TOWER_TOP = 8
    DOTA_BOT_MODE_PUSH_TOWER_MID = 9
    DOTA_BOT_MODE_PUSH_TOWER_BOT = 10
    DOTA_BOT_MODE_DEFEND_TOWER_TOP = 11
    DOTA_BOT_MODE_DEFEND_TOWER_MID = 12
    DOTA_BOT_MODE_DEFEND_TOWER_BOT = 13
    DOTA_BOT_MODE_ASSEMBLE = 14
    DOTA_BOT_MODE_ASSEMBLE_WITH_HUMANS = 15
    DOTA_BOT_MODE_TEAM_ROAM = 16
    DOTA_BOT_MODE_FARM = 17
    DOTA_BOT_MODE_DEFEND_ALLY = 18
    DOTA_BOT_MODE_EVASIVE_MANEUVERS = 19
    DOTA_BOT_MODE_ROSHAN = 20
    DOTA_BOT_MODE_ITEM = 21
    DOTA_BOT_MODE_WARD = 22
    DOTA_BOT_MODE_COMPANION = 23
    DOTA_BOT_MODE_TUTORIAL_BOSS = 24
    DOTA_BOT_MODE_MINION = 25

# .MatchLanguages
class MatchLanguages(IntEnum):
    MATCH_LANGUAGE_INVALID = 0
    MATCH_LANGUAGE_ENGLISH = 1
    MATCH_LANGUAGE_RUSSIAN = 2
    MATCH_LANGUAGE_CHINESE = 3
    MATCH_LANGUAGE_KOREAN = 4
    MATCH_LANGUAGE_SPANISH = 5
    MATCH_LANGUAGE_PORTUGUESE = 6
    MATCH_LANGUAGE_ENGLISH2 = 7

# .ETourneyQueueDeadlineState
class ETourneyQueueDeadlineState(IntEnum):
    k_ETourneyQueueDeadlineState_Normal = 0
    k_ETourneyQueueDeadlineState_Missed = 1
    k_ETourneyQueueDeadlineState_ExpiredOK = 2
    k_ETourneyQueueDeadlineState_SeekingBye = 3
    k_ETourneyQueueDeadlineState_EligibleForRefund = 4
    k_ETourneyQueueDeadlineState_NA = -1
    k_ETourneyQueueDeadlineState_ExpiringSoon = 101

# .EMatchOutcome
class EMatchOutcome(IntEnum):
    k_EMatchOutcome_Unknown = 0
    k_EMatchOutcome_RadVictory = 2
    k_EMatchOutcome_DireVictory = 3
    k_EMatchOutcome_NotScored_PoorNetworkConditions = 64
    k_EMatchOutcome_NotScored_Leaver = 65
    k_EMatchOutcome_NotScored_ServerCrash = 66
    k_EMatchOutcome_NotScored_NeverStarted = 67
    k_EMatchOutcome_NotScored_Canceled = 68

# .ELaneType
class ELaneType(IntEnum):
    LANE_TYPE_UNKNOWN = 0
    LANE_TYPE_SAFE = 1
    LANE_TYPE_OFF = 2
    LANE_TYPE_MID = 3
    LANE_TYPE_JUNGLE = 4
    LANE_TYPE_ROAM = 5

# .EBadgeType
class EBadgeType(IntEnum):
    k_EBadgeType_TI7_Midweek = 1
    k_EBadgeType_TI7_Finals = 2
    k_EBadgeType_TI7_AllEvent = 3
    k_EBadgeType_TI8_Midweek = 4
    k_EBadgeType_TI8_Finals = 5
    k_EBadgeType_TI8_AllEvent = 6

# .ELeagueStatus
class ELeagueStatus(IntEnum):
    LEAGUE_STATUS_UNSET = 0
    LEAGUE_STATUS_UNSUBMITTED = 1
    LEAGUE_STATUS_SUBMITTED = 2
    LEAGUE_STATUS_ACCEPTED = 3
    LEAGUE_STATUS_REJECTED = 4
    LEAGUE_STATUS_CONCLUDED = 5
    LEAGUE_STATUS_DELETED = 6

# .ELeagueRegion
class ELeagueRegion(IntEnum):
    LEAGUE_REGION_UNSET = 0
    LEAGUE_REGION_NA = 1
    LEAGUE_REGION_SA = 2
    LEAGUE_REGION_EUROPE = 3
    LEAGUE_REGION_CIS = 4
    LEAGUE_REGION_CHINA = 5
    LEAGUE_REGION_SEA = 6

# .ELeagueTier
class ELeagueTier(IntEnum):
    LEAGUE_TIER_UNSET = 0
    LEAGUE_TIER_AMATEUR = 1
    LEAGUE_TIER_PROFESSIONAL = 2
    LEAGUE_TIER_MINOR = 3
    LEAGUE_TIER_MAJOR = 4
    LEAGUE_TIER_INTERNATIONAL = 5

# .ELeagueTierCategory
class ELeagueTierCategory(IntEnum):
    LEAGUE_TIER_CATEGORY_AMATEUR = 1
    LEAGUE_TIER_CATEGORY_PROFESSIONAL = 2
    LEAGUE_TIER_CATEGORY_DPC = 3

# .ELeagueFlags
class ELeagueFlags(IntEnum):
    LEAGUE_FLAGS_NONE = 0
    LEAGUE_ACCEPTED_AGREEMENT = 1
    LEAGUE_PAYMENT_EMAIL_SENT = 2
    LEAGUE_COMPENDIUM_ALLOWED = 4
    LEAGUE_COMPENDIUM_PUBLIC = 8

# .ELeagueBroadcastProvider
class ELeagueBroadcastProvider(IntEnum):
    LEAGUE_BROADCAST_UNKNOWN = 0
    LEAGUE_BROADCAST_STEAM = 1
    LEAGUE_BROADCAST_TWITCH = 2
    LEAGUE_BROADCAST_YOUTUBE = 3
    LEAGUE_BROADCAST_OTHER = 100

# .ELeaguePhase
class ELeaguePhase(IntEnum):
    LEAGUE_PHASE_UNSET = 0
    LEAGUE_PHASE_REGIONAL_QUALIFIER = 1
    LEAGUE_PHASE_GROUP_STAGE = 2
    LEAGUE_PHASE_MAIN_EVENT = 3

# .ELeagueFantasyPhase
class ELeagueFantasyPhase(IntEnum):
    LEAGUE_FANTASY_PHASE_UNSET = 0
    LEAGUE_FANTASY_PHASE_MAIN = 1
    LEAGUE_FANTASY_PHASE_QUALIFIER_NA = 2
    LEAGUE_FANTASY_PHASE_QUALIFIER_SA = 3
    LEAGUE_FANTASY_PHASE_QUALIFIER_EUROPE = 4
    LEAGUE_FANTASY_PHASE_QUALIFIER_CIS = 5
    LEAGUE_FANTASY_PHASE_QUALIFIER_CHINA = 6
    LEAGUE_FANTASY_PHASE_QUALIFIER_SEA = 7

# .ELeagueAuditAction
class ELeagueAuditAction(IntEnum):
    LEAGUE_AUDIT_ACTION_INVALID = 0
    LEAGUE_AUDIT_ACTION_LEAGUE_CREATE = 1
    LEAGUE_AUDIT_ACTION_LEAGUE_EDIT = 2
    LEAGUE_AUDIT_ACTION_LEAGUE_DELETE = 3
    LEAGUE_AUDIT_ACTION_LEAGUE_ADMIN_ADD = 4
    LEAGUE_AUDIT_ACTION_LEAGUE_ADMIN_REVOKE = 5
    LEAGUE_AUDIT_ACTION_LEAGUE_ADMIN_PROMOTE = 6
    LEAGUE_AUDIT_ACTION_LEAGUE_STREAM_ADD = 7
    LEAGUE_AUDIT_ACTION_LEAGUE_STREAM_REMOVE = 8
    LEAGUE_AUDIT_ACTION_LEAGUE_IMAGE_UPDATED = 9
    LEAGUE_AUDIT_ACTION_LEAGUE_MESSAGE_ADDED = 10
    LEAGUE_AUDIT_ACTION_LEAGUE_SUBMITTED = 11
    LEAGUE_AUDIT_ACTION_LEAGUE_SET_PRIZE_POOL = 12
    LEAGUE_AUDIT_ACTION_LEAGUE_ADD_PRIZE_POOL_ITEM = 13
    LEAGUE_AUDIT_ACTION_LEAGUE_REMOVE_PRIZE_POOL_ITEM = 14
    LEAGUE_AUDIT_ACTION_LEAGUE_MATCH_START = 15
    LEAGUE_AUDIT_ACTION_LEAGUE_MATCH_END = 16
    LEAGUE_AUDIT_ACTION_LEAGUE_ADD_INVITED_TEAM = 17
    LEAGUE_AUDIT_ACTION_LEAGUE_REMOVE_INVITED_TEAM = 18
    LEAGUE_AUDIT_ACTION_LEAGUE_STATUS_CHANGED = 19
    LEAGUE_AUDIT_ACTION_LEAGUE_STREAM_EDIT = 20
    LEAGUE_AUDIT_ACTION_NODEGROUP_CREATE = 100
    LEAGUE_AUDIT_ACTION_NODEGROUP_DESTROY = 101
    LEAGUE_AUDIT_ACTION_NODEGROUP_ADD_TEAM = 102
    LEAGUE_AUDIT_ACTION_NODEGROUP_REMOVE_TEAM = 103
    LEAGUE_AUDIT_ACTION_NODEGROUP_SET_ADVANCING = 104
    LEAGUE_AUDIT_ACTION_NODEGROUP_EDIT = 105
    LEAGUE_AUDIT_ACTION_NODEGROUP_POPULATE = 106
    LEAGUE_AUDIT_ACTION_NODEGROUP_COMPLETED = 107
    LEAGUE_AUDIT_ACTION_NODE_CREATE = 200
    LEAGUE_AUDIT_ACTION_NODE_DESTROY = 201
    LEAGUE_AUDIT_ACTION_NODE_AUTOCREATE = 202
    LEAGUE_AUDIT_ACTION_NODE_SET_TEAM = 203
    LEAGUE_AUDIT_ACTION_NODE_SET_SERIES_ID = 204
    LEAGUE_AUDIT_ACTION_NODE_SET_ADVANCING = 205
    LEAGUE_AUDIT_ACTION_NODE_SET_TIME = 206
    LEAGUE_AUDIT_ACTION_NODE_MATCH_COMPLETED = 207
    LEAGUE_AUDIT_ACTION_NODE_COMPLETED = 208
    LEAGUE_AUDIT_ACTION_NODE_EDIT = 209

# .DOTA_COMBATLOG_TYPES
class DOTA_COMBATLOG_TYPES(IntEnum):
    DOTA_COMBATLOG_INVALID = -1
    DOTA_COMBATLOG_DAMAGE = 0
    DOTA_COMBATLOG_HEAL = 1
    DOTA_COMBATLOG_MODIFIER_ADD = 2
    DOTA_COMBATLOG_MODIFIER_REMOVE = 3
    DOTA_COMBATLOG_DEATH = 4
    DOTA_COMBATLOG_ABILITY = 5
    DOTA_COMBATLOG_ITEM = 6
    DOTA_COMBATLOG_LOCATION = 7
    DOTA_COMBATLOG_GOLD = 8
    DOTA_COMBATLOG_GAME_STATE = 9
    DOTA_COMBATLOG_XP = 10
    DOTA_COMBATLOG_PURCHASE = 11
    DOTA_COMBATLOG_BUYBACK = 12
    DOTA_COMBATLOG_ABILITY_TRIGGER = 13
    DOTA_COMBATLOG_PLAYERSTATS = 14
    DOTA_COMBATLOG_MULTIKILL = 15
    DOTA_COMBATLOG_KILLSTREAK = 16
    DOTA_COMBATLOG_TEAM_BUILDING_KILL = 17
    DOTA_COMBATLOG_FIRST_BLOOD = 18
    DOTA_COMBATLOG_MODIFIER_STACK_EVENT = 19
    DOTA_COMBATLOG_NEUTRAL_CAMP_STACK = 20
    DOTA_COMBATLOG_PICKUP_RUNE = 21
    DOTA_COMBATLOG_REVEALED_INVISIBLE = 22
    DOTA_COMBATLOG_HERO_SAVED = 23
    DOTA_COMBATLOG_MANA_RESTORED = 24
    DOTA_COMBATLOG_HERO_LEVELUP = 25
    DOTA_COMBATLOG_BOTTLE_HEAL_ALLY = 26
    DOTA_COMBATLOG_ENDGAME_STATS = 27
    DOTA_COMBATLOG_INTERRUPT_CHANNEL = 28
    DOTA_COMBATLOG_ALLIED_GOLD = 29
    DOTA_COMBATLOG_AEGIS_TAKEN = 30
    DOTA_COMBATLOG_MANA_DAMAGE = 31
    DOTA_COMBATLOG_PHYSICAL_DAMAGE_PREVENTED = 32
    DOTA_COMBATLOG_UNIT_SUMMONED = 33
    DOTA_COMBATLOG_ATTACK_EVADE = 34
    DOTA_COMBATLOG_TREE_CUT = 35
    DOTA_COMBATLOG_SUCCESSFUL_SCAN = 36
    DOTA_COMBATLOG_END_KILLSTREAK = 37
    DOTA_COMBATLOG_BLOODSTONE_CHARGE = 38
    DOTA_COMBATLOG_CRITICAL_DAMAGE = 39
    DOTA_COMBATLOG_SPELL_ABSORB = 40
    DOTA_COMBATLOG_UNIT_TELEPORTED = 41
    DOTA_COMBATLOG_KILL_EATER_EVENT = 42

# .EDPCFavoriteType
class EDPCFavoriteType(IntEnum):
    FAVORITE_TYPE_ALL = 0
    FAVORITE_TYPE_PLAYER = 1
    FAVORITE_TYPE_TEAM = 2
    FAVORITE_TYPE_LEAGUE = 3

# .CDOTAClientHardwareSpecs
@dataclass
class CDOTAClientHardwareSpecs:
    logical_processors: Optional[int]
    cpu_cycles_per_second: Optional[fixed64]
    total_physical_memory: Optional[fixed64]
    is_64_bit_os: Optional[bool]
    upload_measurement: Optional[uint64]
    prefer_not_host: Optional[bool]

# .CDOTASaveGame.Player
@dataclass
class Player:
    team: Optional[DOTA_GC_TEAM]
    name: Optional[str]
    hero: Optional[str]

# .CDOTASaveGame.SaveInstance.PlayerPositions
@dataclass
class PlayerPositions:
    x: Optional[float]
    y: Optional[float]

# .CDOTASaveGame.SaveInstance
@dataclass
class SaveInstance:
    game_time: Optional[int]
    team1_score: Optional[int]
    team2_score: Optional[int]
    player_positions: List[PlayerPositions]
    save_id: Optional[int]
    save_time: Optional[int]

# .CDOTASaveGame
@dataclass
class CDOTASaveGame:
    match_id: Optional[uint64]
    save_time: Optional[int]
    players: List[Player]
    save_instances: List[SaveInstance]

# .CMsgDOTACombatLogEntry
@dataclass
class CMsgDOTACombatLogEntry:
    type: Optional[DOTA_COMBATLOG_TYPES]
    target_name: Optional[int]
    target_source_name: Optional[int]
    attacker_name: Optional[int]
    damage_source_name: Optional[int]
    inflictor_name: Optional[int]
    is_attacker_illusion: Optional[bool]
    is_attacker_hero: Optional[bool]
    is_target_illusion: Optional[bool]
    is_target_hero: Optional[bool]
    is_visible_radiant: Optional[bool]
    is_visible_dire: Optional[bool]
    value: Optional[int]
    health: Optional[int]
    timestamp: Optional[float]
    stun_duration: Optional[float]
    slow_duration: Optional[float]
    is_ability_toggle_on: Optional[bool]
    is_ability_toggle_off: Optional[bool]
    ability_level: Optional[int]
    location_x: Optional[float]
    location_y: Optional[float]
    gold_reason: Optional[int]
    timestamp_raw: Optional[float]
    modifier_duration: Optional[float]
    xp_reason: Optional[int]
    last_hits: Optional[int]
    attacker_team: Optional[int]
    target_team: Optional[int]
    obs_wards_placed: Optional[int]
    assist_player0: Optional[int]
    assist_player1: Optional[int]
    assist_player2: Optional[int]
    assist_player3: Optional[int]
    stack_count: Optional[int]
    hidden_modifier: Optional[bool]
    is_target_building: Optional[bool]
    neutral_camp_type: Optional[int]
    rune_type: Optional[int]
    assist_players: List[int]
    is_heal_save: Optional[bool]
    is_ultimate_ability: Optional[bool]
    attacker_hero_level: Optional[int]
    target_hero_level: Optional[int]
    xpm: Optional[int]
    gpm: Optional[int]
    event_location: Optional[int]
    target_is_self: Optional[bool]
    damage_type: Optional[int]
    invisibility_modifier: Optional[bool]
    damage_category: Optional[int]
    networth: Optional[int]
    building_type: Optional[int]
    modifier_elapsed_duration: Optional[float]
    silence_modifier: Optional[bool]
    heal_from_lifesteal: Optional[bool]
    modifier_purged: Optional[bool]
    spell_evaded: Optional[bool]
    motion_controller_modifier: Optional[bool]
    long_range_kill: Optional[bool]
    modifier_purge_ability: Optional[int]
    modifier_purge_npc: Optional[int]
    root_modifier: Optional[bool]
    total_unit_death_count: Optional[int]
    aura_modifier: Optional[bool]
    armor_debuff_modifier: Optional[bool]
    no_physical_damage_modifier: Optional[bool]
    modifier_ability: Optional[int]
    modifier_hidden: Optional[bool]
    inflictor_is_stolen_ability: Optional[bool]
    kill_eater_event: Optional[int]
    unit_status_label: Optional[int]
