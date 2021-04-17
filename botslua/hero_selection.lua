local dkjson = require('game/dkjson')
local pprint = require('bots/pprint')

local RECV_MSG = 'bots/IPC_recv'
local CONF_MSG = 'bots/IPC_config'

local faction = GetTeam()
local str_faction = '' .. faction
local str_player_id = 'HS'
local uid = nil
local ipc_prefix = '[IPC]' .. faction .. '.' .. 'HS'

-- this is copy pasted from bot_generic.lua
-- keep them in-sync
local function send_message(data)
    print(ipc_prefix, dkjson.encode(data))
end

local function load_config()
    local file = loadfile(CONF_MSG)

    if file ~= nil then
        local json_string = file()

        if json_string ~= nil then
            local config, pos, err = dkjson.decode(json_string, 1, nil)

            if err then
                send_message({E = tostring(err)})
                return nil
            end

            return config
        end
    else
        send_message({E = "No config file found"})
    end

    return nil
end

local config = load_config()

-- this is copy pasted from bot_genric.lua
-- keep them in-sync
local function receive_message()
    local file = loadfile(RECV_MSG)

    if file ~= nil then
        local json_string = file()
        if json_string ~= nil then
            local internal, pos, err = dkjson.decode(json_string, 1, nil)

            if err then
                send_message({E = tostring(err)})
                return nil
            end

            -- Make sure we do not execute a message twice
            local new_uid = internal['uid']

            -- First message setup the uid
            if uid == nil then
                uid = new_uid - 1
            elseif new_uid <= uid then
                -- This is expected to happen relatively often
                -- send_message({E = "Message already read " .. uid .. " " .. new_uid})
                return nil
            end

            -- Send error when we skipped a message
            if new_uid ~= uid + 1 then
                send_message({
                    E = 'Message skipped (u: ' .. uid .. ') ' .. ' (nu: ' .. new_uid .. ')'
                })
            end

            -- for efficiency we write a single file with all the information
            -- but the bot can only see its command
            local faction_dat = internal[str_faction]

            if faction_dat ~= nil then
                local player_dat = faction_dat[str_player_id]

                -- Only update the uid if were able to read the message
                uid = new_uid
                send_message({A = uid})
                return player_dat
            else
                send_message({E = "No faction found in message: " .. json_string})
            end
        else
            send_message({E = "json string not found"})
        end
    else
        send_message({E = "No file found"})
    end

    return nil
end

local nohero = 'npc_dota_hero_target_dummy'


-- Get Selected/Banned Hero state
-- We should be able to have the drafting state without querying dota
function draft_state()
    send_message({E = 'Not Implemented'})
end

-- Lane assignment
local lanes = {
    LANE_TOP,
    LANE_TOP,
    LANE_MID,
    LANE_BOT,
    LANE_BOT,
    --
    LANE_TOP,
    LANE_TOP,
    LANE_MID,
    LANE_BOT,
    LANE_BOT,
}


-- Same as SelectHero but checks for errors
local n = 0

local function select_hero(player_id, hero_name, lane)
    if GetSelectedHeroName(player_id) == '' then
        SelectHero(player_id, hero_name)

        success = 0
        if GetSelectedHeroName(player_id) ~= '' then
            success = 1
        else
            success = 0
        end

        send_message({DS = {S = success, B = hero_name, T = faction}})

        -- We picked a hero; increase offset
        lanes[player_id] = lane
        n = n + 1
    else
        send_message({DS = {S = 0, B = hero_name, T = faction, M = 'Player already picked'}})
    end
end

-- Always set first player of each time as captains
if GetGameMode() == GAMEMODE_CM or GetGameMode() == GAMEMODE_CD then
    if faction == TEAM_RADIANT then
        SetCMCaptain(0)
    else
        SetCMCaptain(5)
    end
end


local function pick_hero(player_id, hero_name, lane)
    if not IsPlayerBot(player_id) then
        lanes[player_id] = LANE_NONE
        n = n + 1
        return
    end

    if hero_name == nil then
        return
    end

    if GetGameMode() == GAMEMODE_CM or GetGameMode() == GAMEMODE_CD then
        CMPickHero(hero_name)

        success = 0
        if IsCMPickedHero(faction, hero_name) then
            success = 1
        else
            success = 0
        end

        send_message({DS = {S = success, B = hero_name, T = faction}})
    else
        select_hero(player_id, hero_name, lane)
    end
end

--
local function ban_hero(hero_name)
    if hero_name == nil then
        return
    end

    if GetGameMode() == GAMEMODE_CM or GetGameMode() == GAMEMODE_CD then
        CMBanHero(hero_name)

        success = 0
        if IsCMBannedHero(hero_name) then
            success = 1
        else
            success = 0
        end

        send_message({DS = {S = success, B = hero_name, T = faction}})
    else
        CMBanHero(hero_name)
        send_message({DS = {S = 0, B = hero_name, T = faction, M = 'Hero cannot be banned in the mode'}})
    end
end

local offset = 0
local names = {'Blue', 'Teal', 'Purple', 'Yellow', 'Orange', ' ', ' ', ' ', ' ', ' '}
if GetTeam() == TEAM_DIRE then
    offset = 5
    names = {'Pink', 'Grey', 'Aqua', 'Green', 'Brown', ' ', ' ', ' ', ' ', ' '}
end


-- In-case of a 1v1 mode being selected block the hero slot with dummies
function block_picks()
    if GetGameMode() == GAMEMODE_1V1MID or GetGameMode() == GAMEMODE_MO then
        if GetTeam() == TEAM_RADIANT then
            pick_hero(1, nohero)
            pick_hero(2, nohero)
            pick_hero(3, nohero)
            pick_hero(4, nohero)
        else
            pick_hero(6, nohero)
            pick_hero(7, nohero)
            pick_hero(8, nohero)
            pick_hero(9, nohero)
        end
        n = 0
    end
end


-- hard coded for integration testing
local function default_logic()
    -- 1v1 or mid only
    if GetGameMode() == GAMEMODE_1V1MID or GetGameMode() == GAMEMODE_MO then
        if GetTeam() == TEAM_RADIANT then
            pick_hero(0, 'npc_dota_hero_nevermore')
        else
            pick_hero(5, 'npc_dota_hero_nevermore')
        end
    else
        if GetTeam() == TEAM_RADIANT then
            pick_hero(0, 'npc_dota_hero_antimage')
            pick_hero(1, 'npc_dota_hero_axe')
            pick_hero(2, 'npc_dota_hero_bane')
            pick_hero(3, 'npc_dota_hero_bloodseeker')
            pick_hero(4, 'npc_dota_hero_invoker')
        else
            pick_hero(5, 'npc_dota_hero_drow_ranger')
            pick_hero(6, 'npc_dota_hero_earthshaker')
            pick_hero(7, 'npc_dota_hero_juggernaut')
            pick_hero(8, 'npc_dota_hero_mirana')
            pick_hero(9, 'npc_dota_hero_nevermore')
        end
    end
end


local function get_total_pick_count()
    local count = 5

    if GetGameMode() == GAMEMODE_1V1MID or GetGameMode() == GAMEMODE_MO then
        count = 1
    end

    return count
end

local total_pick_count = get_total_pick_count()

local EnableDraft = '0'
local SelectHero = '1'
local BanHero = '2'
local Lane = '3'
local start_wait_time = config["draft_start_wait"]
local pick_wait_time = config["draft_pick_wait"]
local start_time = 0
local last_pick_time = 0

local p0 = ""
local p1 = ""
local p2 = ""
local p3 = ""
local p4 = ""
local p5 = ""
local p6 = ""
local p7 = ""
local p8 = ""
local p9 = ""

-- This is just here for info
-- bans are not really easily queryable from here
-- so will track those from the python side
-- we need this function to know what the humans are picking
function get_draft_state()
    local pp0 = GetSelectedHeroName(0)
    local pp1 = GetSelectedHeroName(1)
    local pp2 = GetSelectedHeroName(2)
    local pp3 = GetSelectedHeroName(3)
    local pp4 = GetSelectedHeroName(4)
    local pp5 = GetSelectedHeroName(5)
    local pp6 = GetSelectedHeroName(6)
    local pp7 = GetSelectedHeroName(7)
    local pp8 = GetSelectedHeroName(8)
    local pp9 = GetSelectedHeroName(9)
    local state_changed = false

    if pp0 ~= p0 then
        p0 = pp0
        state_changed = true
    elseif pp1 ~= p1 then
        p1 = pp1
        state_changed = true
    elseif pp2 ~= p2 then
        p2 = pp2
        state_changed = true
    elseif pp3 ~= p3 then
        p3 = pp3
        state_changed = true
    elseif pp4 ~= p4 then
        p4 = pp4
        state_changed = true
    elseif pp5 ~= p5 then
        p5 = pp5
        state_changed = true
    elseif pp6 ~= p6 then
        p6 = pp6
        state_changed = true
    elseif pp7 ~= p7 then
        p7 = pp7
        state_changed = true
    elseif pp8 ~= p8 then
        p8 = pp8
        state_changed = true
    elseif pp9 ~= p9 then
        p9 = pp9
        state_changed = true
    end

    -- IsCMBannedHero
    if state_changed then
        send_message({
            DS = {STATE={
                p0, p1, p2, p3, p4,
                p5, p6, p7, p8, p9,
                -- Unsupported
                -- b1, b2, b3, b4, b5,
                -- b6, b7, b8, b9, b10,
                -- b11, b12, b13, b14,
                --
            }}
        })
    end
end

-- Called every frame. Responsible for selecting heroes for bots.
function ThinkOverride()
    -- first call block unavailable player slots
    if uid == nil then
        block_picks()
    end

    -- Wait for the game to load and humans to pick
    -- their heroes
    --
    -- This is necessary because DotaTime() is not 100% reliable
    -- this makes sure start_time is correct because once
    -- DotaTime() returns the right value the time only increase
    start_time = math.min(DotaTime(), start_time)
    if start_wait_time ~= nil and DotaTime() - start_time < start_wait_time then
        -- initialize the last_pick_time to a value that is aligned with the dotatime
        last_pick_time = DotaTime() - pick_wait_time
        return
    end
    --

    -- finished picking
    if n >= 5 then
        return
    end

    get_draft_state()

    -- Give time to Humans to catch up when in picking mode
    if DotaTime() - last_pick_time < pick_wait_time then
        return
    end

    local msg = receive_message()

    if msg == nil then
        return
    end

    local ml_draft = msg[EnableDraft]

    if ml_draft ~= nil and ml_draft == 0 then
         default_logic()

    elseif msg ~= nil then
        -- Bots are manually drafting
        local selected = msg[SelectHero]
        local banned = msg[BanHero]
        local lane = msg[Lane]

        local pid = offset + n

        if selected ~= nil then
            pick_hero(pid, selected, lane)
            last_pick_time = DotaTime()
        end

        if banned ~= nil then
            ban_hero(banned)
            last_pick_time = DotaTime()
        end
    end

    if n >= total_pick_count then
        send_message({DE = 'Draft Over'})
    end
end

-- Called every frame prior to the game starting. Returns ten PlayerID-Lane pairs.
function UpdateLaneAssignmentsOverride()
    return lanes
end

-- Called once, returns a table of player names.
function GetBotNamesOverride()
    return names
end


send_message({DS = 'Draft Starting'})
GetBotNames = GetBotNamesOverride
UpdateLaneAssignments = UpdateLaneAssignmentsOverride
Think = ThinkOverride
