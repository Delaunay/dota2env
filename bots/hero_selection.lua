local dkjson = require('game/dkjson')
local pprint = require('bots/pprint')

local RECV_MSG = 'bots/IPC_recv'
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
local function pick_hero(player_id, hero_name, lane)
    if hero_name == nil then
        return
    end

    if GetSelectedHeroName(player_id) == '' then

        SelectHero(player_id, hero_name)
        -- CMPickHero

        if GetSelectedHeroName(player_id) == '' then
            send_message({E = 'Could not pick hero ' .. hero_name})
            return
        end

        -- We picked a hero; increase offset
        lanes[player_id] = lane
        n = n + 1
    else
        send_message({E = 'Hero not available ' .. hero_name})
    end
end

--
local function ban_hero(hero_name)
    if hero_name == nil then
        return
    end

    -- CMBanHero(hero_name)
    send_message({E = 'Wanted to ban: ' .. hero_name})
end

local offset = 0
local names = {'Blue', 'Teal', 'Purple', 'Yellow', 'Orange'}
if GetTeam() == TEAM_DIRE then
    offset = 5
    names = {'Pink', 'Grey', 'Aqua', 'Green', 'Brown'}
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
            pick_hero(4, 'npc_dota_hero_crystal_maiden')
        else
            pick_hero(5, 'npc_dota_hero_drow_ranger')
            pick_hero(6, 'npc_dota_hero_earthshaker')
            pick_hero(7, 'npc_dota_hero_juggernaut')
            pick_hero(8, 'npc_dota_hero_mirana')
            pick_hero(9, 'npc_dota_hero_nevermore')
        end
    end
end


-- Called every frame. Responsible for selecting heroes for bots.
function ThinkOverride()
    -- first call block unavailable player slots
    if uid == nil then
        block_picks()
    end

    -- finished picking
    if n >= 5 then
        return
    end

    local msg = receive_message()

    if msg == nil then
        return
    end

    local ml_draft = msg[1]

    if ml_draft ~= nil and ml_draft == 0 then
         default_logic()
    elseif msg ~= nil then
        -- Bots are manually drafting
        local selected = msg[2]
        local banned = msg[3]
        local lane = msg[4]

        local pid = offset + n
        pick_hero(pid, selected, lane)
        ban_hero(banned)
    end

    if n >= 5 then
        send_message({P = 'Draft Over'})
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


print('Hero Selection')

GetBotNames = GetBotNamesOverride
UpdateLaneAssignments = UpdateLaneAssignmentsOverride
Think = ThinkOverride
