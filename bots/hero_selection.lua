local ipc = require('bots/ipc')

local RECV_MSG = 'bots/IPC_recv'
local faction = GetTeam()
local str_faction = '' .. faction
local uid = nil
local ipc_prefix = '[IPC]' .. faction .. '.' .. 'HS'

local function send_message(data)
    print(ipc_prefix, dkjson.encode(data))
end

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
                uid = new_uid
                send_message({A = uid})
                return faction_dat
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


-- Called every frame. Responsible for selecting heroes for bots.
function ThinkOverride()
    local msg = receive_message()

    if msg ~= nil then
        send_message({E = 'Not Implemented'})
    end

    if GetTeam() == TEAM_RADIANT then
        SelectHero(0, 'npc_dota_hero_antimage')
        SelectHero(1, 'npc_dota_hero_axe')
        SelectHero(2, 'npc_dota_hero_bane')
        SelectHero(3, 'npc_dota_hero_bloodseeker')
        SelectHero(4, 'npc_dota_hero_crystal_maiden')
    else
        SelectHero(5, 'npc_dota_hero_drow_ranger')
        SelectHero(6, 'npc_dota_hero_earthshaker')
        SelectHero(7, 'npc_dota_hero_juggernaut')
        SelectHero(8, 'npc_dota_hero_mirana')
        SelectHero(9, 'npc_dota_hero_nevermore')
    end
end

-- Called every frame prior to the game starting. Returns ten PlayerID-Lane pairs.
-- function UpdateLaneAssignmentsOverride()
--     return {
--         0 = LANE_TOP,
--         1 = LANE_TOP,
--         2 = LANE_MID,
--         3 = LANE_BOT,
--         4 = LANE_BOT,
--         --
--         5 = LANE_TOP,
--         6 = LANE_TOP,
--         7 = LANE_MID,
--         8 = LANE_BOT,
--         9 = LANE_BOT,
--     }
-- end

-- Called once, returns a table of player names.
function GetBotNamesOverride()
    return {
        'Blue',
        'Teal',
        'Purple',
        'Yellow',
        'Orange',
        'Pink',
        'Grey',
        'Aqua',
        'Green',
        'Brown'
    }
end

print('Hero Selection')
GetBotNames = GetBotNamesOverride
UpdateLaneAssignments = UpdateLaneAssignmentsOverride
Think = ThinkOverride
