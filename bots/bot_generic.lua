local dkjson = require('game/dkjson')
local pprint = require('bots/pprint')

local RECV_MSG = 'bots/IPC_recv'

local player_id = GetBot():GetPlayerID()
local faction = GetBot():GetTeam()

local str_faction = '' .. faction
local str_player_id = '' .. player_id

local uid = 0
local ipc_prefix = '[IPC]' .. faction .. '.' .. player_id

-- Simply print something we can parse in the logs
--  A: acknowledge
--  S: status        R: Ready
--  E: error
--
-- About Efficiency: that log file could become fairly large since we never truncate it
-- Given that dota has control over it is unclear what could be done
-- on a plus side we do jump at the end to read the last lines only
-- but it is unclear is that file will be in RAM most of the time
-- Anyway bots only execute commands sent to them they should not have much to send back
local function send_message(data)
    print(ipc_prefix, dkjson.encode(data))
end

-- Because we are inside a lua VM with limited capabilities
-- We use lua file to receive messages included in the source file
-- Others approach are not doable because
-- Lua 5.1 is embedded into Source 2, we have no library to link against
-- we could try to compile lua 5.1 as a shared library and compile our C package
-- and we might be able to load tcp socket lib but it is assuming require wasnt stripped
--
-- About efficiency: we only update a single file that is going to be fairly small and
-- read 10 times before being overriden, the file should always be cached in RAM
-- so this method in terms of speed should be fine
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
            if new_uid <= uid then
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

-- Action Enum
-- Keep in sync with action.py
local AMoveToLocation                = 0   -- ( vLocation )
local AMoveDirectly                  = 1   -- ( vLocation )
local AMoveToUnit                    = 2   -- ( hUnit )
local AAttackUnit                    = 3   -- ( hUnit, bOnce = True )
local AAttackMove                    = 4   -- ( vLocation )
local AUseAbility                    = 5   -- ( hAbility )
local AUseAbilityOnEntity            = 6   -- ( hAbility, hTarget )
local AUseAbilityOnLocation          = 7   -- ( hAbility, vLocation )
local AUseAbilityOnTree              = 8   -- ( hAbility, iTree )
local APickUpRune                    = 9   -- ( nRune )
local APickUpItem                    = 10  -- ( hItem )
local ADropItem                      = 11  -- ( hItem, vLocation )
local ADelay                         = 12  -- ( fDelay )
local APurchaseItem                  = 13  -- ( sItemName )
local ASellItem                      = 14  -- ( hItem )
local ADisassembleItem               = 15  -- ( hItem )
local ASetItemCombineLock            = 16  -- ( hItem, bLocked )
local ASwapItems                     = 17  -- ( index1, index2 )
local ABuyback                       = 18  -- ()
local AGlyph                         = 19  -- ()
local ALevelAbility                  = 20  -- ( sAbilityName )
local ATakeOutpost                   = 21  -- ()
local ACourierBurst                  = 22
local ACourierEnemySecret            = 23
local ACourierReturn                 = 24
local ACourierSecret                 = 25
local ACourierTakeStash              = 26
local ACourierTransfert              = 27
local NotUsed1 = 28
local NotUsed2 = 29
local NotUsed3 = 30
local NotUsed4 = 31

-- TODO check how to get the bot courier
local hCourier = GetCourier(0)
local bot = GetBot()

-- Map the action ID to its function
-- This is all the actions the bots can make
local function get_action_table()
    local actionHandler = {}
    actionHandler[AMoveToLocation]       = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_MoveToLocation(Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[AMoveDirectly]         = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_MoveDirectly(Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[AMoveToUnit]           = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_MoveToUnit(Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[AAttackUnit]           = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_AttackUnit(Vector(vLoc[1], vLoc[2], vLoc[3]), true) end
    actionHandler[AAttackMove]           = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_AttackMove(Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[AUseAbility]           = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_UseAbility(Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[AUseAbilityOnEntity]   = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_UseAbilityOnEntity(hAbility, hTarget) end
    actionHandler[AUseAbilityOnLocation] = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_UseAbilityOnLocation(hAbility, Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[AUseAbilityOnTree]     = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_UseAbilityOnTree(hAbility, iTree) end
    actionHandler[APickUpRune]           = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_PickUpRune(nRune) end
    actionHandler[APickUpItem]           = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_PickUpItem(hItem) end
    actionHandler[ADropItem]             = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_DropItem(hItem, Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[ADelay]                = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:Action_Delay(Vector(vLoc[1], vLoc[2], vLoc[3])) end
    actionHandler[APurchaseItem]         = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_PurchaseItem(sItem) end
    actionHandler[ASellItem]             = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_SellItem(hItem) end
    actionHandler[ADisassembleItem]      = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_DisassembleItem(hItem) end
    actionHandler[ASetItemCombineLock]   = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_SetItemCombineLock(hItem, not hItem.IsCombineLock()) end
    actionHandler[ASwapItems]            = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_SwapItems(ix1, ix2) end
    actionHandler[ABuyback]              = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Buyback() end
    actionHandler[AGlyph]                = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Glyph() end
    actionHandler[ALevelAbility]         = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_LevelAbility(sAbilityName) end
    actionHandler[ATakeOutpost]          = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return "" end
    actionHandler[ACourierBurst]         = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_BURST) end
    actionHandler[ACourierEnemySecret]   = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_ENEMY_SECRET_SHOP) end
    actionHandler[ACourierReturn]        = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_RETURN) end
    actionHandler[ACourierSecret]        = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_SECRET_SHOP) end
    actionHandler[ACourierTakeStash]     = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_TAKE_STASH_ITEMS) end
    actionHandler[ACourierTransfert]     = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_TRANSFER_ITEMS) end
    actionHandler[NotUsed1]              = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return "" end
    actionHandler[NotUsed2]              = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return "" end
    actionHandler[NotUsed3]              = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return "" end
    actionHandler[NotUsed4]              = function(vLoc, hUnit, hAbility, hTarget, iTree, nRune, fDelay, sItem, hItem, ix1, ix2, sAbilityName) return "" end
    return actionHandler
end

local action_table = get_action_table()

local function get_constant()
    local x_min, y_min, x_max, y_max = GetWorldBounds()
    -- int GetHeightLevel( vLocation )
    -- bool IsLocationVisible( vLocation )
    -- bool IsLocationPassable( vLocation )
    -- GetNeutralSpawners()
    -- GetDroppedItemList()
    -- vector GetTreeLocation( nTree )
    -- vector GetRuneSpawnLocation( nRuneLoc )
    -- vector GetShopLocation( nTeam, nShop )
    -- GetNearbyTrees
    -- GetNearbyHeroes
    -- GetNearbyCreeps
    -- GetNearbyLaneCreeps
    -- GetNearbyNeutralCreeps
    -- GetNearbyTowers
    -- GetNearbyBarracks
end

-- From the original dotaservice
local function get_player_info()
    local player_ids = {}

    for _, pid in pairs(GetTeamPlayers(TEAM_RADIANT)) do
        table.insert(player_ids, pid)
    end

    for _, pid in pairs(GetTeamPlayers(TEAM_DIRE)) do
        table.insert(player_ids, pid)
    end

    local players = {}
    for _, pid in pairs(player_ids) do
        local player = {}
        player['id'] = pid
        player['is_bot'] = IsPlayerBot(pid)
        player['team_id'] = GetTeamForPlayer(pid)
        player['hero'] = GetSelectedHeroName(pid)
        table.insert(players, player)
    end

    send_message({P = players})
end

-- Decode the message and execute the requested command
local function execute_rpc(message)
    local action    = message['0']
    local vLoc      = message['1']
    local hUnit     = message['2']
    local hAbility  = message['3']
    local hTarget   = message['4']
    local iTree     = message['5']
    local nRune     = message['6']
    local fDelay    = message['7']
    local sItem     = message['8']
    local hItem     = message['9']
    local ix1       = message['10']
    local ix2       = message['11']
    local sAbility  = message['12']

    -- No action ignore this is valid
    if action == nil then
        return
    end

    local fun = action_table[action]

    -- Specified action does not exist
    if fun == nil then
        pprint(message)
        pprint(action_table)
        send_message({E = 'Action `' .. action .. '` does not exist'})
        return
    end

    fun(vLoc,
        hUnit,
        hAbility,
        hTarget,
        iTree,
        nRune,
        fDelay,
        sItem,
        hItem,
        ix1,
        ix2,
        sAbility)
end

-- A single Game will generate 10 sample, one for each bot
local function delegate_think()
    -- Ready to process a new message
    -- Usefully for debugging but it pollutes the console
    -- send_message({S = "R"})

    -- Message uid used to know if we missed a message
    local uid = 0
    local message = receive_message()

    if message ~= nil then
        execute_rpc(message)
    end
end

-- Print the Base game information
get_player_info()

-- Take over the AI entirely
Think = delegate_think
