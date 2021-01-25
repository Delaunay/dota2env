local dkjson = require('game/dkjson')

local RECV_MSG = 'bots/IPC_recv'
local player_id = GetBot():GetPlayerID()
local faction = '' .. GetBot():GetTeam()
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
            local faction_dat = internal[faction]

            if faction_dat ~= nil then
                local player_dat = faction_dat[player_id]

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
AMoveToLocation                = 0   -- ( vLocation )
AMoveDirectly                  = 1   -- ( vLocation )
AMoveToUnit                    = 2   -- ( hUnit )
AAttackUnit                    = 3   -- ( hUnit, bOnce = True )
AAttackMove                    = 4   -- ( vLocation )
AUseAbility                    = 5   -- ( hAbility )
AUseAbilityOnEntity            = 6   -- ( hAbility, hTarget )
AUseAbilityOnLocation          = 7   -- ( hAbility, vLocation )
AUseAbilityOnTree              = 8   -- ( hAbility, iTree )
APickUpRune                    = 9   -- ( nRune )
APickUpItem                    = 10  -- ( hItem )
ADropItem                      = 11  -- ( hItem, vLocation )
ADelay                         = 12  -- ( fDelay )
APurchaseItem                  = 13  -- ( sItemName )
ASellItem                      = 14  -- ( hItem )
ADisassembleItem               = 15  -- ( hItem )
ASetItemCombineLock            = 16  -- ( hItem, bLocked )
ASwapItems                     = 17  -- ( index1, index2 )
ABuyback                       = 18  -- ()
AGlyph                         = 19  -- ()
ALevelAbility                  = 20  -- ( sAbilityName )
ATakeOutpost                   = 21  -- ()
ACourierBurst                  = 22
ACourierEnemySecret            = 23
ACourierReturn                 = 24
ACourierSecret                 = 25
ACourierTakeStash              = 26
ACourierTransfert              = 27
NotUsed1 = 28
NotUsed2 = 29
NotUsed3 = 30
NotUsed4 = 31

-- Map the action ID to its function
local ActionHandler = {
    MoveToLocation                = function() end,
    MoveDirectly                  = function() end,
    MoveToUnit                    = function() end,
    AttackUnit                    = function() end,
    AttackMove                    = function() end,
    UseAbility                    = function() end,
    UseAbilityOnEntity            = function() end,
    UseAbilityOnLocation          = function() end,
    UseAbilityOnTree              = function() end,
    PickUpRune                    = function() end,
    PickUpItem                    = function() end,
    DropItem                      = function() end,
    Delay                         = function() end,
    PurchaseItem                  = function() end,
    SellItem                      = function() end,
    DisassembleItem               = function() end,
    SetItemCombineLock            = function() end,
    SwapItems                     = function() end,
    Buyback                       = function() end,
    Glyph                         = function() end,
    LevelAbility                  = function() end,
    TakeOutpost                   = function() end,
    CourierBurst                  = function() end,
    CourierEnemySecret            = function() end,
    CourierReturn                 = function() end,
    CourierSecret                 = function() end,
    CourierTakeStash              = function() end,
    CourierTransfert              = function() end,
    NotUsed1                      = function() end,
    NotUsed2                      = function() end,
    NotUsed3                      = function() end,
    NotUsed4                      = function() end,
}

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
    send_message({E = message})
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
