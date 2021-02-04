-- Could not open botcpp: /media/setepenre/local/SteamLibraryLinux/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots/botcpp_radiant.so: cannot open shared object file: No such file or directory

local dkjson = require('game/dkjson')
local pprint = require('bots/pprint')

local RECV_MSG = 'bots/IPC_recv'
local player_id = GetBot():GetPlayerID()
local faction = GetBot():GetTeam()

local str_faction = '' .. faction
local str_player_id = '' .. player_id

local uid = nil
local ipc_prefix = '[IPC]' .. faction .. '.' .. player_id

local SILENT = 0
local level = SILENT


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
--
-- I tried to move this out into its own package but integration test started failing
-- package seems slower
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

                if level ~= SILENT then
                    send_message({A = uid})
                end

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
local AStop                          = 0
local AMoveToLocation                = 1   -- ( vLocation )
local AMoveDirectly                  = 2   -- ( vLocation )
local AMoveToUnit                    = 3   -- ( hUnit )
local AAttackUnit                    = 4   -- ( hUnit, bOnce = True )
local AAttackMove                    = 5   -- ( vLocation )
local AUseAbility                    = 6   -- ( hAbility )
local AUseAbilityOnEntity            = 7   -- ( hAbility )
local AUseAbilityOnLocation          = 8   -- ( hAbility, vLocation )
local AUseAbilityOnTree              = 9   -- ( hAbility, iTree )
local APickUpRune                    = 10  -- ( nRune )
local APickUpItem                    = 11  -- ( hItem )
local ADropItem                      = 12  -- ( vLocation )
local APurchaseItem                  = 13  -- ( sItemName )
local ASellItem                      = 14  -- ( hItem )
local ADisassembleItem               = 15  -- ( hItem )
local ASetItemCombineLock            = 16  -- ( bLocked )
local ASwapItems                     = 17  -- ( index1, index2 )
local ABuyback                       = 18  -- ()
local AGlyph                         = 19  -- ()
local ALevelAbility                  = 20  -- ( sAbilityName )
local ACourierBurst                  = 21
-- local ACourierEnemySecret            = 23
local ACourierReturn                 = 22
local ACourierSecret                 = 23
local ACourierTakeStash              = 24
local CourierTransfer                = 25
local NotUsed5 = 26
local NotUsed0 = 27
local NotUsed1 = 28
local NotUsed2 = 29
local NotUsed3 = 30
local NotUsed4 = 31


-- look for the item that is close to the item we are looking for
-- our vloc should be exact
local function get_dropped_items(vloc)
    for _, obj in pairs(GetDroppedItemList()) do
        pprint.pprint(obj)
        -- Useless ?
        -- { item =  {}, owner = {}, playerid= 0}
    end

    return nil
end


local function _get_world_size()
    local x_min, y_min, x_max, y_max = GetWorldBounds()
    send_message({I = {
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    }, T = 'BOUNDS'})
end

local function _get_trees()
    -- GetAllTrees
    for i= 0,2500
    do
        local loc = GetTreeLocation(i)
        local tree = {i, loc.x, loc.y, loc.z}
        send_message({I = tree, T = 'TREE'})
    end
end

local function _get_runes()
    for i = 0,10
    do
        local loc = GetRuneSpawnLocation(i)
        local runes = {i, loc.x, loc.y, loc.z}
        send_message({I = runes, T = 'RUNE'})
    end
end


local function _get_shop()
    for team = 0,4
    do
        for shop = 0,10
        do
            local loc = GetShopLocation(team, shop)
            local runes = {team + shop * 10, team, shop, loc.x, loc.y, loc.z}
            send_message({I = runes, T = 'SHOP'})
        end
    end
end

local function _get_neutrals()
    local neutrals = GetNeutralSpawners()

    -- Looks useless
    -- {
    --      speed =  'normal',
    --      team =  2 ,
    --      type  =  'small'
    -- }

    for _, data in pairs(neutrals) do
        local speed = data.speed
        local team = data.team
        local type = data.type

        local result = {team, type, speed}
        send_message({I = result, T = 'NEUTRALS'})
    end
end


-- from https://stackoverflow.com/questions/2620377/lua-reflection-get-list-of-functions-fields-on-an-object
local seen = {}
function dump(t,i)
    seen[t]=true
    local s={}
    local n=0
    for k in pairs(t) do
        n=n+1 s[n]=k
    end
    table.sort(s)
    for k,v in ipairs(s) do
        print(i,v)
        v=t[v]
        if type(v)=="table" and not seen[v] then
            dump(v,i.."\t")
        end
    end
end


-- Dump a bunch of useful information
local function get_info()
    -- DebugDrawCircle
    -- DebugDrawLine
    -- DebugDrawText
    -- DebugPause

    -- CreateHTTPRequest
    -- CreateRemoteHTTPRequest

    -- GetLaneFrontAmount
    -- GetLaneFrontLocation
    -- GetLocationAlongLane

    -- int GetHeightLevel( vLocation )
    -- bool IsLocationVisible( vLocation )
    -- bool IsLocationPassable( vLocation )

    -- dump(_G,"")
    -- _get_neutrals()

     -- GetDroppedItemList()

    -- Useful for making a minimap of the vision
    ---------
    -- GetNearbyTrees
    -- GetNearbyHeroes
    -- GetNearbyCreeps
    -- GetNearbyLaneCreeps
    -- GetNearbyNeutralCreeps
    -- GetNearbyTowers
    -- GetNearbyBarracks
end



local bot = GetBot()
-- is it better to use GetCourierForPlayer ?
local hCourier = GetCourier(bot:GetPlayerID())


local function get_item(slot)
    local hItem = bot:GetItemInSlot(slot)
    if hItem == nil then
        send_message({E = 'Could not find item in slot ' .. slot})
        return nil
    end
    return hItem
end


local function get_ability(slot)
    local hAbility = bot:GetAbilityInSlot(slot - 17)
    if hAbility == nil then
        send_message({E = 'Could not find ability in slot ' .. slot})
        return nil
    end
    return hAbility
end


local function get_ability_handle(slot)
    local handle = nil

    if slot >= 17 then
        -- Ability slot
        handle = get_ability(slot)
    else
        -- Item slot
        handle = get_item(slot)
    end

    if handle == nil then
        return handle
    end

    return handle
end


-- Map the action ID to its function
-- This is all the actions the bots can make
local function get_action_table()
    -- Need a way to map to unit Handle from python
    local actionHandler = {}
    actionHandler[AStop]                 = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        return bot:Action_ClearActions(true)
    end
    actionHandler[AMoveToLocation]       = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        return bot:Action_MoveToLocation(vLoc)
    end
    actionHandler[AMoveDirectly]         = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        return bot:Action_MoveDirectly(vLoc)
    end
    actionHandler[AMoveToUnit]           = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        if hUnit == nil then
            return
        end
        return bot:Action_MoveToUnit(GetBotByHandle(hUnit))
    end
    actionHandler[AAttackUnit]           = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        if hUnit == nil then
            return
        end
        return bot:Action_AttackUnit(GetBotByHandle(hUnit), true)
   end
    actionHandler[AAttackMove]           = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        return bot:Action_AttackMove(vLoc)
    end
    actionHandler[AUseAbility]           = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hAbility = get_ability_handle(nSlot)
        if hAbility == nil then
            return
        end
        return bot:Action_UseAbility(hAbility)
    end
    actionHandler[AUseAbilityOnEntity]   = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        if hUnit == nil then
            return
        end

        local hAbility = get_ability_handle(nSlot)
        if hAbility == nil then
            return
        end
        return bot:Action_UseAbilityOnEntity(hAbility, GetBotByHandle(hUnit))
    end
    actionHandler[AUseAbilityOnLocation] = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hAbility = get_ability_handle(nSlot)
        if hAbility == nil then
            return
        end
        return bot:Action_UseAbilityOnLocation(hAbility, vLoc)
    end
    actionHandler[AUseAbilityOnTree]     = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hAbility = get_ability_handle(nSlot)
            if hAbility == nil then
                return
            end
        return bot:Action_UseAbilityOnTree(hAbility, iTree)
    end
    actionHandler[APickUpRune]           = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        -- It is easier to test if we move & pickup
        -- since pickup only happens if we are close enough and we do not move
        -- Note: like in a game the action needs to be called twice
        -- once to discover the rune and another to pick it up
        -- Note: not sure about the difference between ActionQueue and ActionPush
        local loc = GetRuneSpawnLocation(nRune)
        bot:ActionQueue_MoveToLocation(loc)
        return bot:ActionPush_PickUpRune(nRune)
    end
    actionHandler[APickUpItem]           = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        if hUnit == nil then
            return
        end
        local hItem = GetBotAbilityByHandle(hUnit)
        return bot:Action_PickUpItem(hItem)
    end
    actionHandler[ADropItem]             = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hItem = get_item(nSlot)
        if hItem == nil then
            return
        end
        return bot:Action_DropItem(hItem, vLoc)
    end
    actionHandler[APurchaseItem]         = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        if sItem == nil then
            return
        end
        return bot:ActionImmediate_PurchaseItem(sItem)
    end
    actionHandler[ASellItem]             = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hItem = get_item(nSlot)
        if hItem == nil then
            return
        end
        return bot:ActionImmediate_SellItem(hItem)
    end
    actionHandler[ADisassembleItem]      = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hItem = get_item(nSlot)
        if hItem == nil then
            return
        end
        return bot:ActionImmediate_DisassembleItem(hItem)
    end
    actionHandler[ASetItemCombineLock]   = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hItem = get_item(nSlot)
        if hItem == nil then
            return
        end
        return bot:ActionImmediate_SetItemCombineLock(hItem, not hItem:IsCombineLocked())
    end
    -- when swapping item there is a script that automatically transfer items to inventory
    -- if the inventory has space
    actionHandler[ASwapItems]            = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        return bot:ActionImmediate_SwapItems(nSlot, ix2) end
    actionHandler[ABuyback]              = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        return bot:ActionImmediate_Buyback() end
    actionHandler[AGlyph]                = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        return bot:ActionImmediate_Glyph() end
    actionHandler[ALevelAbility]         = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
        local hAbility = get_ability_handle(nSlot)
        if hAbility == nil then
            return
        end
        local sAbilityName = hAbility:GetName()
        return bot:ActionImmediate_LevelAbility(sAbilityName)
    end
    actionHandler[ACourierBurst]         = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_BURST) end
    -- actionHandler[ACourierEnemySecret]   = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_ENEMY_SECRET_SHOP) end
    actionHandler[ACourierReturn]        = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_RETURN) end
    actionHandler[ACourierSecret]        = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_SECRET_SHOP) end
    actionHandler[ACourierTakeStash]     = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_TAKE_STASH_ITEMS) end
    actionHandler[CourierTransfer]       = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return bot:ActionImmediate_Courier(hCourier, COURIER_ACTION_TRANSFER_ITEMS) end
    actionHandler[NotUsed0]              = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return "" end
    actionHandler[NotUsed1]              = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return "" end
    actionHandler[NotUsed2]              = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return "" end
    actionHandler[NotUsed3]              = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return "" end
    -- used for debug print
    actionHandler[NotUsed4]              = function(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2) return get_info() end
    return actionHandler
end

local action_table = get_action_table()

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
    local nSlot     = message['3']
    local iTree     = message['4']
    local nRune     = message['5']
    local sItem     = message['6']
    local ix2       = message['7']

    -- No action ignore this is valid
    if action == nil then
        return
    end

    local fun = action_table[action]

    -- Specified action does not exist
    if fun == nil then
        pprint(message)
        send_message({E = 'Action `' .. action .. '` does not exist'})
        return
    end

    -- Fix argument type
    if vLoc ~= nil then
        x = vLoc[1]
        y = vLoc[2]
        z = vLoc[3]

        if z == nil then
            z = bot:GetLocation().z
        end

        vLoc = Vector(x, y, z)
    end

    -- Execute actions
    fun(vLoc, hUnit, nSlot, iTree, nRune, sItem, ix2)
end


-- from https://developer.valvesoftware.com/wiki/Dota_Bot_Scripting#UNIT_SCOPED_FUNCTIONS
-- > This function will be called once per frame for every minion under control by a bot.
-- > For example, if you implemented it in bot_beastmaster.lua,
-- > it would constantly get called both for your boar and hawk while they're summoned and alive.
-- > The handle to the bear/hawk unit is passed in as hMinionUnit.
-- > Action commands that are usable on your hero are usable on the passed-in hMinionUnit.
--
-- Figure out how this would work with a NNet
local function delegate_minion_think(hMinionUnit)

end

-- A single Game will generate 10 sample, one for each bot
local function delegate_think()
    -- Ready to process a new message
    -- Usefully for debugging but it pollutes the console
    -- send_message({S = "R"})

    -- Do model inference there


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
