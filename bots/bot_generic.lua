local dkjson = require('game/dkjson')
local msgpack = require('bots/msgpack')

local RECV_MSG = 'bots/IPC_recv'
local player_id = GetBot():GetPlayerID()
local faction = GetBot():GetTeam()

local function send_message(data)
    print('[IPC]', dkjson.encode(data))
end

-- Because we are inside a lua VM with limited capabilities
-- We use lua file to receive messages included in the source file
-- Others approach are not doable because
--      Lua 5.1 is embedded into Source 2, we have no library to link against
--      we could try to compile lua 5.1 as a shared library and compile our C package
--      and we might be able to load tcp socket lib but it is assuming require wasnt stripped
local function receive_message()
    local file = loadfile(RECV_MSG)

    if file ~= nil then
        local json_string = file()
        if json_string ~= nil then
            local dict, pos, err = dkjson.decode(json_string, 1, nil)

            if err then
                send_message({E = tostring(err)})
                return nil
            end

            return dict
        end
    end

    return nil
end

-- A single Game will generate 10 sample for each player
local function delegate_think()
    --
    send_message({S = "R"})

    -- Message uid used to know if we missed a message
    local uid = 0
    local message = receive_message()

    if message ~= nil then
        print(message)
    end
end



-- Take over the AI entirely
Think = delegate_think
