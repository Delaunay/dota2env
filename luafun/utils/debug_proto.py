import base64


with open('/home/setepenre/work/LuaFun/botscpp/decode.txt', 'rb') as f:
    data = f.read()


from luafun.game.dota2.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


print(type(data))
msg = CMsgBotWorldState()
msg.ParseFromString(data)

