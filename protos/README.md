# Game Proto

* https://dev.dota2.com/forum/dota-2/bot-scripting/265685-current-cmsgbotworldstate?t=284435#post1437167
* https://github.com/SteamDatabase/GameTracking-Dota2/tree/master/Protobufs
* https://github.com/SteamDatabase/GameTracking-Dota2/blob/master/Protobufs/dota_gcmessages_common_bot_script.proto

```
protoc --proto_path=. --python_out=../luafun/game/dota2 *
```
