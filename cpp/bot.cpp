#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>

#include <google/protobuf/text_format.h>

#include "protos/dota_gcmessages_common_bot_script.pb.h"


using namespace std;


extern "C" void Init(int team_id, void *b, void *c) {

}

// Note that because we only receive our time state
// this is not suited for training because we need both state to compute the
// symmetric reward

// v21 = &a1->CMsgBot[751];
// Observe(libraryTeamID, &a1->CMsgBot[751]);
extern "C" void Observe(int team_id, const CMsgBotWorldState& ws) {
    for (CMsgBotWorldState_Unit unit : ws.units() ) {
        cout << "PlayerID: " << unit.player_id() << endl;
        cout << "loc: x=" << unit.location().x() << " y=" << unit.location().y() << endl;
    }
}

// callDynamicallyLoadedLibrary(a1, v28, (__m128)time_delta)
// libraryTeamID = (__int64 *)*(unsigned int *)baseFuncPtr;
// LODWORD(baseFuncPtr) = Act(libraryTeamID, v21);
extern "C" void* Act(int team_id, CMsgBotWorldState &msg) {
    std::string s;
    if (google::protobuf::TextFormat::PrintToString(msg, &s)) {
        std::cout << "Your message:\n" << s;
    } else {
        std::cerr << "Message not valid (partial content: " << msg.ShortDebugString() << ")\n";
    }
}

//
extern "C" void Shutdown() {

}