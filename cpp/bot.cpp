// g++ -shared -o botcpp_radiant.so -fPIC botcpp_radiant.cpp dota_gcmessages_common_bot_script.pb.cc -std=c++11 -lprotobuf

extern "C" void Init((int team_id, void *b, void *c) {

}

// Note that because we only receive our time state
// this is not suited for training because we need both state to compute the
// symmetric reward

// v21 = &a1->CMsgBot[751];
// Observe(libraryTeamID, &a1->CMsgBot[751]);
extern "C" void Observe(int team_id, const CMsgBotWorldState& ws) {

}

// callDynamicallyLoadedLibrary(a1, v28, (__m128)time_delta)
// libraryTeamID = (__int64 *)*(unsigned int *)baseFuncPtr;
// LODWORD(baseFuncPtr) = Act(libraryTeamID, v21);
extern "C" void* Act(int team_id, CMsgBotWorldState &msg) {

}

//
extern "C" void Shutdown() {

}