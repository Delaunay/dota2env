// g++ -shared -o botcpp_radiant.so -fPIC botcpp_radiant.cpp dota_gcmessages_common_bot_script.pb.cc -std=c++11 -lprotobuf

extern "C" void Init(void * a, void *b, void *c) {

}


extern "C" void Observe(int team_id, const CMsgBotWorldState& ws) {

}

extern "C" void * Act(int team_id) {

}

extern "C" void Shutdown() {

}