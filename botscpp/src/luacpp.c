
//class Cat {
//public:
//    Cat() {}
//    Cat(std::string const&) {}
//    virtual ~Cat() {}
//
//public:
//    void setName(const std::string&) {}
//    const std::string& getName() const {}
//    void eat(const std::list<std::string>& foods) {}
//
//};

//#include "luaa.hpp"
//using namespace luaaa;
//
//void luaaa_register_luafun(lua_State* state) {
//    // To export it:
//    auto s = LuaClass<Cat>(state, "AwesomeCat");
//    s.ctor<std::string>();
//    s.fun("setName", &Cat::setName);
//    s.fun("getName", &Cat::getName);
//    s.fun("eat", &Cat::eat);
//    s.def("tag", "Cat");
//}

#include <lua.h>
#include <lauxlib.h>

#ifdef _MSC_VER
#define LUAFUN __declspec(dllexport)
#else
#define LUAFUN
#endif


static int l_mult50(lua_State* L)
{
    double number = luaL_checknumber(L, 1);
    lua_pushnumber(L, number*50);
    return 1;
}

static const struct luaL_Reg luafunLib [] = {
    {"mult50", l_mult50}, //Your function name, and the function reference after
    {NULL, NULL}
};


LUAFUN int luaopen_luafun(lua_State* L) {
	luaL_register(L, "luafun", luafunLib);
	return 1;
}
