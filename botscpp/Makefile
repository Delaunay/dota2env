
.PHONY: compile
PROTOS := $(wildcard protos/*.pb.cc)


generate-protos:
	protoc -I=../protos  --cpp_out=protos/ ../protos/*.proto

# This is the attempt at making a full C++ bot
compile-botscpp: generate-protos
	g++ -I. -shared -o bin/bots.so -fPIC src/bot.cpp src/utility.cpp $(PROTOS) -std=c++11 -lprotobuf
	cp -f bin/bots.so ../bots/botcpp_radiant.so

# that was an attempt at making a C++ extension for lua
# does not work because dota does not want to load it
# works with lua regular interpreter
# probably related to missing `package.loadlib`
compile-luacpp:
	# git checkout v5.1
	# g++ -shared -I ../dependencies/lua/ -I dependencies/ -o bin/luafun.so -fPIC src/luacpp.cpp -std=c++11
	gcc -shared -I ../dependencies/lua/ -I dependencies/ -o bin/luafun.so -fPIC src/luacpp.c -std=c11