

PROTOS := $(wildcard protos/*.pb.cc)


generate-protos:
	protoc -I=../protos  --cpp_out=protos/ ../protos/*.proto

compile: generate-protos
	g++ -shared -o bots.so -fPIC bot.cpp $(PROTOS)  -std=c++11 -lprotobuf

install: compile
	ln bots.so ../bots/botcpp_radiant.so
	ln bots.so ../bots/botcpp_dire.so
