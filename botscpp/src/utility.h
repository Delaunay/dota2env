#ifndef BOT_UTILITY_HEADER
#define BOT_UTILITY_HEADER

#include <iostream>

std::ostream& logfile(int i = 0);

void _print();

template<typename A, typename ...Args>
void _print(const A &a, Args... args) {
    logfile() << a << ' ';
    _print(args...);
}

template<typename ...Args>
void print(Args... args) {
    logfile() << "[CPP]" << ' ';
    _print(args...);
}

#endif
