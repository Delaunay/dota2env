#include "utility.h"

#include <fstream>
#include <sstream>

void _print(){
    logfile() << '\n';
    logfile() << std::flush;
}

std::string logfilename(int team_id) {
    std::stringstream ss;
    ss << "cpp_" << team_id << ".log";
    return ss.str();
}

std::ostream& logfile(int team_id) {
    static std::ofstream file = std::ofstream(logfilename(team_id).c_str());
    return file;
}
