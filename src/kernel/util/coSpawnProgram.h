#ifndef VRB_REMOTE_LAUCHER_SPAWN_PROGRAMM_H
#define VRB_REMOTE_LAUCHER_SPAWN_PROGRAMM_H

#include "coExport.h"

#include <vector>
#include <string>
namespace covise
{
    //args: first arg must be executable name, last arg must be nullptr;
    UTILEXPORT void spawnProgram(const std::vector<const char *> &args);

    //name: executable name, args: command line args
    UTILEXPORT void spawnProgram(const std::string &name, const std::vector<std::string> &args);

    //returns the " " separated tokens from the commandLine string as a vector
    UTILEXPORT std::vector<const char*> parseCmdArgString(const std::string &commandLine);

} //namespace covise

#endif