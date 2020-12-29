#ifndef VRB_REMOTE_LAUCHER_SPAWN_PROGRAMM_H
#define VRB_REMOTE_LAUCHER_SPAWN_PROGRAMM_H

#include "coExport.h"

#include <vector>
#include <string>
namespace covise
{
    //execPath: executable path, args: first arg must be executable name, last arg must be nullptr;
    UTILEXPORT void spawnProgram(const char* execPath, const std::vector<const char *> &args);

    //execPath: executable path, args: command line args
    UTILEXPORT void spawnProgram(const std::string & execPath, const std::vector<std::string> &args);

    //execPath: executable path, debugCommands: coCoviseConfig::getEntry("System.CRB.DebugCommand"), args: first arg must be executable name, last arg must be nullptr;
    UTILEXPORT void spawnProgramWithDebugger(const char* execPath, const std::string &debugCommands,const std::vector<const char*>& args);
    //execPath: executable path, debugCommands: coCoviseConfig::getEntry("System.CRB.MemcheckCommand"), args: first arg must be executable name, last arg must be nullptr;
    UTILEXPORT void spawnProgramWithMemCheck(const char* execPath, const std::string &debugCommands,const std::vector<const char*>& args);

    //returns the " " separated tokens from the commandLine string as a vector
    UTILEXPORT std::vector<const char*> parseCmdArgString(const std::string &commandLine);


} //namespace covise

#endif