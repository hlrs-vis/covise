#ifndef VRB_REMOTE_LAUCHER_SPAWN_PROGRAMM_H
#define VRB_REMOTE_LAUCHER_SPAWN_PROGRAMM_H

#include "export.h"
#include "MessageTypes.h"

#include <vector>
#include <string>
namespace vrb{
namespace launcher
{
    REMOTELAUNCHER_EXPORT void spawnProgram(Program p, const std::vector<std::string> &args);

    REMOTELAUNCHER_EXPORT void spawnProgram(const char *name, const std::vector<std::string> &args);
} // namespace launcher
} // vrb

#endif