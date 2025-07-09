#ifndef HLRS_DEMO_TUI_H
#define HLRS_DEMO_TUI_H
#include <nlohmann/json.hpp>
#include "launch.h"
class DemoTui
{
public:
    DemoTui(const nlohmann::json &demos, int id);
    DemoLauncher launcher;
    int runningDemoId = -1;
};

#endif // HLRS_DEMO_TUI_H