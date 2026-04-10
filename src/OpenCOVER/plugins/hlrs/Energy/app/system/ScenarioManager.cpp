#include "ScenarioManager.h"
#include <initializer_list>
#include <filesystem>
#include <algorithm>

namespace
{
constexpr auto IGNORE_SCENARIO_DIRS = { "static" };
auto getScenarioDirs(std::string_view scenarioDir)
{
    std::vector<std::string> scenarios;
    for (const auto &entry : std::filesystem::directory_iterator(scenarioDir))
        if (entry.is_directory() && !std::any_of(IGNORE_SCENARIO_DIRS.begin(), IGNORE_SCENARIO_DIRS.end(), [&](const auto &ignore)
                { return ignore == entry.path().filename().string(); }))
            scenarios.emplace_back(entry.path().filename().string());
    return scenarios;
}
}

ScenarioManager::ScenarioManager(opencover::ui::Group *parentMenu, std::string_view scenarioDir)
    : m_selectionList(new opencover::ui::SelectionList("Scenarios", parentMenu))
{
    setScenarios(getScenarioDirs(scenarioDir));
}

void ScenarioManager::setScenarios(const std::vector<std::string> &names)
{
    m_selectionList->setList(names);
    m_selectionList->select(0);
}
