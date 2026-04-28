#include "ScenarioManager.h"
#include <filesystem>
#include <algorithm>
#include <vector>
#include <string>
#include <string_view>

namespace
{
constexpr auto IGNORE_SCENARIO_DIRS = { "static" };
}

ScenarioManager::ScenarioManager(std::string_view scenarioDir)
{
    loadScenarios(scenarioDir);
}

void ScenarioManager::loadScenarios(std::string_view scenarioDir)
{
    if (!std::filesystem::exists(scenarioDir))
        return;

    for (const auto &entry : std::filesystem::directory_iterator(scenarioDir))
    {
        if (entry.is_directory() && !std::any_of(IGNORE_SCENARIO_DIRS.begin(), IGNORE_SCENARIO_DIRS.end(), [&](const auto &ignore)
                { return ignore == entry.path().filename().string(); }))
        {
            m_scenarios.emplace_back(entry.path().filename().string());
        }
    }

    if (!m_scenarios.empty())
    {
        m_currentIndex = 0;
    }
}

void ScenarioManager::selectScenario(int index)
{
    if (index >= 0 && index < static_cast<int>(m_scenarios.size()))
    {
        m_currentIndex = index;
        if (m_onChanged)
        {
            m_onChanged(m_currentIndex);
        }
    }
}

std::string ScenarioManager::getCurrentScenarioString() const
{
    if (m_currentIndex >= 0 && m_currentIndex < static_cast<int>(m_scenarios.size()))
    {
        return m_scenarios[m_currentIndex];
    }
    return "";
}

Scenario ScenarioManager::getSelectedScenario() const
{
    return { getCurrentScenarioIndex(), getCurrentScenarioString() };
}
