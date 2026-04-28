#pragma once
#include "Scenario.h"
#include <functional>
#include <string_view>
#include <vector>
#include <string>

class ScenarioManager
{
public:
    explicit ScenarioManager(std::string_view scenarioDir);

    const std::vector<std::string> &getScenarioNames() const { return m_scenarios; }
    int getCurrentScenarioIndex() const { return m_currentIndex; }
    std::string getCurrentScenarioString() const;
    Scenario getSelectedScenario() const;

    void selectScenario(int index);
    void setOnScenarioChanged(std::function<void(int)> cb) { m_onChanged = cb; }

private:
    void loadScenarios(std::string_view scenarioDir);

    std::vector<std::string> m_scenarios;
    int m_currentIndex = 0;
    std::function<void(int)> m_onChanged;
};
