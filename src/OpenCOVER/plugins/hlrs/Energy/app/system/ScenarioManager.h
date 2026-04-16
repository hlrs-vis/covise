#pragma once
#include "Scenario.h"
#include <cover/ui/Group.h>
#include <cover/ui/SelectionList.h>

#include <functional>
#include <string_view>

class ScenarioManager
{
public:
    ScenarioManager(opencover::ui::Group *parentMenu, std::string_view scenarioDir);

    void setOnScenarioChanged(std::function<void(int)> cb)
    {
        m_selectionList->setCallback(cb);
    }

    int getCurrentScenarioIndex() const
    {
        return m_selectionList->selectedIndex();
    }

    auto getCurrentScenarioString() const
    {
        return m_selectionList->selectedItem();
    }

    Scenario getScenario() const { return { getCurrentScenarioIndex(), getCurrentScenarioString() }; }

private:
    void setScenarios(const std::vector<std::string> &names);

    opencover::ui::SelectionList *m_selectionList;
};
