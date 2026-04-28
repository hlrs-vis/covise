#pragma once
#include "Scenario.h"
#include "lib/core/interfaces/ui/ISelectionList.h"
#include "lib/core/interfaces/ui/IGUIFactory.h"
#include <cover/ui/Group.h>
#include <cover/ui/SelectionList.h>

#include <functional>
#include <string_view>

class ScenarioManager
{
public:
    ScenarioManager(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent, std::string_view scenarioDir);

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

    std::unique_ptr<core::interface::ui::ISelectionList> m_selectionList;
};
