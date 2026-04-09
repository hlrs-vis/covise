#pragma once
#include <cover/ui/Group.h>
#include <cover/ui/SelectionList.h>

#include <functional>

class ScenarioManager
{
public:
    ScenarioManager(opencover::ui::Group *parentMenu);

    void setOnScenarioChanged(std::function<void(int)> cb)
    {
        m_scenarioChangedCb = cb;
    }

    void setScenarios(const std::vector<std::string> &names)
    {
        m_selectionList->setList(names);
        m_selectionList->select(0);
    }

    int getCurrentScenario() const
    {
        return m_selectionList->selectedIndex();
    }

    auto getCurrentScenarioString() const
    {
        return m_selectionList->selectedItem();
    }

private:
    opencover::ui::SelectionList *m_selectionList;
    std::function<void(int)> m_scenarioChangedCb;
};
