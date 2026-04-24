#pragma once

#include "BaseUI.h"
#include "lib/core/interfaces/ui/IButton.h"
#include "lib/core/interfaces/ui/ISelectionList.h"
#include <lib/core/interfaces/ui/IGUIFactory.h>

#include <memory>

class SimulationUI : BaseUI
{
public:
    SimulationUI(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent);
    void setLiftCallback(const std::function<void(bool)> &func) { m_lift->setCallback(func); }
    void setEnergyGridSelectionCallback(const std::function<void(int)> &func) { m_energyGrids->setCallback(func); }

private:
    void init(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent);

    std::unique_ptr<core::interface::ui::IMenu> m_menu;
    std::unique_ptr<core::interface::ui::ISelectionList> m_energyGrids;
    std::unique_ptr<core::interface::ui::IButton> m_lift;

    // Powergrid UI => use own class
    // opencover::ui::Menu *m_powerGridMenu;
    // opencover::ui::Button *m_updatePowerGridSelection;
};
