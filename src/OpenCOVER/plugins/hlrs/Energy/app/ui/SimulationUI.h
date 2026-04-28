#pragma once

#include "BaseUI.h"
#include "app/system/EnergyType.h"
#include "app/system/Storage.h"
#include <lib/core/interfaces/ui/IButton.h>
#include <lib/core/interfaces/ui/ISelectionList.h>
#include <lib/core/interfaces/ui/IGUIFactory.h>

#include <memory>
#include <map>

class SimulationUI : BaseUI
{
public:
    SimulationUI(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent);
    void setLiftCallback(const std::function<void(bool)> &func) { m_lift->setCallback(func); }
    void setEnergyGridSelectionCallback(const std::function<void(int)> &func) { m_energyGrids->setCallback(func); }
    void setStorageSelectionCallback(EnergyType type, const std::function<void(int)> &func);
    void setStorageSelectionDefault(EnergyType type, Storage storage);
    Storage getSelectedStorage(EnergyType type);

private:
    void init(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent);

    std::unique_ptr<core::interface::ui::IMenu> m_menu;
    std::unique_ptr<core::interface::ui::ISelectionList> m_energyGrids;
    std::unique_ptr<core::interface::ui::IButton> m_lift;
    std::map<EnergyType ,std::unique_ptr<core::interface::ui::ISelectionList>> m_storage;

    // Powergrid UI => use own class
    // opencover::ui::Menu *m_powerGridMenu;
    // opencover::ui::Button *m_updatePowerGridSelection;
};
