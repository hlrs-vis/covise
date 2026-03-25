#pragma once

#include "BaseUI.h"
#include <lib/core/interfaces/ui/IMenu.h>
#include <lib/core/interfaces/ui/IButton.h>
#include <lib/core/interfaces/ui/IGUIFactory.h>
#include <memory>

class EnergyUI : BaseUI
{
public:
    EnergyUI(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent);
    void setSwitchCallback(BtnCallback func) { setBtnCallback(m_energySwitchControlButton.get(), func); }
    void setControlCallback(BtnCallback func) { setBtnCallback(m_gridControlButton.get(), func); }
    auto getTabMenu() { return m_tab.get(); }
    auto getControlMenu() { return m_controlPanel.get(); }

private:
    void init(const core::interface::ui::IGUIFactory &factory, core::interface::ui::IComponent *parent);
    void initOverview(const core::interface::ui::IGUIFactory &factory);

    std::unique_ptr<core::interface::ui::IMenu> m_tab;
    std::unique_ptr<core::interface::ui::IMenu> m_controlPanel;
    std::unique_ptr<core::interface::ui::IButton> m_gridControlButton;
    std::unique_ptr<core::interface::ui::IButton> m_energySwitchControlButton;
};
