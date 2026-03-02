#pragma once

#include "BaseUI.h"

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Manager.h>
#include <cover/coTUIListener.h>
#include <cover/coTabletUI.h>

class EnergyUI : BaseUI
{
public:
    EnergyUI(const std::string &name, opencover::ui::Manager *manager);
    void setSwitchCallback(BtnCallback func) { setBtnCallback(m_energySwitchControlButton, func); }
    void setControlCallback(BtnCallback func) { setBtnCallback(m_gridControlButton, func); }
    auto getTabMenu() { return m_tab; }
    auto getControlMenu() { return m_controlPanel; }
    auto getTabPanel() { return m_tabPanel; }

private:
    void init();
    void initOverview();

    opencover::ui::Menu *m_tab;
    opencover::ui::Menu *m_controlPanel;
    opencover::coTUITab *m_tabPanel;
    opencover::ui::Button *m_gridControlButton;
    opencover::ui::Button *m_energySwitchControlButton;
};
