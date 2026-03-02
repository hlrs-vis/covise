#include "EnergyUI.h"

#include <cover/coVRPluginSupport.h>

using namespace opencover;

EnergyUI::EnergyUI(const std::string &name, opencover::ui::Manager *manager)
    : BaseUI(name, cover->ui)
    , m_tab(nullptr)
    , m_controlPanel(nullptr)
    , m_tabPanel(nullptr)
    , m_gridControlButton(nullptr)
    , m_energySwitchControlButton(nullptr)
{
    init();
}

void EnergyUI::init()
{
    m_tab = new ui::Menu("Energy", this);
    m_tab->setText("Energy");

    initOverview();
}

void EnergyUI::initOverview()
{
    m_controlPanel = new ui::Menu(m_tab, "Control");
    m_controlPanel->setText("Control_Panel");

    m_energySwitchControlButton = new ui::Button(m_controlPanel, "BuildingSwitch");
    m_energySwitchControlButton->setText("Buildings");
    m_energySwitchControlButton->setState(true);

    m_gridControlButton = new ui::Button(m_controlPanel, "GridSwitch");
    m_gridControlButton->setText("Grid");
    m_gridControlButton->setState(true);
}
