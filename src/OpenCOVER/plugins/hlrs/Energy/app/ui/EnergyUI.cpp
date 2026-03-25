#include "EnergyUI.h"

using namespace core::interface::ui;

EnergyUI::EnergyUI(const IGUIFactory &factory, const std::string &name, IComponent *parent)
    : BaseUI(name, parent)
    , m_tab(nullptr)
    , m_gridControlButton(nullptr)
    , m_energySwitchControlButton(nullptr)
{
    init(factory, parent);
}

void EnergyUI::init(const IGUIFactory &factory, IComponent *parent)
{
    m_tab = factory.createMenu("Energy", parent);

    initOverview(factory);
}

void EnergyUI::initOverview(const IGUIFactory &factory)
{
    m_controlPanel = factory.createMenu("Control", m_tab.get());

    m_energySwitchControlButton = factory.createButton(m_controlPanel.get(), "BuildingSwitch");
    m_energySwitchControlButton->setText("Buildings");
    m_energySwitchControlButton->setState(true);

    m_gridControlButton = factory.createButton(m_controlPanel.get(), "GridSwitch");
    m_gridControlButton->setText("Grid");
    m_gridControlButton->setState(true);
}
