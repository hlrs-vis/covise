#include "SimulationUI.h"
#include "BaseUI.h"
#include "app/system/EnergyType.h"
#include <stdexcept>
#include <algorithm>

using namespace core::interface::ui;

SimulationUI::SimulationUI(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent)
    : BaseUI(name, parent)
{
    init(factory, name, parent);
}

void SimulationUI::init(const core::interface::ui::IGUIFactory &factory, const std::string &name, core::interface::ui::IComponent *parent)
{
    if (!parent)
        throw std::runtime_error("SimulationUI cannot be initialized properly because the parent is NULL.");

    m_menu = factory.createMenu("Simulation", parent);

    m_lift = factory.createButton(m_menu.get(), "Up");

    m_energyGrids = factory.createSelectionList(m_menu.get(), "Grid Selection");
    std::vector<std::string> typeNames(ENERGYTYPE_RANGE.size());
    std::transform(ENERGYTYPE_RANGE.begin(), ENERGYTYPE_RANGE.end(), typeNames.data(), [&](EnergyType type)
        { return EnergyTypeToString(type); });
    m_energyGrids->setList(typeNames);
}
