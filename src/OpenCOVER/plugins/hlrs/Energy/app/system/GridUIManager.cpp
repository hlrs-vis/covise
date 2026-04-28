#include "GridUIManager.h"

GridUIManager::GridUIManager(GridRenderer &renderer, core::interface::ui::IComponent *parentMenu)
    : m_renderer(renderer)
{
}

void GridUIManager::updateUIState(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> result)
{
    // TODO: Implement UI updates (color bars, etc)
}

void GridUIManager::handleLift(bool on)
{
    constexpr float uplift = 30.0f;
    auto active = on ? 1 : 0;
    m_renderer.translate(osg::Vec3f(0, 0, uplift * active));
}
