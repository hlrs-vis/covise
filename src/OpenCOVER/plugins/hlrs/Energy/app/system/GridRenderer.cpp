#include "GridRenderer.h"
#include <osg/Switch>

GridRenderer::GridRenderer(osg::ref_ptr<osg::Group> rootNode)
    : m_root(rootNode)
{
    for (auto type : ENERGYTYPE_RANGE)
    {
        m_gridNodes[type] = new osg::Switch();
        m_root->addChild(m_gridNodes[type]);
    }
}

void GridRenderer::buildGrid(EnergyType type)
{
}

void GridRenderer::setData(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> data)
{
}

void GridRenderer::update()
{
}

void GridRenderer::updateStep(int timestep)
{
}

void GridRenderer::setVisible(EnergyType type, bool visible)
{
}
