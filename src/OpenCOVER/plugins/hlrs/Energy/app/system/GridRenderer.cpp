#include "GridRenderer.h"
#include "Storage.h"
#include "Scenario.h"
#include "DataFactory.h"

GridRenderer::GridRenderer(osg::ref_ptr<osg::Switch> rootNode, const GridRenderConfig &config, core::interface::ILogger &logger)
    : core::ClassLogger(logger, "GridRenderer")
    , m_config(config)
    , m_root(rootNode)
{
    for (auto type : ENERGYTYPE_RANGE)
    {
        m_gridNodes[type] = new osg::MatrixTransform;
        m_gridNodes[type]->setName(EnergyTypeToString(type));
        m_root->addChild(m_gridNodes[type]);
    }
}

void GridRenderer::buildGrid(EnergyType type, DataLoadManager &loader)
{
    Scenario staticScenario { -1, "static" };
    auto package = loader.fetch(Storage::CSV, staticScenario, type);
    if (package)
    {
        // auto grid = DataFactory::create(*package, type, getLogger());
    }
    else
    {
        std::string typeString(EnergyTypeToString(type));
        error("Failed to build grid for " + typeString);
    }
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
