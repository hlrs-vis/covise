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
    std::string typeString(EnergyTypeToString(type));
    if (package)
    {
        if (!m_config.parent)
            m_config.parent = m_gridNodes[type];

        m_gridMap[type] = DataFactory::create(*package, type, getLogger(), typeString, m_config);
    }
    else
    {
        error("Failed to build grid for " + typeString + "; Datapackage not read correctly.");
    }
}

void GridRenderer::setData(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> data)
{
    m_activeData[type] = data;
    // TODO: set shader correctly
}

void GridRenderer::update()
{
    for (auto &[_, grid] : m_gridMap)
        if (grid)
            grid->update();
}

void GridRenderer::updateStep(int timestep)
{
    for (auto &[_, grid] : m_gridMap)
        if (grid)
            grid->updateTime(timestep);
}

void GridRenderer::setVisible(EnergyType type, bool visible)
{
}
