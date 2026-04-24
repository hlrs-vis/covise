#include "GridRenderer.h"
#include "Storage.h"
#include "Scenario.h"
#include "DataFactory.h"
#include "app/osg/presentation/EnergyGrid.h"
#include "app/system/EnergyType.h"
#include "cover/coVRAnimationManager.h"
#include <memory>
#include <lib/core/utils/osgUtils.h>

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
    core::utils::osgUtils::switchTo(m_gridNodes[EnergyType::POWER], m_root);
    opencover::coVRAnimationManager::instance()->setNumTimesteps(100, m_root);
}

void GridRenderer::buildGrid(EnergyType type, DataLoadManager &loader)
{
    Scenario staticScenario { -1, "static" };
    auto package = loader.fetch(Storage::CSV, staticScenario, type);
    std::string typeString(EnergyTypeToString(type));
    if (package)
    {
        m_config.parent = m_gridNodes[type];
        m_gridMap[type] = DataFactory::create(*package, type, getLogger(), typeString, m_config);
    }
    else
    {
        error("Failed to build grid for " + typeString + "; Datapackage not read correctly.");
    }
}

void GridRenderer::setData(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> data, const std::string &species)
{
    if (m_gridMap.find(type) == m_gridMap.end())
    {
        std::string typeString(EnergyTypeToString(type));
        warn("Need to build grid for type " + typeString + " before updating shader.");
        return;
    }

    m_activeData[type] = data;
    // TODO: rework this to initialize it without species
    auto energyGrid = std::dynamic_pointer_cast<EnergyGrid>(m_gridMap[type]);
    if (energyGrid)
        energyGrid->setData(*data, species);
}

void GridRenderer::updateColorMapInShader(const opencover::ColorMap &map, EnergyType type)
{
    auto energyGrid = std::dynamic_pointer_cast<EnergyGrid>(m_gridMap[type]);
    if (energyGrid)
        // TODO: need to adjust this in EnergyGrid
        energyGrid->setColorMap(map, map);
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

void GridRenderer::setVisible(bool visible)
{
    for (auto [type, node] : m_gridNodes)
    {
        std::string typeString(EnergyTypeToString(type));
        if (visible)
        {
            if (auto it = m_gridMap.find(type); it != m_gridMap.end())
            {
                m_root->addChild(node);
            }
            else
            {
                warn("Grid for type " + typeString + " has not been initialized.");
                continue;
            }
        }
        else
        {
            m_root->removeChild(node);
        }
    }
}

void GridRenderer::translate(const osg::Vec3f &translate)
{
    for (auto &[_, mat] : m_gridNodes)
        mat->setMatrix(osg::Matrix::translate(translate));
}
