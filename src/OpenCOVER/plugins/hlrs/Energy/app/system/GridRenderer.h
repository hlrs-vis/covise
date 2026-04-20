#pragma once
#include "EnergyType.h"
#include "DataLoadManager.h"
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/ClassLogger.h>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osg/ref_ptr>

struct GridRenderConfig {
    std::vector<double> offset;
    std::string font;
    osg::ref_ptr<osg::MatrixTransform> parent = nullptr;
    static constexpr float sphereRadius = 2.0f;
    static constexpr float connectionsRadius = 1.0f;
};

class GridRenderer : core::ClassLogger
{
public:
    GridRenderer(osg::ref_ptr<osg::Switch> rootNode, const GridRenderConfig& offset, core::interface::ILogger& logger);

    void buildGrid(EnergyType type, DataLoadManager& loader);
    void setData(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> data);
    void update();
    void updateStep(int timestep);
    void setVisible(bool visible);

private:
    GridRenderConfig m_config;
    osg::ref_ptr<osg::Switch> m_root;
    std::map<EnergyType, osg::ref_ptr<osg::MatrixTransform>> m_gridNodes;
    std::map<EnergyType, std::unique_ptr<core::interface::IEnergyGrid>> m_gridMap;
    std::map<EnergyType, std::shared_ptr<core::simulation::SimulationResult>> m_activeData;
};
