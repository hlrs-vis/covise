#pragma once
#include "EnergyType.h"
#include "DataLoadManager.h"
#include <lib/core/interfaces/IMovable.h>
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/ClassLogger.h>
#include <osg/Group>
#include <osg/Switch>
#include <osg/ref_ptr>

struct GridRenderConfig {
    std::vector<double> offset;
    std::string font;
};

class GridRenderer : core::ClassLogger
{
public:
    GridRenderer(osg::ref_ptr<osg::Switch> rootNode, const GridRenderConfig& offset, core::interface::ILogger& logger);

    void buildGrid(EnergyType type, DataLoadManager& loader);
    void setData(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> data);
    void update();
    void updateStep(int timestep);
    void setVisible(EnergyType type, bool visible);

private:
    GridRenderConfig m_config;
    osg::ref_ptr<osg::Switch> m_root;
    std::map<EnergyType, osg::ref_ptr<osg::MatrixTransform>> m_gridNodes;
    std::map<EnergyType, std::shared_ptr<core::simulation::SimulationResult>> m_activeData;
};
