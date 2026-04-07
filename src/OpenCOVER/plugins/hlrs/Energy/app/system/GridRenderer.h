#pragma once
#include "EnergyType.h"
#include <lib/core/simulation/simulationresult.h>
#include <osg/Group>
#include <osg/ref_ptr>

class GridRenderer
{
public:
    GridRenderer(osg::ref_ptr<osg::Group> rootNode);

    void buildGrid(EnergyType type);
    void setData(EnergyType type, std::shared_ptr<core::simulation::SimulationResult> data);
    void updateStep(int timestep);
    void setVisible(EnergyType type, bool visible);

private:
    osg::ref_ptr<osg::Group> m_root;
    std::map<EnergyType, osg::ref_ptr<osg::Switch>> m_gridNodes;
    std::map<EnergyType, std::shared_ptr<core::simulation::SimulationResult>> m_activeData;
};
