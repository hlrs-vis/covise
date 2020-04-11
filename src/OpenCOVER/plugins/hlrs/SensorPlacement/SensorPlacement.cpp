#include <iostream>


#include "SensorPlacement.h"


using namespace opencover;


bool SensorPlacementPlugin::init()
{
    std::cout<<"SensorPlacementPlugin loaded"<<std::endl;

    return true;
}
void SensorPlacementPlugin::preFrame()
{

}
SensorPlacementPlugin::SensorPlacementPlugin() : ui::Owner("SensorPlacementPlugin", cover->ui)
{
}
SensorPlacementPlugin::~SensorPlacementPlugin()
{
    std::cout <<"Shut down Sensor Placement Plugin" << std::endl;
};

COVERPLUGIN(SensorPlacementPlugin)