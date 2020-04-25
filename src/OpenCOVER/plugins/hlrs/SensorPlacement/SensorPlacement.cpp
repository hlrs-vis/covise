#include <iostream>
#include <memory> 
#include <future>
#include "SensorPlacement.h"
#include "Profiling.h"
#include "Helper.h"
using namespace opencover;

/*void Data::UpdateCameras()
{
    for(const auto& cam : Data::GetCameras())
    {
       GetInstance().m_Futures.push_back(std::async(std::Launch::async,Camera::calcVisibilityMatrix,cam));
    }
}
*/
bool SensorPlacementPlugin::init()
{
    std::cout<<"SensorPlacementPlugin loaded"<<std::endl;

    return true;
}
void SensorPlacementPlugin::preFrame()
{
    zone->preFrame();
}
SensorPlacementPlugin::SensorPlacementPlugin() : ui::Owner("SensorPlacementPlugin", cover->ui)
{
    SP_PROFILE_BEGIN_SESSION("Init","SensorPlacement-Startup.json");

    SP_PROFILE_FUNCTION();
     for(size_t i{};i<100000;i++)
     {
        SP_PROFILE_SCOPE();
     //   Data::AddCamera(myHelpers::make_unique<Camera>());
     }
     std::cout<<"first profile ended"<<std::endl;
     for(const auto& cam : Data::GetCameras())
     {
        SP_PROFILE_SCOPE();
     }
    //Data::UpdateCameras();
    osg::Matrix m;
    zone = new Zone(m);


};

SensorPlacementPlugin::~SensorPlacementPlugin()
{
    std::cout<<"Shut down Sensor Placement"<<std::endl;
    SP_PROFILE_END_SESSION();

}

COVERPLUGIN(SensorPlacementPlugin)