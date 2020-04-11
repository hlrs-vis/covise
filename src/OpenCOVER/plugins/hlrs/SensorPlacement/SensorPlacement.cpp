#include <iostream>
#include <memory> 

#include "SensorPlacement.h"


using namespace opencover;
namespace myHelpers{
template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

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
     for(size_t i{};i<100000;i++)
        Data::AddCamera(myHelpers::make_unique<Camera>());

     for(const auto& cam : Data::GetCameras())
        cam->calcVisibilityMatrix();
};

COVERPLUGIN(SensorPlacementPlugin)