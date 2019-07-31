#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>

using namespace opencover;

class Truck
{
public:

    osg::Box *truck;
    osg::Vec3 pos;

    Truck(osg::Vec3 pos);
    ~Truck();
    virtual bool destroy();

private:
    const float length = 1.0f;
    const float width = 2.0f;
    const float height = 5.0f;

    osg::ref_ptr<osg::Geode> truckGeode;
};



