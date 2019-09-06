#pragma once

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>
#include<osgText/Font>
#include<osgText/Text>

#include<Sensor.h>
using namespace opencover;

class Truck
{
    friend class mySensor;
public:

    osg::Box *truck;
    osg::Vec3 pos;

    Truck(osg::Vec3 pos);
    Truck(const Truck&) { ++count; }
    ~Truck();
    virtual bool destroy();
    static size_t count;

    osg::ref_ptr<osg::Geode> getTruckDrawable()const{return truckGeode;}
    void updateColor();
    void resetColor();

private:
    const float length = 2.0f;
    const float width = 2.0f;
    const float height = 2.0f;

    osg::ref_ptr<osg::Geode> truckGeode;
    osg::ref_ptr<osgText::Text> text;
    osg::ref_ptr<osg::TessellationHints> hint;
    osg::ShapeDrawable *truckDrawable;
    void setStateSet(osg::StateSet *stateSet);
};



