#pragma once

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>
#include<osgText/Font>
#include<osgText/Text>

using namespace opencover;

class Truck
{
public:

    osg::Box *truck;
    osg::Vec3 pos;

    Truck(osg::Vec3 pos);
    Truck(const Truck&) { ++count; }
    ~Truck();
    virtual bool destroy();
    static size_t count;

private:
    const float length = 2.0f;
    const float width = 2.0f;
    const float height = 2.0f;

    osg::ref_ptr<osg::Geode> truckGeode;
    osg::ref_ptr<osgText::Text> text;

};



