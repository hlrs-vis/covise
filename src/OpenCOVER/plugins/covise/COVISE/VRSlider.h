/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRSLIDER_H
#define VRSLIDER_H

/*! \file
 \brief  3D slider for specifying scalar values

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 1998
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   28.09.1998
 */

#include <util/DLinkList.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>

namespace osg
{
class Geode;
}

#include <osg/Node>
#include <osg/Shape>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
namespace opencover
{
class RenderObject;

// A Slider Attribute has the following form
// SLIDER%d %cmodule \n instance \n host \n int/float \n parameterName
// \n min \n max \n value \n geometryType \n radius\n
//   (number_of_points \n x \n y \n z \n x\n ....)
//
//   %c = M (Menu) | V (VR) | v (VR line not visible),
//   S (spline) | s (spline line not visible)
//   geometryType = Sphere | SphereSegment | none

#ifdef PINBOARD
class VRButton;
#endif

// class definitions
class Slider
    : public vrui::coTrackerButtonInteraction
{
public:
    float min, max, value;

    osg::ref_ptr<osg::MatrixTransform> dcs; // dcs of Sphere
    osg::ref_ptr<osg::MatrixTransform> sphereTransform;

    osg::ref_ptr<osg::Node> node; // Geometry node, this slider belongs to
    osg::ref_ptr<osg::Node> line;

    // return true, if ths attribute os from the same module/parameter
    int isSlider(const char *attrib);
    bool updateInteraction();
#ifdef PINBOARD
    void updateMenu();
#endif
    void updateParameter();
    void updateValue(osg::Vec3 position, osg::Vec3 direction);
    float getMinDist(float x, float y, float z);
    float getMinDist(osg::Vec3 position, osg::Vec3 direction);
#ifdef PINBOARD
    static void menuCallback(void *sider, buttonSpecCell *spec);
#endif
    Slider(const char *attrib, const char *sattrib, osg::Node *n);
    ~Slider();
    char getSliderType();
    float getLength();

    virtual void startInteraction();
    virtual void stopInteraction();
    virtual void doInteraction();

private:
#ifdef PINBOARD
    void updateSpec(buttonSpecCell *spec);
#endif
    float getPoint(float *points, int i, int numPoints);
    float intersect(osg::Vec3 point, osg::Vec3 norm,
                    osg::Vec3 p1, osg::Vec3 p2);
    void updatePosition();

    int floatSlider;
    char *sattrib; // linestrip
    float *xcoords;
    float *ycoords;
    float *zcoords;
    float *length;
    float totalLength;
    int numPoints;
    float radius;
#ifdef PINBOARD
    VRButton *button;
#endif

    std::string feedback_information;
    std::string moduleName;
    std::string parameterName;
    std::string instanceName;
    std::string hostName;
    std::string dataType;
    char sliderType;
    std::string subMenu;
    std::string geometryType;
    float oldValue;
};

class SliderList : public covise::DLinkList<Slider *>
{
public:
    static SliderList *instance();

    /// add all Sliders defined in this Do to the menue
    /// if they are not jet there
    /// otherwise update the node field
    //void add( coDistributedObject *dobj, osg::Node *n);
    void add(RenderObject *robj, osg::Node *n);
    Slider *find(osg::Node *geode);
    void removeAll(osg::Node *geode);
    Slider *find(osg::Vec3 position, osg::Vec3 direction, float *distance);
    Slider *find(const char *attrib);

    void update();

private:
    SliderList();
    SliderList(const SliderList &);
    ~SliderList();
};
}
// done
#endif
