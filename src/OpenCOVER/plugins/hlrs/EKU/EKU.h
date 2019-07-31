/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EKUEXAMPLEPLUGIN_H
#define EKUEXAMPLEPLUGIN_H
/****************************************************************************\
 **                                                            (C)2018 HLRS  **
 **                                                                          **
 ** Description: EKU camera position optimization for a bore field           **
 **                                                                          **
 **                                                                          **
 ** Author: Matthias Epple	                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Juni 2019  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
namespace opencover
{
namespace ui
{
class Button;
class Menu;
class Group;
class Slider;
class Label;
class Action;
}
}

#include <cover/coVRPlugin.h>

#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/ui/Action.h>

#include <cover/coVRPluginSupport.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>
#include <osg/PositionAttitudeTransform>

#include <iostream>
#include <vector>

#include <cover/ui/Owner.h>
#include<Cam.h>
#include<Truck.h>

using namespace opencover;

class EKU: public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    EKU();
    ~EKU();
    bool init();


    void doAddTruck();
    void doRemoveTruck();
    void doAddCam();
    void doRemoveCamera();
    //osg::Material *mtl;

    std::vector<Truck*> trucks;
    std::vector<Cam*> cameras;


private:
    //UI
    static EKU *plugin;
    ui::Menu *EKUMenu  = nullptr;
    ui::Action *AddTruck = nullptr, *RmvTruck = nullptr,*AddCam = nullptr;
    ui::Slider *FOVRegulator = nullptr, *VisibilityRegulator = nullptr;
    ui::Group *Frame = nullptr;
    ui::Label *Label = nullptr;

    //
  /*  //Position of Objects
    osg::PositionAttitudeTransform* moveDown;
    osg::PositionAttitudeTransform* moveToSide;
    osg::PositionAttitudeTransform* moveUp;

   */

    //Landscape
    osg::Geode* createPolygon();

};
#endif
