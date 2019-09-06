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
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include<osg/Node>



#include <iostream>
#include <vector>
#include<functional>

#include <cover/ui/Owner.h>
#include<Cam.h>
#include<Truck.h>
#include<GA.hpp>
#include<FileReader.hpp>
#include<Sensor.h>



using namespace opencover;

class mySensor;
class EKU: public opencover::coVRPlugin, public opencover::ui::Owner
{
    friend class mySensor;
public:
    EKU();
    ~EKU();
    bool init();
    void doAddTruck();
    void doRemoveTruck();
    void doAddCam();
    void doRemoveCamera();
    //osg::Material *mtl;
   void preFrame();

    std::vector<Truck*> trucks;
    std::vector<Cam*> cameras;
    std::vector<CamDrawable*> finalCams;

    GA *ga;
    static EKU *plugin;
    osg::ref_ptr<osg::Node> scene;

private:
    //UI
    ui::Menu *EKUMenu  = nullptr;
    ui::Action *AddTruck = nullptr, *RmvTruck = nullptr,*AddCam = nullptr;
    ui::Slider *FOVRegulator = nullptr, *VisibilityRegulator = nullptr;
    ui::Group *Frame = nullptr;
    ui::Label *Label = nullptr;
    ui::Button *MakeCamsInvisible = nullptr;

    osg::MatrixTransform *mymtf;
    vrui::coTrackerButtonInteraction *myinteraction;
    bool interActing;
    coSensorList sensorList;
    std::vector<mySensor*> userInteraction;


    //Landscape
    osg::Geode* createPolygon();
    osg::Geode* createPoints();
    osg::ref_ptr<osg::Group> finalScene;

  //  FileReaderWriter *readerWriter;
  //  FindNamedNode fnn;//NOTE: make to pointer




};
#endif
