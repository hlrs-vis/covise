/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SPRAY_H
#define SPRAY_H

#include "nodevisitorvertex.h"
#include "nozzlemanager.h"
#include "parser.h"
#include "raytracer.h"

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/Quat>
#include <osg/Vec3f>

#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRTui.h>
#include <PluginUtil/coSphere.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/VRSceneGraph.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Label.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Label.h>
#include <cover/ui/SelectionList.h>

using namespace covise;
using namespace opencover;


#include "string.h"


//Set singletons to 0
nozzleManager* nozzleManager::_instance = 0;
raytracer* raytracer::_instance = 0;

class SprayPlugin : public coVRPlugin, public ui::Owner
{
private:
    osg::Group *scene;
    osg::Geode* floorGeode;
    osg::Geode* testBoxGeode;

    osg::MatrixTransform *baseTransform;

    std::list<int> idGeo;


    //All menus and submenus
    ui::Menu* sprayMenu_ = nullptr;
    ui::Menu* nozzleCreateMenu = nullptr;
    ui::Menu* nozzleCreateMenuStandard = nullptr;
    ui::Menu* nozzleCreateMenuImage = nullptr;
    ui::Menu* nozzleEditMenu_ = nullptr;
    ui::Menu* saveMenu_ = nullptr;
    ui::Menu* loadMenu_ = nullptr;
    ui::Menu* testMenu = nullptr;
    ui::Menu* bbEditMenu = nullptr;

    //Actions on main menu
    ui::Action* edit_ = nullptr;
    ui::Action* save_ = nullptr;
    ui::Action* load_ = nullptr;
    ui::Action* create_ = nullptr;
    ui::Action* remove_ = nullptr;

    //Buttons on main menu
    ui::Button* sprayStart_ = nullptr;

    //EditFields on main menu
    ui::EditField* newGenCreate_ = nullptr;
    ui::EditField* scaleFactorParticle = nullptr;

    //Selection list on main menu
    ui::SelectionList* nozzleIDL = nullptr;

    //Labels on main menu
    ui::Label* outputField_ = nullptr;

    //Variables on main menu
    int nozzleID = 0;
    int currentNozzleID = -1;
    bool sprayStart = false;
    bool creating = false;
    bool editing = false;
    bool TESTING = true;
    osg::Vec3 newBoundingBox = osg::Vec3(2000,2000,2000);
    class nozzle* editNozzle;

    /************************************************************/

    //Actions on edit menu
    ui::Action* acceptEdit_ = nullptr;

    //EditFields on edit menu
    ui::EditField* red_ = nullptr;
    ui::EditField* green_ = nullptr;
    ui::EditField* blue_ = nullptr;
    ui::EditField* alpha_ = nullptr;
    ui::EditField* param1 = nullptr;
    ui::EditField* param2 = nullptr;
    ui::EditField* rDeviation = nullptr;
    ui::EditField* rMinimum = nullptr;

    ui::EditField* moveX = nullptr;
    ui::EditField* moveY = nullptr;
    ui::EditField* moveZ = nullptr;

    //Buttons on edit menu
    ui::Button* interaction = nullptr;

    //Slider on edit menu
    ui::Slider* pressureSlider_ = nullptr;
    ui::Slider* rotX = nullptr;
    ui::Slider* rotY = nullptr;
    ui::Slider* rotZ = nullptr;

    //Variables on edit menu
    float scaleValue_ = 1;
    float deviation = 0;
    float minimum = 0;
    osg::Matrix memMat;
    osg::Vec3 transMat = osg::Vec3(0,0,0);
    osg::Vec4 newColor = osg::Vec4(1,1,1,1);

    /************************************************************/

    //EditFields on create menu
    ui::EditField* pathNameFielddyn_ = nullptr;
    ui::EditField* fileNameFielddyn_ = nullptr;
    ui::EditField* nozzleNameFielddyn_ = nullptr;

    //Variables on create menu
    float sprayAngle_ = 0;
    std::string decoy_ = "";

    /************************************************************/

    //EditFields on save/load menu
    std::string pathNameField_ = "";
    std::string fileNameField_ = "";
    std::string nozzleNameField_ = "";


public:
    SprayPlugin();

    bool init();
    bool destroy();

    bool update();

    void createTestBox(osg::Vec3 initPos, osg::Vec3 scale);
    void createTestBox(osg::Vec3 initPos, osg::Vec3 scale, bool manual);

    void createAndRegisterImageNozzle();
    void createAndRegisterStandardNozzle();

};
#endif // SPRAY_H
