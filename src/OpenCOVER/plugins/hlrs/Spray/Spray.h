#ifndef SPRAY_H
#define SPRAY_H

#include "../../../../../../../../usr/include/osg/MatrixTransform"
#include "../../../../../../../../usr/include/osg/Matrix"
#include "../../../../../../../../usr/include/osg/ShapeDrawable"
#include "../../../../../../../../usr/include/osg/Shape"
#include "../../../../../../../../usr/include/osg/Quat"
#include "../../../../../../../../usr/include/osg/Vec3f"

#include "../../../../kernel/config/CoviseConfig.h"
using namespace covise;

#include "../../../../OpenCOVER/cover/coVRPluginSupport.h"
#include "../../../../OpenCOVER/cover/coVRFileManager.h"
#include "../../../../OpenCOVER/cover/coVRPlugin.h"
#include "../../../../OpenCOVER/cover/coVRTui.h"
#include "../../../../OpenCOVER/PluginUtil/coSphere.h"
#include "../../../../OpenCOVER/PluginUtil/coVR3DTransRotInteractor.h"
#include <cover/VRSceneGraph.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include "../../../../OpenCOVER/cover/ui/Button.h"
#include "../../../../OpenCOVER/cover/ui/Slider.h"
#include "../../../../OpenCOVER/cover/ui/Label.h"
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Label.h>
using namespace opencover;

#include "string.h"
#include <stdio.h>

#include "nodevisitorvertex.h"

#include "nozzlemanager.h"

#include "parser.h"

#include "raytracer.h"

nozzleManager* nozzleManager::_instance = 0;
raytracer* raytracer::_instance = 0;

class SprayPlugin : public coVRPlugin, public ui::Owner
{
private:
    osg::Group *scene;
    osg::Geode* floorGeode;
    osg::Geode* testBoxGeode;

    osg::MatrixTransform *baseTransform;

    class nozzle* editNozzle;

    int nozzleID = 0;
    int currentNozzleID = -1;
    std::list<int> idGeo;

    osg::Vec4 newColor = osg::Vec4(1,1,1,1);

    ui::Menu* sprayMenu_ = nullptr;
    ui::Menu* tempMenu = nullptr;
    ui::Menu* nozzleEditMenu_ = nullptr;
    ui::EditField* currentNozzle_ = nullptr;
    ui::Button* sprayStart_ = nullptr;
    ui::Action* edit_ = nullptr;
    ui::Action* save_ = nullptr;
    ui::Action* load_ = nullptr;
    ui::Action* create_ = nullptr;
    ui::Action* remove_ = nullptr;
    ui::Label* numField = nullptr;
    ui::EditField* pathNameFielddyn_ = nullptr;
    ui::EditField* fileNameFielddyn_ = nullptr;
    ui::EditField* nozzleNameFielddyn_ = nullptr;
    ui::EditField* newGenCreate_ = nullptr;
    std::string pathNameField_ = "";
    std::string fileNameField_ = "";
    std::string nozzleNameField_ = "";
    float sprayAngle_ = 0;
    std::string decoy_ = "";

    ui::EditField* red_ = nullptr;
    ui::EditField* green_ = nullptr;
    ui::EditField* blue_ = nullptr;
    ui::EditField* alpha_ = nullptr;
    ui::Slider* pressureSlider_ = nullptr;
    float scaleValue_ = 1;
    ui::Action* acceptEdit_ = nullptr;

    bool sprayStart = false;
    bool creating = false;
    bool editing = false;
    bool TESTING = true;

//#if TESTING

    ui::Menu* testMenu = nullptr;
    osg::Matrix memMat;
    osg::Vec3 transMat = osg::Vec3(0,0,0);
//    ui::Action* rotXneg = nullptr;
//    ui::Action* rotXpos = nullptr;
//    ui::Action* rotYneg = nullptr;
//    ui::Action* rotYpos = nullptr;
//    ui::Action* rotZneg = nullptr;
//    ui::Action* rotZpos = nullptr;
    ui::Slider* rotX = nullptr;
    ui::Slider* rotY = nullptr;
    ui::Slider* rotZ = nullptr;
    ui::EditField* moveX = nullptr;
    ui::EditField* moveY = nullptr;
    ui::EditField* moveZ = nullptr;


//#endif


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
