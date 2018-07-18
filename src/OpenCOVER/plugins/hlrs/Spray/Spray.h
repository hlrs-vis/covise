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
#include "../../../cover/ui/SelectionList.h"
#include "../../../cover/coVRPluginSupport.h"

using namespace covise;
using namespace opencover;


#include "string.h"



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

    int nozzleID = -1;
    int currentNozzleID = -1;
    std::list<int> idGeo;

    osg::Vec4 newColor = osg::Vec4(1,1,1,1);
    osg::Vec3 newBoundingBox = osg::Vec3(2000,2000,2000);

    ui::Menu* sprayMenu_ = nullptr;
    ui::Menu* nozzleCreateMenu = nullptr;
    ui::Menu* nozzleCreateMenuStandard = nullptr;
    ui::Menu* nozzleCreateMenuImage = nullptr;
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

    ui::SelectionList* nozzleIDL = nullptr;

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
