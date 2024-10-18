#ifndef TOOLMACHINETOOLCHANGER_H
#define TOOLMACHINETOOLCHANGER_H

#include "Utility.h"
#include "VrmlNode.h"

#include <cover/ui/Menu.h>
#include <cover/ui/VectorEditField.h>

#include <cover/ui/Slider.h>
#include <cover/ui/Action.h>

#include <osg/Node>
#include <osg/ref_ptr>
#include <osg/MatrixTransform>

#include <OpenConfig/file.h>

#include <memory>

class Arm;
class Changer;
class ToolModel;

struct ToolChangerFiles{
    const std::string arm, swapArm, cover;
};

class ToolChanger : public LogicInterface
{
public:
    ToolChanger(opencover::ui::Menu *menu, opencover::config::File *file, ToolChangerNode *node);
    ~ToolChanger(); //do not implement destructor in header because it can't delete m_animationFinder
    void update();
private:

    enum AnimationState{ BeforeSwap, Swapping, AfterSwap, LAST} m_animationState = BeforeSwap;
    void positionArms(float offset);
    void changeTools();
    bool animationStepReached(AnimationState state);
    void swapTools();
    void init();

    opencover::ui::Menu *m_menu = nullptr;
    ToolChangerNode *m_toolChangerNode = nullptr;
    bool m_initialized = false;
    std::vector<std::unique_ptr<Arm>> m_arms;
    bool m_update = false;
    int m_currentArm = 0;
    opencover::ui::EditField *m_anim;
    opencover::ui::Slider *m_maxSpeed;
    float m_speed = 0;
    float m_offset = 0;
    opencover::ui::Action *m_action;
    bool m_changeTool = false;
    Arm *m_selectedArm = nullptr, *m_playingArm = nullptr;
    float m_distanceToSeletedArm = 0;
    bool decellerate = false;
    std::unique_ptr<Changer> m_changer;
    osg::ref_ptr<osg::MatrixTransform> m_trans;
    float m_changeDuration = 0; //total duration of the tool change animation
    float m_changeTime = 0; // left duration of the tool change animation
    //in % of the total animation time
    const std::array<float, static_cast<int>(AnimationState::LAST)> m_animationStateTime {20, 80, 100};
    osg::MatrixTransform *m_toolHead = nullptr;
    std::unique_ptr<ToolModel> m_toolHeadTool;
    osg::Matrix m_toolHeadMatrix;
    osg::MatrixTransform *m_door = nullptr;
    osg::Matrix m_doorTransform;
    const float m_doorAnimationDuration = 0.5;
    float m_doorAnimationTime = 0;
    const osg::Vec3 m_doorOffset = {0, 0, -0.5};
    opencover::config::File *m_configFile = nullptr;
};


#endif // TOOLMACHINETOOLCHANGER_H
