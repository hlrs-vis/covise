#include "ToolChanger.h"

#include "Arm.h"
#include "Changer.h"
#include "Tool.h"

#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osgAnimation/Animation>
#include <osgAnimation/BasicAnimationManager>
#include <osgAnimation/RigGeometry>
#include <osgAnimation/RigTransformHardware>
#include <osgAnimation/Skeleton>
#include <osgAnimation/Skeleton>
#include <osgAnimation/StackedRotateAxisElement>
#include <osgAnimation/StackedScaleElement>
#include <osgAnimation/UpdateBone>
#include <osgDB/ReadFile>
#include <osgGA/GUIEventHandler>

#include <cover/coVRPluginSupport.h>
#include <cover/ui/Slider.h>
#include <OpenConfig/array.h>

#include <cassert>
#include <chrono>
#include <thread>

#include <PluginUtil/coColorMap.h>

using namespace opencover;

const float fbxScale = 0.001;

void applyColor(osg::Node* node, const osg::Vec4& color)
{
    osg::ref_ptr<osg::Material> material = new osg::Material;
    material->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    material->setShininess(osg::Material::FRONT_AND_BACK, 0.5);
    
    osg::StateSet* stateSet = node->getOrCreateStateSet();
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);
    node->setStateSet(stateSet);
}

ToolChanger::ToolChanger(opencover::ui::Menu *menu, opencover::config::File *file, ToolChangerNode *node)
: m_toolChangerNode(node)
, m_menu(menu)
, m_configFile(file)
{}

ToolChanger::~ToolChanger() = default;

void ToolChanger::init()
{
    if(!m_toolChangerNode->allInitialized())
        return;
    osg::ref_ptr model = osgDB::readNodeFile(m_toolChangerNode->arm->get()); 

    m_trans = new osg::MatrixTransform;
    m_trans->setName("ToolChanger");

    //position toolChanger 
    osg::Matrix m = osg::Matrix::translate(-1, 0, 0.8);
    m = osg::Matrix::rotate(-osg::PI_2, osg::Vec3(1, 0, 0)) * m;
    m = osg::Matrix::rotate(-osg::PI_2, osg::Vec3(0, 1, 0)) * m;
    m = osg::Matrix::scale(fbxScale, fbxScale, fbxScale) * m;
    osg::Matrix finalCorrection = osg::Matrix::translate(-19.4889, -45, 23.151);
    m = finalCorrection * m;
    m_trans->setMatrix(m);


    m_anim = new ui::EditField(m_menu, "selectTool");
    m_anim->setValue("55");
    // m_anim->setCallback([this](const std::string &value){
    //     m_arms[m_anim->number()]->play();
    // });
    m_maxSpeed = new ui::Slider(m_menu, "speed");
    m_maxSpeed->setBounds(-1, 1);
    m_maxSpeed->setValue(0.5);

    m_action = new ui::Action(m_menu, "changeTool");
    m_action->setCallback([this](){
        m_changeTool = true;
        m_selectedArm = m_arms[m_anim->number()].get();
        m_distanceToSeletedArm = m_selectedArm->getDistance();
    });

    auto changer = osgDB::readNodeFile(m_toolChangerNode->changer->get());
    m_changer = std::make_unique<Changer>(changer, m_trans); 
    m_changeDuration = m_changer->getAnimationDuration();

    auto coverNode = osgDB::readNodeFile(m_toolChangerNode->cover->get());
    m_trans->addChild(coverNode); 
    cover->getObjectsRoot()->addChild(m_trans);

    auto materials = m_configFile->array<std::string>("materials", "materials");
    for(auto &m : materials->value())
    {
        auto color = m_configFile->array<double>("materials", m);
        if(color->size() != 4)
        {
            std::cerr << "color array has to have 4 elements" << std::endl;
            continue;
        }
        auto node = findTransform(model, m);
        if(!node)
            node = findTransform(changer, m);
        if (!node)
            node = findTransform(coverNode, m);
        if(node)
        {
            applyColor(node, osg::Vec4(color->value()[0], color->value()[1], color->value()[2], color->value()[3]));
        }
    }
    for (size_t i = 0; i < numArms; i++)
    {
        m_arms.push_back(std::make_unique<Arm>(model, m_trans, i));
    }
    m_toolHead = toOsg(m_toolChangerNode->toolHead->get());

    m_initialized = true;
}

void ToolChanger::update()
{
    if(!m_initialized)
        init();
    if(!m_initialized)
        return;
    if(m_playingArm)
    {
        changeTools();
        return;
    }

    if(!m_selectedArm)
        return;
    

    if(m_selectedArm->getDistance() <= 1)
    {
        positionArms(numArms - m_selectedArm->id());
        m_selectedArm->play();
        m_changer->play();
        m_playingArm = m_selectedArm;
        m_selectedArm = nullptr;
        m_speed = 0;
        decellerate = false;
        std::cerr << "position reached " << std::endl;
        m_changeTime = m_changeDuration;
        m_doorAnimationTime = m_doorAnimationDuration;
        return;
    }
    auto maxSpeed = m_maxSpeed->value() * cover->frameDuration();
    //decelerate
    auto a = acceletation *  cover->frameDuration();
    if(m_speed * a / 2 > numArms - m_selectedArm->getDistance()) //fix distance juming when becoming > numArms
    {
        m_speed -= maxSpeed / a;
        std::cerr << "decellerating " << m_speed << std::endl;
    }
    else if(m_speed < maxSpeed)
        m_speed += maxSpeed  / a;
    // if(!decellerate && m_speed > 0 && numFramesUntilStop(m_speed, m_selectedArm->getDistance()) <= acceletation)
    // {
    //     decellerate = true;
    //     std::cerr << "start decellerating" << std::endl;
    // }
    // if(decellerate)
    // {
    //     m_speed -= m_maxSpeed->value() / acceletation;
    //     std::cerr << "decellerating " << m_speed << std::endl;
    // }
    // else if(m_speed < m_maxSpeed->value())
    //     m_speed += m_maxSpeed->value() / acceletation;

    m_offset += m_speed ;
    if(m_maxSpeed->value() == 0)
        return;
    if(m_offset > numArms)
        m_offset = 0;
    if(m_offset < 0)
        m_offset = numArms;
    positionArms(m_offset);
    

}

void ToolChanger::positionArms(float offset)
{
    for (size_t i = 0; i < numArms; i++)
    {
        auto offset = m_offset;
        if(offset + i>= numArms)
            offset -= numArms;
        m_arms[i]->position(offset);
    }
}

bool ToolChanger::animationStepReached(AnimationState state)
{
    return m_changeTime <= m_changeDuration * (1 - (m_animationStateTime[static_cast<int>(state)] / 100));
}

void ToolChanger::changeTools(){
    m_changeTime -= cover->frameDuration();
    switch (m_animationState)
    {
    case AnimationState::BeforeSwap:
    {
        if(animationStepReached(m_animationState))
        {
            auto tool = m_playingArm->takeTool();
            if(tool)
                m_changer->giveTool(std::move(tool), Changer::End::front);
            
            if(m_toolHeadTool)
            {
                m_toolHeadTool->model()->setMatrix(m_toolHeadMatrix);
                m_changer->giveTool(std::move(m_toolHeadTool), Changer::End::back);
            }

            
            m_animationState = AnimationState::Swapping;

        }
    }
    break;
    case AnimationState::Swapping:
    {
        if(animationStepReached(m_animationState))
        {
            std::cerr << "swapping" << std::endl;
            auto tool = m_changer->takeTool(Changer::End::back);
            //m_toolHEad->takeTool();
            if(tool)
                m_playingArm->giveTool(std::move(tool));
            
            tool = m_changer->takeTool(Changer::End::front);
            if(tool)
            {
                m_toolHeadTool = std::move(tool);
                m_toolHeadMatrix = m_toolHeadTool->model()->getMatrix();
                auto m = m_toolHeadMatrix;
                m = osg::Matrix::scale(fbxScale, fbxScale, fbxScale) * m;
                m = osg::Matrix::rotate(osg::PI, osg::Vec3(0, 1, 0)) * m;
                m.setTrans(0,-0.04,0);
                m_toolHeadTool->model()->setMatrix(m);
                m_toolHeadTool->setParent(m_toolHead);
            }
            m_animationState = AnimationState::AfterSwap;

        }
    }
    break;
    case AnimationState::AfterSwap:
    {
        if(animationStepReached(m_animationState))
        {
            std::cerr << "after swap" << std::endl;
            swapTools();
            
            m_playingArm = nullptr;
            m_animationState = AnimationState::BeforeSwap;
            m_doorAnimationTime = -m_doorAnimationDuration;

        }
    }
    break;
    
    default:
        break;
    }
}

void ToolChanger::swapTools(){
    auto t1 = m_changer->takeTool(Changer::End::front);
    auto t2 = m_changer->takeTool(Changer::End::back);
    if(t1)
        m_changer->giveTool(std::move(t1), Changer::End::back);
    if(t2)
        m_changer->giveTool(std::move(t2), Changer::End::front);
}
