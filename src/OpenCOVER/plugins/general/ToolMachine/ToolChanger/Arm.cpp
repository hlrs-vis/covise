#include "Arm.h"
#include "Utility.h"

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osgAnimation/UpdateMatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/Geode>
#include <osg/Material>

#include <functional>
#include <iostream>
#include <cassert>

#include <cover/coVRPluginSupport.h>

using namespace opencover;

osg::MatrixTransform* createSphere(const osg::Vec3& pos, float radius)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(), radius)));
    osg::ref_ptr<osg::Material> material = new osg::Material;
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 0, 0, 1));
    geode->getOrCreateStateSet()->setAttribute(material);
    osg::MatrixTransform* mt = new osg::MatrixTransform;
    mt->setMatrix(osg::Matrix::translate(pos));
    mt->addChild(geode);
    static size_t count = 0;
    mt->setName("sphere" + std::to_string(count++));
    return mt;
}

void printChildren(const osg::Node *node, std::string path = "")
{

    auto group = node->asGroup();
    if(node->asTransform())
        std::cerr << "transform at " << path  << node->getName() << std::endl;

    if(!group)
        return;
    for (size_t i = 0; i < group->getNumChildren() ; i++)
    {
        printChildren(group->getChild(i), path + node->getName() + "/");
    }
    
}

osg::Matrix calculatePath(int id, float offset)
{

    auto abs = (id + offset) * pathDistance;

    const std::array<std::function<osg::Vec3(float)>, 4> segments = {
        [](float rel) { return osg::Vec3(-pathRadius * sin(rel * osg::PI), -pathRadius + pathRadius * cos(rel * osg::PI), 0); },
        [](float rel) { return osg::Vec3(rel * pathSideLength, 0, 0); },
        [](float rel) { return osg::Vec3(pathRadius * sin(rel * osg::PI), pathRadius - pathRadius * cos(rel * osg::PI), 0); },
        [](float rel) { return osg::Vec3(-rel * pathSideLength, 0, 0); }
    };

    const std::array<std::function<osg::Vec3(float)>, 4> segmentsNormals = {
        [](float rel) { return osg::Vec3(-sin(rel * osg::PI), cos(rel * osg::PI), 0); },
        [](float rel) { return osg::Vec3(0, -1, 0); },
        [](float rel) { return osg::Vec3(-sin(rel * osg::PI), -cos(rel * osg::PI), 0); },
        [](float rel) { return osg::Vec3(0, 1, 0); }
    };
    std::array<float, 4> distanceInSection = {0, 0, 0, 0};
    float pos = 0;
    for (size_t i = 0; i < 4; i++)
    {
        pos += parts[i];
        if(abs <= pos)
        {
            distanceInSection[i] = abs - (pos - parts[i]); 
            break;
        }else
        {
            distanceInSection[i] = parts[i];
        } 
    }
    osg::Vec3 posVec{0,0,0};
    osg::Vec3 ortho = segmentsNormals[0](0); //start Value for first element
    size_t secNum = 0;
    for (size_t i = 0; i < 4; i++)
    {
        if(distanceInSection[i] == 0)
        break;

        ortho = segmentsNormals[i](distanceInSection[i] / parts[i]);
        posVec += segments[i](distanceInSection[i] / parts[i]);
        secNum = i;
    }
    ortho.normalize();
    osg::Matrix posMat;
    
    posMat.makeTranslate(posVec);
    osg::Quat q;
    auto angle = acos(ortho * osg::Vec3(0, 1, 0));
    if(secNum==0)
        angle *= -1;
    q.makeRotate(angle, osg::Vec3{0, 0, -1});
    posMat.setRotate(q);
    return posMat;
}

Arm::Arm(osg::ref_ptr<osg::Node> model, osg::Group* parent, int id, const osg::Vec4& color)
: m_model(static_cast<osg::Node*>(model->clone(osg::CopyOp::DEEP_COPY_ALL)))
, m_transform (new osg::MatrixTransform)
, m_id(id)
, m_tool(std::make_unique<ToolModel>(m_model))
{
    m_transform->addChild(m_model);
    parent->addChild(m_transform);
    position(0);

    m_model->accept(m_animation);
    auto &animations = m_animation.m_am->getAnimationList();
    // for(auto &a : m_animation.m_am->getAnimationList())
    // {
    //     std::cerr << "arm " << id << " has animation " << a->getName() << std::endl;
    // }
    // applyColor(color);
    m_toolParent = m_tool->parent();

    // if(id == 0)
    // {
    //     BoneParser boneParser;
    //     m_swapSocket1->accept(boneParser);
    //     auto sphere = createSphere({0,0,0}, 10);
    //     m_swapSocket1->addChild(sphere);
    //     // m_model->accept(boneParser);
    // }
}

void Arm::position(float offset)
{
    m_transform->setMatrix(calculatePath(m_id, offset));
    m_distance = m_id + offset;
}

void Arm::play()
{
    auto animation = m_animation.m_am->getAnimationList().front();
    animation->setPlayMode(osgAnimation::Animation::ONCE);
    m_animation.m_am->playAnimation(animation, 1, 1); // Play once
    m_playtime = m_intPlayTime;
    
}

float Arm::getDistance() const
{
    return m_distance;
}

int Arm::id() const
{
    return m_id;
}

ToolModel *Arm::tool()
{
    return m_tool.get();
}

void Arm::giveTool(ToolModel::ptr &&tool)
{
    m_tool = std::move(tool);
    m_tool->setParent(m_toolParent);
}
ToolModel::ptr Arm::takeTool()
{
    auto tool = std::move(m_tool);
    m_tool = nullptr;
    return tool;
}

bool Arm::isPlaying() const
{
    return m_animation.m_am->isPlaying(m_animation.m_am->getAnimationList().front()->getName());;
}

void swapParent(osg::Group *newParent, const std::vector<osg::MatrixTransform*> &children)
{
    for(auto c : children)
    {
        auto oldParent = c->getParent(0);
        oldParent->removeChild(c);
        newParent->addChild(c);
    }
}

void Arm::update(){
    
}

void Arm::applyColor(const osg::Vec4& color)
{
    osg::ref_ptr<osg::Material> material = new osg::Material;
    material->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    material->setShininess(osg::Material::FRONT_AND_BACK, 0.5);

    osg::StateSet* stateSet = m_model->getOrCreateStateSet();
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    stateSet = findTransform(m_model, "Werkzeug")->getOrCreateStateSet();
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
}