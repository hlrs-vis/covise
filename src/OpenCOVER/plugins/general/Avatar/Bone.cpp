#include "Bone.h"
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>

#include <osgAnimation/UpdateBone>
#include <array>

using namespace opencover;

BoneFinder::BoneFinder(const std::string &name) 
: osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
, m_nodeName(name) {}


void BoneFinder::apply(osg::Node& node) {

    // std::cerr << "nodename " << node.getName();
    node.getUpdateCallback();
    // if(dynamic_cast<osgAnimation::Skeleton*>(&node))
    //     std::cerr << " is a skeleton";
    // if(dynamic_cast<osgAnimation::Bone*>(&node))
    //     std::cerr << " is a Bone";
    // if(dynamic_cast<osgAnimation::RigGeometry*>(&node))
    //     std::cerr << " is a RigGeometry";
    // std::cerr << std::endl;
    if(node.getName() == m_nodeName)
    {
        bone = dynamic_cast<osgAnimation::Bone*>(&node);
        return;
    }
    traverse(node);
}

AnimationManagerFinder::AnimationManagerFinder() 
: osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN) {}

void AnimationManagerFinder::apply(osg::Node& node) {

    if (m_am.valid())
        return;

    if (node.getUpdateCallback()) {       
        m_am = dynamic_cast<osgAnimation::BasicAnimationManager*>(node.getUpdateCallback());
        return;
    }
    
    traverse(node);
}

float getBoneLenght(osgAnimation::UpdateBone *bone)
{
    if(bone)
    {
        for (auto i = bone->getStackedTransforms().begin(); i < bone->getStackedTransforms().end(); i++)
        {
            if((*i)->getName() == "translate")
            {
                auto translate = dynamic_cast<osgAnimation::StackedTranslateElement*>(i->get());
                auto t = translate->getTranslate();
                return t.length();
            }
        }
    }
    return 0;
}

void removeOrientationAndScale(osgAnimation::UpdateBone *bone)
{
    for (auto i = bone->getStackedTransforms().begin(); i < bone->getStackedTransforms().end(); i++)
    {
        if((*i)->getName() != "translate")
        {
            i = bone->getStackedTransforms().erase(i);
        }
    }
}

void replaceAnimationRotation(osgAnimation::Bone *bone, osgAnimation::StackedQuaternionElement* rotationToUse)
{
    auto cb = dynamic_cast<osgAnimation::UpdateBone*>(bone->getUpdateCallback());
    if(cb)
    {
        removeOrientationAndScale(cb);
        cb->getStackedTransforms().push_back(rotationToUse);
    }
}

Bone::Bone(Bone *parent, const std::string &name, osg::Node* model, ui::Menu *menu)
:rotation(new osgAnimation::StackedQuaternionElement("quaternion", osg::Quat(0, osg::Y_AXIS)))
, parent(parent)
, name(name)
{
    BoneFinder bf(name);
    model->accept(bf);
    bone = bf.bone;
    replaceAnimationRotation(bone, rotation);
    length = getBoneLenght(dynamic_cast<osgAnimation::UpdateBone*>(bone->getChild(0)->getUpdateCallback()));
}