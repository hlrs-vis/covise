#ifdef HAVE_FBX

#ifndef ANIMATED_AVATAR_PLUGIN_FBX_PROVIDER_H
#define ANIMATED_AVATAR_PLUGIN_FBX_PROVIDER_H

#include "ModelProvider.h"

#include <osg/NodeVisitor>

#include <osgAnimation/BasicAnimationManager>
#include <osgAnimation/RigTransformHardware>
#include <osgAnimation/Animation>
#include <osgAnimation/Skeleton>
#include <osgGA/GUIEventHandler>
#include <osgAnimation/StackedRotateAxisElement>


struct BoneFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osgAnimation::Bone> bone;
    BoneFinder(const std::string &name);
    void apply(osg::Node& node) override;
private:
    std::string m_nodeName;
};

struct AnimationManagerFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osgAnimation::BasicAnimationManager> m_am;
    AnimationManagerFinder();
    void apply(osg::Node& node) override;
};

class FbxProvider : public ModelProvider
{
public:
    osg::Node *loadModel(const std::string &filename) override;
    void m_playAnimation(Animation animation, float weight, float delay) override;
private:
    osgAnimation::AnimationList m_animations;
    AnimationManagerFinder m_animationFinder;
    osg::ref_ptr<osgAnimation::StackedRotateAxisElement> m_rotation;
    osg::ref_ptr<osgAnimation::Bone> m_bone;
    bool m_rotateButtonPresed = false;

};


#endif // ANIMATED_AVATAR_PLUGIN_FBX_PROVIDER_H
#endif // HAVE_FBX
