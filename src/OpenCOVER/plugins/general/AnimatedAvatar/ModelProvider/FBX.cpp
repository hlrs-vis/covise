


#ifdef HAVE_FBX
#include "FBX.h"

#include <osgDB/ReadFile>
#include <array>

#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Action.h>
#include <cover/coVRPluginSupport.h>
#include <osgAnimation/Skeleton>
#include <osgAnimation/RigGeometry>
#include <osgAnimation/UpdateBone>

using namespace opencover;

osgAnimation::Channel* createChannel( const char* name,
 const osg::Vec3& axis, float rad )
{
 osg::ref_ptr<osgAnimation::QuatSphericalLinearChannel> ch =
 new osgAnimation::QuatSphericalLinearChannel;
 ch->setName( "quaternion" );
 ch->setTargetName( name );
 osgAnimation::QuatKeyframeContainer* kfs =
 ch->getOrCreateSampler()->getOrCreateKeyframeContainer();
 kfs->push_back( osgAnimation::QuatKeyframe(
 0.0, osg::Quat(0.0, axis)) );
 kfs->push_back( osgAnimation::QuatKeyframe(
 8.0, osg::Quat(rad, axis)) );
 return ch.release();
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

BoneFinder::BoneFinder(const std::string &name) 
: osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
, m_nodeName(name) {}

void BoneFinder::apply(osg::Node& node) {

    std::cerr << "nodename " << node.getName();
    node.getUpdateCallback();
    if(dynamic_cast<osgAnimation::Skeleton*>(&node))
        std::cerr << " is a skeleton";
    if(dynamic_cast<osgAnimation::Bone*>(&node))
        std::cerr << " is a Bone";
    if(dynamic_cast<osgAnimation::RigGeometry*>(&node))
        std::cerr << " is a RigGeometry";
    std::cerr << std::endl;
    if(node.getName() == m_nodeName)
    {
        bone = dynamic_cast<osgAnimation::Bone*>(&node);
        return;
    }
    traverse(node);
}

osg::Node *FbxProvider::loadModel(const std::string &filename)
{
    osg::MatrixTransform *trans = new osg::MatrixTransform;
    auto scale = osg::Matrix::scale(osg::Vec3f(10, 10, 10));
    auto rot1 = osg::Matrix::rotate(osg::PI / 2, 1,0,0);
    auto rot2 = osg::Matrix::rotate(osg::PI, 0,0,1);
    auto transM = osg::Matrix::translate(osg::Vec3f{0,0,-1300});
    trans->setMatrix(scale * rot1 * rot2 * transM);
    auto model = osgDB::readNodeFile(filename);  
    if(!model)
        return nullptr;
    model->accept(m_animationFinder);
    m_animations = m_animationFinder.m_am->getAnimationList();
    ui::Owner * owner = new ui::Owner("fbxTest", cover->ui);
    ui::Menu *fbxMenu = new ui::Menu("fbxAvatar", owner);
    int i = 0;
    trans->addChild(model);
    std::cerr<< "loaded model " << filename << std::endl;

    BoneFinder bf("mixamorig:RightHand");
    model->accept(bf);
    m_bone = bf.bone;
    m_rotation = new osgAnimation::StackedRotateAxisElement("hand", osg::Y_AXIS, 0);
    auto cb = dynamic_cast<osgAnimation::UpdateBone*>(bf.bone->getUpdateCallback());
    if(cb)
    {
        std::cerr << " UpdateBone found " << cb->getName() <<  std::endl;
        osgAnimation::StackedTransform st = cb->getStackedTransforms();
        st.push_back(m_rotation);
    }

    osgAnimation::Animation *anim = new osgAnimation::Animation;
    anim->setPlayMode( osgAnimation::Animation::PPONG );
    anim->addChannel( createChannel("mixamorig:RightHand", osg::Y_AXIS, osg::PI_4) );
    anim->setName("hand");
    m_animationFinder.m_am->registerAnimation(anim);

    for(const auto & anim : m_animations)
    {
        std::cerr << "avatar has animation " << i++ << " " << anim->getName() << std::endl;
        anim->setPlayMode(osgAnimation::Animation::PlayMode::LOOP);
        auto slider = new ui::Slider(fbxMenu, anim->getName());
        slider->setBounds(0,1);
        slider->setCallback([this, &anim](double val, bool x){
            // m_animationFinder.m_am->playAnimation(anim, 1, val);
        });
        m_animationFinder.m_am->stopAnimation(anim);
    }

    auto moveBone = new ui::Action(fbxMenu, "move right hand");
    moveBone->setCallback([this](){
        m_rotateButtonPresed = true;
    });
    m_animationFinder.m_am->playAnimation(anim);
    return trans;
}

void FbxProvider::m_playAnimation(Animation animation, float weight, float delay)
{
    if(m_rotateButtonPresed)
    {
        m_rotateButtonPresed = false;
        m_rotation->setAngle(m_rotation->getAngle() + osg::PI_2);
        std::cerr << "updated rotation: " << m_rotation->getAngle() << std::endl;
        // m_bone->dirtyBound();
        // m_animationFinder.m_am->update(0);
    }
    
    constexpr std::array<int, (int)Animation::LAST> animationConversion = {5, 10, 6, 12, 7, 8, 9, 13}; //convert from enum to the model specific animations of the used test avatar
    return;
    std::cerr << "trying to start animation " << m_animations[animationConversion[(int)animation]]->getName() << "  "  << (int)animation << "/" << m_animations.size() << std::endl;
    m_animationFinder.m_am->stopAnimation(m_animations[animationConversion[(int)currentAnimation()]]);
    m_animationFinder.m_am->playAnimation(m_animations[animationConversion[(int)animation]]);
}

#endif