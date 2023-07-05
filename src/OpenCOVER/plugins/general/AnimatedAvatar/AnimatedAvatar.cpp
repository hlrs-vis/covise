#include "AnimatedAvatar.h"


#include <osg/MatrixTransform>
#include <cover/coVRPluginSupport.h>
#include <cover/VRAvatar.h>

using namespace opencover;

AnimatedAvatar::AnimatedAvatar(const osg::ref_ptr<osgCal::CoreModel> &model, int partnerId)
: m_partnerId(partnerId)
, m_transform(new osg::MatrixTransform)
, m_model(new osgCal::Model)
{
    m_partner = coVRPartnerList::instance()->get(partnerId);
    m_partner->getAvatar()->headTransform->addChild(m_transform);
    m_model->load(model);
    m_transform->addChild(m_model);
    cover->getObjectsRoot()->addChild(m_partner->getAvatar()->headTransform);

    auto scale = osg::Matrix::scale(osg::Vec3f(10, 10, 10));
    auto rot = osg::Matrix::rotate(osg::PI, 0,0,1);
    auto trans = osg::Matrix::translate(osg::Vec3f{0,0,-100});
    m_transform->setMatrix(trans * scale * rot);

    auto animNames = model->getAnimationNames();
    for(const auto &animName : animNames)
        std::cerr << "animeName: " << animName << std::endl;
    m_model->blendCycle(0, 1, 0);
}

auto animations = {"skeleton_idle", "skeleton_strut", "skeleton_walk", "skeleton_jog", "skeleton_wave", "skeleton_shoot_arrow", "skeleton_hiphop"};

void AnimatedAvatar::update()
{
    auto headTrans = m_partner->getAvatar()->headTransform->getMatrix();
    auto handTrans = m_partner->getAvatar()->handTransform->getMatrix();
    
    auto headDiff = headTrans.getTrans() - m_lastHeadPosition.getTrans();
    auto headDiffLenght = headDiff.length();
    std::cerr << "headDiffLenght " << headDiffLenght << std::endl;
    const float walkThreashhold = 100;
    const float runThreashhold = 2 * walkThreashhold;
    if(headDiffLenght < walkThreashhold)
    {
       setExclusiveAnimation(0); 
    }
    else if(headDiffLenght < runThreashhold)
    {
        setExclusiveAnimation(2);
    }
    else
    {
        setExclusiveAnimation(3);
    }
    if (headTrans.getTrans().z() < handTrans.getTrans().z())
    {
        setExclusiveAnimation(4);
    }

    m_lastHandPosition = handTrans;
    m_lastHeadPosition = headTrans;

}

void AnimatedAvatar::setExclusiveAnimation(int animationsIndex)
{
    for (size_t i = 0; i < animations.size(); i++)
    {
        if(i != animationsIndex)
        {
            m_model->clearCycle(i, 0);
        }
    }
    m_model->blendCycle(animationsIndex,1,0);
}

AnimatedAvatar::~AnimatedAvatar()
{
    cover->getObjectsRoot()->removeChild(m_model);
}