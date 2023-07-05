#include "AnimatedAvatar.h"


#include <osg/MatrixTransform>
#include <cover/coVRPluginSupport.h>
#include <cover/VRAvatar.h>

#include "ModelProvider/Cal3d.h"
#ifdef HAVE_FBX
#include "ModelProvider/FBX.h"
#endif
#include <cmath>
using namespace opencover;

AnimatedAvatar::AnimatedAvatar(const std::string &filename, int partnerId)
: m_partnerId(partnerId)
{
    
    auto fileExtention = filename.substr(filename.find_last_of('.'));
    std::cerr << "fileExtention " << fileExtention << std::endl;
    if(fileExtention == ".cfg")
    {
        m_model = std::make_unique<Cal3dProvider>();
    }
#ifdef HAVE_FBX
    else if(fileExtention == ".fbx")
        m_model = std::make_unique<FbxProvider>();
#endif
    if(!m_model)
        return;
    m_partner = coVRPartnerList::instance()->get(partnerId);
    m_partner->getAvatar()->headTransform->addChild(m_model->loadModel(filename));
    cover->getObjectsRoot()->addChild(m_partner->getAvatar()->headTransform);
}

bool movedForward(const osg::Matrix &t1, const osg::Matrix &t2)
{
    osg::Quat q;
    auto t3 = osg::Matrix::inverse(t1);
    t3.get(q);
    osg::Vec3d carDirection = q * osg::Vec3d(1.0, 0.0, 0.0); // getting car's
    auto div = t1.getTrans() - t2.getTrans();
    if(div == osg::Vec3d())
        return false;
    //project to xy-plane
    div = osg::Vec3d{div.x(), 0, div.z()};
    carDirection = osg::Vec3d{carDirection.x(), 0, carDirection.z()};


    div.normalize();
    carDirection.normalize();
    auto angle = acos(div * carDirection);
    // std::cerr << "angle " << angle << std::endl;
    return angle > -osg::PI_2 && angle > osg::PI_2;
}

void AnimatedAvatar::update()
{
    if(!m_model)
        return;
    m_model->playAnimation(ModelProvider::Idle, 1, 0.5);
    return;
    auto fps = 1 / cover->frameDuration();
    osg::Matrix lastHeadPosition;
    bool updated = false;
    auto headTrans = m_partner->getAvatar()->headTransform->getMatrix();
    // auto handTrans = m_partner->getAvatar()->handTransform->getMatrix();
    while(m_lastHeadPositions.size() > fps)
    {
        lastHeadPosition = m_lastHeadPositions.front();
        m_lastHeadPositions.pop();
        updated = true;
    }
    if(updated)
    {

        auto headDiff = headTrans.getTrans() - lastHeadPosition.getTrans();
        auto headDiffLenght = headDiff.length();
        // std::cerr << "headDiffLenght " << headDiffLenght << std::endl;
        const float walkThreashhold = 1;
        const float runThreashhold = 4 * walkThreashhold;
        ModelProvider::Animation animation = ModelProvider::Idle;
        auto forward = movedForward(headTrans, lastHeadPosition);
        if(headDiffLenght > walkThreashhold && headDiffLenght < runThreashhold)
        {
            
            animation = forward ? ModelProvider::Walk : ModelProvider::WalkBack;
        }
        else if(headDiffLenght > runThreashhold)
        {
            animation = forward ? ModelProvider::Run : ModelProvider::RunBack;
        }
        if (headTrans.getTrans().z() < headTrans.getTrans().z())
        {
            animation = ModelProvider::Wave;
        }
        if(m_model->currentAnimation() != animation)
        {
            m_model->playAnimation(animation, 1, 0.5);
        }
        // std::cerr << "animation changeed to " << (int)animation << " headDiffLenght = " << headDiffLenght << std::endl;
        // std::cerr << headTrans.getTrans().x() << ", " << headTrans.getTrans().y() << ", " << headTrans.getTrans().z() << std::endl;
        // std::cerr << lastHeadPosition.getTrans().x() << ", " << lastHeadPosition.getTrans().y() << ", " << lastHeadPosition.getTrans().z() << std::endl;
        // std::cerr << std::endl;
    }
    // m_lastHandPositions.push(handTrans);
    m_lastHeadPositions.push(headTrans);

}

