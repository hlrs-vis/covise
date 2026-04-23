#include "PlanarAvatarControls.h"

PlanarAvatarControls::PlanarAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName)
    : GhostAvatarControls(pathToFbx, armNodeName, headNodeName)
{
    setBaseRotation(osg::Quat(1, 0, 0, 0)); // make sure avatar is upright

    setForwardDirection({ 0, -1, 0 });
    setUpDirection({ 0, 0, -1 });

    m_armAdjustMatrix.set(
        0, 0, 1, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1);

    m_headAdjustMatrix.set(
        1, 0, 0, 0,
        0, 0, -1, 0,
        0, 1, 0, 0,
        0, 0, 0, 1);
}

osg::Vec3 PlanarAvatarControls::getEyeOffset() const
{
    return { 0, 0, 0.8f * getBounds(2) };
}

void PlanarAvatarControls::updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix)
{
    auto targetHeight = headMatrix.getTrans().z() - floorMatrix.getTrans().z();
    float scale = targetHeight / (getInitialBounds())[1];

    auto headForward = getForwardDirection() * osg::Matrix::rotate(headMatrix.getRotate());
    auto up = getUpDirection();
    up.normalize();

    // Evaluate turn-around by yaw only: ignore head pitch/roll.
    headForward -= up * (headForward * up);
    bool flipBody = false;
    if (headForward.length2() > 1e-8)
    {
        headForward.normalize();
        const auto refForward = getForwardDirection();
        flipBody = (headForward * refForward) < 0.0f;
    }

    osg::Quat bodyFlip;
    if (flipBody)
    {
        constexpr double pi = 3.14159265358979323846;
        bodyFlip.makeRotate(pi, up);
    }
    else
    {
        bodyFlip.makeRotate(0.0, up);
    }

    m_avatarTrans->setMatrix(osg::Matrix::scale(scale, scale, scale) *
                             osg::Matrix::rotate(getBaseRotation()) *
                             osg::Matrix::rotate(bodyFlip) *
                             osg::Matrix::translate(floorMatrix.getTrans()));

/*     m_avatarTrans->setMatrix(osg::Matrix::scale(scale, scale, scale) *
                             osg::Matrix::rotate(getBaseRotation()) *
                             osg::Matrix::translate(floorMatrix.getTrans())); */

    // TODO: the adjust matrix should be part of the bone and not have to be passed to these methods
    if (m_armBone)
        moveBoneToTarget(*m_armBone, handMatrix.getTrans(), m_armAdjustMatrix);

    if (m_headBone)
    {
        // Compensate the body flip so the head doesn't over-rotate
        osg::Quat headRot = flipBody ? bodyFlip.inverse() * headMatrix.getRotate() : headMatrix.getRotate();
        rotateBone(*m_headBone, headRot, m_headAdjustMatrix);
    }
}