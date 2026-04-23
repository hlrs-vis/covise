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

    m_avatarTrans->setMatrix(osg::Matrix::scale(scale, scale, scale) *
                             osg::Matrix::rotate(getBaseRotation()) *
                             osg::Matrix::translate(floorMatrix.getTrans()));

    // TODO: the adjust matrix should be part of the bone and not have to be passed to these methods
    if (m_armBone)
        moveBoneToTarget(*m_armBone, handMatrix.getTrans(), m_armAdjustMatrix);

    if (m_headBone)
        rotateBone(*m_headBone, headMatrix.getRotate(), m_headAdjustMatrix);
}