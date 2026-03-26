#include "PlanarAvatarControls.h"

PlanarAvatarControls::PlanarAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName)
    : GhostAvatarControls(pathToFbx, armNodeName, headNodeName)
{
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

void PlanarAvatarControls::updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix)
{
    m_avatarTrans->setMatrix(floorMatrix);

    if (m_armBone)
        moveBoneToTarget(*m_armBone, handMatrix.getTrans(), m_armAdjustMatrix);

    if (m_headBone)
        makeBonePointAtTarget(*m_headBone, headMatrix.getTrans(), m_headAdjustMatrix, m_headBaseVector);
}