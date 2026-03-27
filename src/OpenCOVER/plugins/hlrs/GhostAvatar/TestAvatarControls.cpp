#include "TestAvatarControls.h"

TestAvatarControls::TestAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName)
    : GhostAvatarControls(pathToFbx, armNodeName, headNodeName)
{
    if (m_armNodeName == "LeftArm")
    {
        m_armAdjustMatrix.set(
            1, 0, 0, 0,
            0, 0, -1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1);
    }
    else if (m_armNodeName == "RightArm")
    {
        m_armAdjustMatrix.set(
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, -1, 0, 0,
            0, 0, 0, 1);
    }
}

void TestAvatarControls::updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix)
{
    m_avatarTrans->setMatrix(floorMatrix);

    if (m_armBone)
        makeBonePointAtTarget(*m_armBone, handMatrix.getTrans(), m_armAdjustMatrix, m_armBaseVector);
}