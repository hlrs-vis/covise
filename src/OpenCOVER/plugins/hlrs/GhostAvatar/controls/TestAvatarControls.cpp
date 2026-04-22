#include "TestAvatarControls.h"

TestAvatarControls::TestAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName)
    : GhostAvatarControls(pathToFbx, armNodeName, headNodeName)
{
    setBaseRotation(osg::Quat(0, 0, 0.707107, 0.707107)); // make sure the avatar is facing forward

    setForwardDirection({ 1.0, 0.0, 0.0 });
    setUpDirection({ 0.0, 0.0, 1.0 });

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

osg::Vec3 TestAvatarControls::getEyeOffset() const
{
    auto bounds = getBounds();
    return { 0, 0, 0.7f * getBounds(2) };
}

void TestAvatarControls::updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix)
{
    auto targetHeight = headMatrix.getTrans().z() - floorMatrix.getTrans().z();
    float scale = targetHeight / (getInitialBounds())[1];

    m_avatarTrans->setMatrix(osg::Matrix::scale(scale, scale, scale) *
                             osg::Matrix::rotate(getBaseRotation()) * osg::Matrix::rotate(headMatrix.getRotate()) * 
                             osg::Matrix::translate(floorMatrix.getTrans()));

    if (m_armBone)
        makeBonePointAtTarget(*m_armBone, handMatrix.getTrans(), m_armAdjustMatrix, m_armBaseVector);
}