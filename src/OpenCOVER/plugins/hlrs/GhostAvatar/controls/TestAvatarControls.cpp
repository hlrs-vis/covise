#include "TestAvatarControls.h"

TestAvatarControls::TestAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName)
    : GhostAvatarControls(pathToFbx, armNodeName, headNodeName)
{
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

void TestAvatarControls::updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix)
{
    m_avatarTrans->setMatrix(floorMatrix);
    osg::Matrix translationMatrix;
    translationMatrix.makeTranslate(floorMatrix.getTrans());
    osg::Matrix rotationMatrix;
    rotationMatrix.makeRotate(floorMatrix.getRotate());

    auto targetHeight = headMatrix.getTrans().z() - floorMatrix.getTrans().z();

    float scale = targetHeight / m_initialHeight;

    // scale matrix
    osg::Matrix scaleMatrix;
    scaleMatrix.makeScale(scale, scale, scale);

    m_avatarTrans->setMatrix(scaleMatrix * rotationMatrix * translationMatrix);

    if (m_armBone)
        makeBonePointAtTarget(*m_armBone, handMatrix.getTrans(), m_armAdjustMatrix, m_armBaseVector);
}