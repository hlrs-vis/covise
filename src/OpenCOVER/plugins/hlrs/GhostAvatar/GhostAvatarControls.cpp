#include <osgDB/ReadFile>
#include <cover/coVRPluginSupport.h> // includes cover

#include "GhostAvatarControls.h"

using namespace opencover;

GhostAvatarControls::GhostAvatarControls()
{
    // set correct axis conventions for the GhostAvatar model
    // TODO: don't hard code this...
    if (m_armNodeName == "LeftArm")
    {
        m_adjustMatrix.set(
            1, 0, 0, 0,
            0, 0, -1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1);
    }
    else if (m_armNodeName == "RightArm")
    {
        m_adjustMatrix.set(
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, -1, 0, 0,
            0, 0, 0, 1);
    }
    else if (m_armNodeName == "Arm")
    {
        m_adjustMatrix.set(
            0, 0, 1, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1);
    }

    if (m_headNodeName == "Head")
    {
        m_adjustMatrixHead.set(
            1, 0, 0, 0,
            0, 0, -1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1);
    }
}

void GhostAvatarControls::loadAvatar()
{
    auto model = osgDB::readNodeFile(m_pathToFbx);
    m_avatarTrans = new osg::MatrixTransform();
    m_avatarTrans->setName("AvatarTrans");
    m_avatarTrans->addChild(model);
    cover->getObjectsRoot()->addChild(m_avatarTrans);

    m_avatarTrans->accept(m_parser);
}

void GhostAvatarControls::preFrame(const osg::Matrix &m_floor, const osg::Matrix &m_hand, const osg::Matrix &m_head)
{
    m_avatarTrans->setMatrix(m_floor);

    auto armNode = m_parser.findNode(m_armNodeName);
    if (armNode != m_parser.nodeToBoneMap.end())
    {
        auto &armBoneParser = armNode->second;
        if (armBoneParser.pos)
        {
            // matrices to convert between local and world coordinates
            armLocalToWorldMat = armNode->second.parent->osgNode->getWorldMatrices(cover->getObjectsRoot())[0];
            auto worldToLocalMat = osg::Matrix::inverse(armLocalToWorldMat);

            auto localArmPos = armNode->second.initialPos;
            worldArmPos = localArmPos * armLocalToWorldMat;

            worldArmTargetPos = m_hand.getTrans();
            auto localTargetPos = worldArmTargetPos * worldToLocalMat;

            osg::Vec3 localTargetDir = localTargetPos - localArmPos;
            // apply axis-convention correction
            osg::Vec3 adjustedTargetDir = m_adjustMatrix * localTargetDir;

            // move bone to interactor position
            armBoneParser.pos->setTranslate(adjustedTargetDir);
        }
    }
    else
    {
        std::cerr << "The avatar does not seem to have a bone called " << m_armNodeName << " -> can't move its arm!\n";
    }

    auto headNode = m_parser.findNode(m_headNodeName);
    if (headNode != m_parser.nodeToBoneMap.end())
    {
        auto &headNodeParser = headNode->second;
        if (headNodeParser.rot)
        {
            // matrices to convert between local and world coordinates
            headLocalToWorldMat = headNode->second.parent->osgNode->getWorldMatrices(cover->getObjectsRoot())[0];
            auto worldToLocalMat = osg::Matrix::inverse(headLocalToWorldMat);

            auto localHeadPos = headNode->second.initialPos;
            worldHeadPos = localHeadPos * headLocalToWorldMat;

            worldHeadTargetPos = m_head.getTrans();
            auto localTargetPos = worldHeadTargetPos * worldToLocalMat;

            osg::Vec3 localTargetDir = localTargetPos - localHeadPos;
            localTargetDir.normalize();

            // apply axis-convention correction
            osg::Vec3 adjustedTargetDir = m_adjustMatrixHead * localTargetDir;

            // rotate the arm bone to point to the target
            osg::Quat rotation;
            if (m_headBaseVec[0] != 0 || m_headBaseVec[1] != 0 || m_headBaseVec[2] != 0)
            {
                rotation.makeRotate(m_headBaseVec, adjustedTargetDir);
                headNodeParser.rot->setQuaternion(rotation);
            }
        }
    }
    else
    {
        std::cerr << "The avatar does not seem to have a bone called " << m_headNodeName << " -> can't move its head!\n";
    }
}