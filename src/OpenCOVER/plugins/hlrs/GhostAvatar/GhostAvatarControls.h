#ifndef COVER_PLUGIN_GHOSTAVATAR_GhostAvatarControls_H
#define COVER_PLUGIN_GHOSTAVATAR_GhostAvatarControls_H

#include <string>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osg/Vec3>
#include <osg/Vec4>

#include "BoneParser.h"

class GhostAvatarControls
{
public:
    // TODO: add filepath to constructor
    GhostAvatarControls();

    void loadAvatar();

    void preFrame(const osg::Matrix &m_floor, const osg::Matrix &m_hand, const osg::Matrix &m_head);

    // TODO: simplify and make private with getters!
    osg::Vec3 worldArmPos;
    osg::Vec3 worldArmTargetPos;
    osg::Matrix armLocalToWorldMat;

    osg::Vec3 worldHeadPos;
    osg::Vec3 worldHeadTargetPos;
    osg::Matrix headLocalToWorldMat;

    osg::Vec3 m_armBaseVec = { 0, 1, 0 };
    osg::Vec3 m_headBaseVec = { 0, 1, 0 };
    osg::Matrix m_adjustMatrix = osg::Matrix::identity();
    osg::Matrix m_adjustMatrixHead = osg::Matrix::identity();

private:
    std::string m_pathToFbx = "/data/STARTS-ECHO/Avatars/planarAvatar/PLANEE6.fbx";

    std::string m_armNodeName = "Arm"; // "LeftArm"
    std::string m_headNodeName = "Head";

    BoneParser m_parser;
    osg::MatrixTransform *m_avatarTrans = nullptr;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_GhostAvatarControls_H