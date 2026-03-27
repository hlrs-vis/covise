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
    GhostAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName);
    GhostAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName, const osg::Vec3 &armBaseVector, const osg::Vec3 &headBaseVector, const osg::Matrix &armAdjustMatrix, const osg::Matrix &headAdjustMatrix);
    virtual ~GhostAvatarControls() = default;

    void loadAvatar();

    virtual void updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix) = 0;

    osg::Vec3 getArmBaseVector() const;
    osg::Vec3 getHeadBaseVector() const;

    osg::Matrix getArmAdjustMatrix() const;
    osg::Matrix getHeadAdjustMatrix() const;

    void setArmBaseVector(const osg::Vec3 &vector);
    void setHeadBaseVector(const osg::Vec3 &vector);

    void setArmAdjustMatrix(const osg::Matrix &matrix);
    void setHeadAdjustMatrix(const osg::Matrix &matrix);

    void setArmAdjustMatrix(int row, const osg::Vec3 &vector);
    void setHeadAdjustMatrix(int row, const osg::Vec3 &vector);

    osg::Matrix getArmLocalToWorldMatrix() const;
    osg::Matrix getHeadLocalToWorldMatrix() const;

    osg::Vec3 getInitialArmPosition() const;
    osg::Vec3 getInitialHeadPosition() const;

protected:
    std::string m_pathToFbx;

    std::string m_armNodeName;
    std::string m_headNodeName;

    osg::MatrixTransform *m_avatarTrans = nullptr;

    BoneParser m_parser;
    BoneParser::Bone *m_armBone;
    BoneParser::Bone *m_headBone;

    osg::Vec3 m_armBaseVector;
    osg::Vec3 m_headBaseVector;

    osg::Matrix m_armAdjustMatrix;
    osg::Matrix m_headAdjustMatrix;

    void moveBoneToTarget(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix);
    void makeBonePointAtTarget(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix, const osg::Vec3 &baseVector);

    osg::Vec3 getInitialBonePosition(const BoneParser::Bone &bone) const;
    osg::Matrix getLocalToWorldMatrix(const BoneParser::Bone &bone) const;
    osg::Vec3 getPositionInLocalCoordinates(const BoneParser::Bone &bone, const osg::Vec3 &positionInWorldCoordinates) const;
    osg::Vec3 getLocalTargetVector(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix) const;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_GhostAvatarControls_H