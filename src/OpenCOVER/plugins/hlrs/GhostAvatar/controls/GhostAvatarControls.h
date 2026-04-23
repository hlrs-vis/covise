#ifndef COVER_PLUGIN_GHOSTAVATAR_CONTROLS_GhostAvatarControls_H
#define COVER_PLUGIN_GHOSTAVATAR_CONTROLS_GhostAvatarControls_H

#include <string>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Quat>
#include <osg/ref_ptr>
#include <osg/Vec3>

#include "BoneParser.h"

class GhostAvatarControls
{
public:
    GhostAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName);
    GhostAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName, const osg::Vec3 &armBaseVector, const osg::Vec3 &headBaseVector, const osg::Matrix &armAdjustMatrix, const osg::Matrix &headAdjustMatrix);
    virtual ~GhostAvatarControls();

    void loadAvatar();

    osg::ref_ptr<osg::Node> getAvatarNode() const;

    osg::Vec3 getForwardDirection() const;
    osg::Vec3 getUpDirection() const;
    virtual osg::Vec3 getEyeOffset() const;

    virtual void updateBones(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix) = 0;

    bool hasArm();
    bool hasHead();

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

    osg::Quat getBaseRotation() const;
    void setBaseRotation(const osg::Quat &rotation);

    osg::Vec3 getInitialBounds() const;
    osg::Vec3 getBounds() const;
    float getBounds(int i) const;

    void setForwardDirection(const osg::Vec3 &direction);
    void setUpDirection(const osg::Vec3 &direction);

    std::string m_nodeName = "AvatarTrans";
    std::string m_armNodeName;
    std::string m_headNodeName;

    osg::ref_ptr<osg::MatrixTransform> m_avatarTrans;

    BoneParser m_parser;
    BoneParser::Bone *m_armBone = nullptr;
    BoneParser::Bone *m_headBone = nullptr;

    bool m_hasArm = false;
    bool m_hasHead = false;

    osg::Vec3 m_armBaseVector;
    osg::Vec3 m_headBaseVector;

    osg::Matrix m_armAdjustMatrix;
    osg::Matrix m_headAdjustMatrix;

    void loadArmBone();
    void loadHeadBone();

    void rotateBone(const BoneParser::Bone& bone, const osg::Quat & rotation, const osg::Matrix& adjustMatrix);

    void moveBoneToTarget(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix);
    void makeBonePointAtTarget(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix, const osg::Vec3 &baseVector);

    osg::Vec3 getInitialBonePosition(const BoneParser::Bone &bone) const;
    osg::Matrix getLocalToWorldMatrix(const BoneParser::Bone &bone) const;
    osg::Vec3 getPositionInLocalCoordinates(const BoneParser::Bone &bone, const osg::Vec3 &positionInWorldCoordinates) const;
    osg::Vec3 getLocalTargetVector(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix) const;

private:
    osg::Quat m_baseRotation = osg::Quat(0, 0, 0, 1);
    osg::Vec3 m_initialBounds = { 0, 0, 0 };

    osg::Vec3 m_forwardDirection = { 1.0, 0.0, 0.0 };
    osg::Vec3 m_upDirection = { 0.0, 0.0, 1.0 };
};

#endif // COVER_PLUGIN_GHOSTAVATAR_CONTROLS_GhostAvatarControls_H