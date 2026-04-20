#include <osg/BoundingSphere>
#include <osg/ComputeBoundsVisitor>
#include <osgAnimation/RigGeometry>
#include <osgDB/ReadFile>
#include <cover/coVRPluginSupport.h> // includes cover

#include "GhostAvatarControls.h"

#include <iostream>

using namespace opencover;

GhostAvatarControls::GhostAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName)
    : GhostAvatarControls(pathToFbx, armNodeName, headNodeName, { 0, 1, 0 }, { 0, 1, 0 }, osg::Matrix::identity(), osg::Matrix::identity())
{
}

GhostAvatarControls::GhostAvatarControls(const std::string &pathToFbx, const std::string &armNodeName, const std::string &headNodeName, const osg::Vec3 &armBaseVector, const osg::Vec3 &headBaseVector, const osg::Matrix &armAdjustMatrix, const osg::Matrix &headAdjustMatrix)

    : m_pathToFbx(pathToFbx)
    , m_armNodeName(armNodeName)
    , m_headNodeName(headNodeName)
    , m_armBaseVector(armBaseVector)
    , m_headBaseVector(headBaseVector)
    , m_armAdjustMatrix(armAdjustMatrix)
    , m_headAdjustMatrix(headAdjustMatrix)
{
}

GhostAvatarControls::~GhostAvatarControls()
{
    if (m_avatarTrans && cover && cover->getObjectsRoot())
        cover->getObjectsRoot()->removeChild(m_avatarTrans);
}

void GhostAvatarControls::loadAvatar()
{
    auto model = osgDB::readNodeFile(m_pathToFbx);
    if (!model)
    {
        std::cerr << "GhostAvatarControls::loadAvatar: failed to load model " << m_pathToFbx << "\n";
        return;
    }
    if (m_avatarTrans)
        cover->getObjectsRoot()->removeChild(m_avatarTrans);

    m_avatarTrans = new osg::MatrixTransform();
    m_avatarTrans->setName(m_nodeName);
    m_avatarTrans->addChild(model);

    cover->getObjectsRoot()->addChild(m_avatarTrans);

    m_avatarTrans->accept(m_parser);

    // Todo: Do we really need this anymore? Only seems to swap x and y for ghost avatar, no difference for planar avatar...
    struct RigGeometryBoundsFixer : public osg::NodeVisitor
    {
        RigGeometryBoundsFixer()
            : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
        {
        }
        void apply(osg::Geode &geode) override
        {
            for (unsigned int i = 0; i < geode.getNumDrawables(); ++i)
            {
                if (auto rig = dynamic_cast<osgAnimation::RigGeometry *>(geode.getDrawable(i)))
                {
                    std::cout << "Found RigGeometry '" << rig->getName() << "' -> setting compute bounding box callback\n";
                    rig->setComputeBoundingBoxCallback(
                        new osgAnimation::RigComputeBoundingBoxCallback(1.0));
                    static_cast<osgAnimation::RigComputeBoundingBoxCallback *>(
                        rig->getComputeBoundingBoxCallback())
                        ->reset();
                    rig->dirtyBound();
                }
            }
            traverse(geode);
        }
    };
    RigGeometryBoundsFixer rigFixer;
    m_avatarTrans->accept(rigFixer);

    osg::ComputeBoundsVisitor computeBounds;
    m_avatarTrans->accept(computeBounds);
    osg::BoundingBox boundingBox = computeBounds.getBoundingBox();

    float sizeX = boundingBox.xMax() - boundingBox.xMin();
    float sizeY = boundingBox.yMax() - boundingBox.yMin();
    float sizeZ = boundingBox.zMax() - boundingBox.zMin();

    m_initialHeight = sizeY;

    std::cerr << "--------- x size: " << sizeX << ", y size: " << sizeY << ", z size: " << sizeZ << std::endl;

    loadArmBone();
    loadHeadBone();
}

osg::ref_ptr<osg::Node> GhostAvatarControls::getAvatarNode() const
{
    return m_avatarTrans;
}

bool GhostAvatarControls::hasArm()
{
    return m_hasArm;
}

bool GhostAvatarControls::hasHead()
{
    return m_hasHead;
}

osg::Vec3 GhostAvatarControls::getArmBaseVector() const
{
    return m_armBaseVector;
}

osg::Vec3 GhostAvatarControls::getHeadBaseVector() const
{
    return m_headBaseVector;
}

osg::Matrix GhostAvatarControls::getArmAdjustMatrix() const
{
    return m_armAdjustMatrix;
}

osg::Matrix GhostAvatarControls::getHeadAdjustMatrix() const
{
    return m_headAdjustMatrix;
}

void GhostAvatarControls::setArmBaseVector(const osg::Vec3 &vector)
{
    m_armBaseVector = vector;
}

void GhostAvatarControls::setHeadBaseVector(const osg::Vec3 &vector)
{
    m_headBaseVector = vector;
}

void GhostAvatarControls::setArmAdjustMatrix(const osg::Matrix &matrix)
{
    m_armAdjustMatrix = matrix;
}

void GhostAvatarControls::setHeadAdjustMatrix(const osg::Matrix &matrix)
{
    m_headAdjustMatrix = matrix;
}

void GhostAvatarControls::setArmAdjustMatrix(int row, const osg::Vec3 &vector)
{
    m_armAdjustMatrix(row, 0) = vector.x();
    m_armAdjustMatrix(row, 1) = vector.y();
    m_armAdjustMatrix(row, 2) = vector.z();
}

void GhostAvatarControls::setHeadAdjustMatrix(int row, const osg::Vec3 &vector)
{
    m_headAdjustMatrix(row, 0) = vector.x();
    m_headAdjustMatrix(row, 1) = vector.y();
    m_headAdjustMatrix(row, 2) = vector.z();
}

osg::Matrix GhostAvatarControls::getArmLocalToWorldMatrix() const
{
    return getLocalToWorldMatrix(*m_armBone);
}

osg::Matrix GhostAvatarControls::getHeadLocalToWorldMatrix() const
{
    return getLocalToWorldMatrix(*m_headBone);
}

osg::Vec3 GhostAvatarControls::getInitialArmPosition() const
{
    return getInitialBonePosition(*m_armBone);
}

osg::Vec3 GhostAvatarControls::getInitialHeadPosition() const
{
    return getInitialBonePosition(*m_headBone);
}

void GhostAvatarControls::loadArmBone()
{
    m_armBone = m_parser.getBoneByName(m_armNodeName);
    if (m_armBone)
    {
        m_hasArm = true;
    }
    else
    {
        if (m_armNodeName != "")
            std::cerr << "The avatar does not seem to have a bone called " << m_armNodeName << " -> can't move its arm!\n";
    }
}

void GhostAvatarControls::loadHeadBone()
{
    m_headBone = m_parser.getBoneByName(m_headNodeName);
    if (m_headBone)
    {
        m_hasHead = true;
    }
    else
    {
        if (m_headNodeName != "")
            std::cerr << "The avatar does not seem to have a bone called " << m_headNodeName << " -> can't move its head!\n";
    }
}

void GhostAvatarControls::moveBoneToTarget(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix)
{
    auto targetVector = getLocalTargetVector(bone, targetPosition, adjustMatrix);

    bone.pos->setTranslate(targetVector);
}

void GhostAvatarControls::makeBonePointAtTarget(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix, const osg::Vec3 &baseVector)
{
    osg::Vec3 targetVector = getLocalTargetVector(bone, targetPosition, adjustMatrix);

    if (baseVector[0] != 0 || baseVector[1] != 0 || baseVector[2] != 0)
    {
        osg::Quat rotation;
        rotation.makeRotate(baseVector, targetVector);
        bone.rot->setQuaternion(rotation);
    }
}

osg::Vec3 GhostAvatarControls::getInitialBonePosition(const BoneParser::Bone &bone) const
{
    return bone.initialPos * getLocalToWorldMatrix(bone);
}

osg::Matrix GhostAvatarControls::getLocalToWorldMatrix(const BoneParser::Bone &bone) const
{
    return bone.parent->osgNode->getWorldMatrices(cover->getObjectsRoot())[0];
}

osg::Vec3 GhostAvatarControls::getPositionInLocalCoordinates(const BoneParser::Bone &bone, const osg::Vec3 &positionInWorldCoordinates) const
{
    auto localToWorldMatrix = getLocalToWorldMatrix(bone);
    return positionInWorldCoordinates * osg::Matrix::inverse(localToWorldMatrix);
}

osg::Vec3 GhostAvatarControls::getLocalTargetVector(const BoneParser::Bone &bone, const osg::Vec3 &targetPosition, const osg::Matrix &adjustMatrix) const
{
    auto targetPositionInLocalCoordinates = getPositionInLocalCoordinates(bone, targetPosition);

    // apply axis-convention correction, if necessary
    return adjustMatrix * (targetPositionInLocalCoordinates - bone.initialPos);
}
