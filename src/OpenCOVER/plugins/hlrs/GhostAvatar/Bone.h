#ifndef COVER_PLUGIN_GHOSTAVATAR_Bone_H
#define COVER_PLUGIN_GHOSTAVATAR_Bone_H

#include <osgAnimation/StackedTranslateElement>
#include <osgAnimation/StackedQuaternionElement>
#include <osgAnimation/Bone>

#include <osg/NodeVisitor>


#include <array>
#include <map>
#include <memory>

struct BoneParser : public osg::NodeVisitor {
struct Bone
{
    osg::ref_ptr<osgAnimation::StackedQuaternionElement> rot; //rotation to manipulate joint
    osg::Vec3 basePos; //initial position of the joint
    // const osgAnimation::StackedTranslateElement* basePos; //initial position of the joint
    Bone *parent = nullptr; 
    osg::Node *osgNode = nullptr;
    std::unique_ptr<osg::Vec3> controlPoint; //the bone should bend towards this position
    osg::MatrixTransform *ikSphere = nullptr; //sphere to visualize the bone
};

public:

    typedef  std::map<const osg::Node*, Bone> NodeMap;
    BoneParser();
    void apply(osg::Node& node);
    NodeMap::iterator findNode(const std::string &name);
    osg::Vec3 claculateBoneDistance(const std::string &boneName1, const std::string &boneName2);
    NodeMap nodeToIk;
    uint32_t ikId = 0;
    osgAnimation::Bone *root;
    const std::array<const char*, 5> effectorName{"mixamorig:Head", "mixamorig:LeftHand", "mixamorig:RightHand", "mixamorig:LeftFoot", "mixamorig:RightFoot"};
    // const std::array<const char*, 1> effectorName{"mixamorig:RightHand"};
    const std::array<size_t, 5> effectorChainLenghts{1, 2, 2, 2, 2};
};

#endif // COVER_PLUGIN_GHOSTAVATAR_Bone_H