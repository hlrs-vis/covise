#ifndef COVER_PLUGIN_GHOSTAVATAR_BoneParser_H
#define COVER_PLUGIN_GHOSTAVATAR_BoneParser_H

// TODO: - find more scalable character animation library
//       - need at least skin animation that runs on the GPU
//       - also want to animate humans walking for the traffic simulations

#include <map>
#include <iostream>

#include <osg/NodeVisitor>
#include <osgAnimation/Bone>
#include <osgAnimation/StackedQuaternionElement>
#include <osgAnimation/StackedTranslateElement>

struct BoneParser : public osg::NodeVisitor
{
    struct Bone
    {
        osg::ref_ptr<osgAnimation::StackedQuaternionElement> rot = nullptr;
        osg::ref_ptr<osgAnimation::StackedTranslateElement> pos = nullptr;

        osg::Quat initialRot = { 0, 0, 0, 1 };
        osg::Vec3 initialPos = { 0, 0, 0 };

        Bone *parent = nullptr;
        osg::Node *osgNode = nullptr;
    };

public:
    typedef std::map<const osg::Node *, Bone> NodeToBoneMap;

    NodeToBoneMap nodeToBoneMap;
    osgAnimation::Bone *root;

    BoneParser();
    void apply(osg::Node &node);
    NodeToBoneMap::iterator getMapEntryByName(const std::string &name);
    Bone *getBoneByName(const std::string &name);
    void printBoneHierarchy() const;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_BoneParser_H