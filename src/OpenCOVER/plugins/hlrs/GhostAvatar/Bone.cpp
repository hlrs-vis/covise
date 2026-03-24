#include <algorithm>
#include <string>
#include <unordered_set>

#include <osgAnimation/Bone>
#include <osgAnimation/StackedMatrixElement>
#include <osgAnimation/UpdateBone>

#include "Bone.h"

BoneParser::BoneParser()
    : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
}

void initializeBoneTransforms(osg::Node &node, BoneParser::Bone &bone)
{
    auto &boneTransforms = dynamic_cast<osgAnimation::UpdateBone *>(node.getUpdateCallback())->getStackedTransforms();
    osgAnimation::StackedMatrixElement *boneMatrix = nullptr;

    // first check if bone can be translated and/or rotated...
    for (const auto &transform : boneTransforms)
    {
        if (auto translator = dynamic_cast<osgAnimation::StackedTranslateElement *>(transform.get()))
        {
            bone.pos = translator;
            bone.initialPos = translator->getTranslate();
        }
        else if (auto rotator = dynamic_cast<osgAnimation::StackedQuaternionElement *>(transform.get()))
        {
            bone.rot = rotator;
            bone.initialRot = rotator->getQuaternion();
        }
        else if (auto matrix = dynamic_cast<osgAnimation::StackedMatrixElement *>(transform.get()))
        {
            boneMatrix = matrix;
        }
    }

    // ... if not, add translate/quaternion element to make sure we can transform the bone
    if (!bone.pos)
    {
        bone.pos = new osgAnimation::StackedTranslateElement;
        boneTransforms.push_back(bone.pos);

        if (boneMatrix)
            bone.initialPos = boneMatrix->getMatrix().getTrans();
    }

    if (!bone.rot)
    {
        bone.rot = new osgAnimation::StackedQuaternionElement;
        boneTransforms.push_back(bone.rot);

        if (boneMatrix)
            bone.initialRot = boneMatrix->getMatrix().getRotate();
    }
}

void BoneParser::apply(osg::Node &node)
{
    if (auto osgBone = dynamic_cast<osgAnimation::Bone *>(&node))
    {
        BoneParser::Bone bone;
        bone.osgNode = &node;

        if (osgBone->getBoneParent())
            bone.parent = &nodeToBoneMap[osgBone->getBoneParent()];
        else
            root = osgBone;

        initializeBoneTransforms(node, bone);

        nodeToBoneMap.emplace(std::make_pair(&node, bone));
    }

    traverse(node);
}

BoneParser::NodeToBoneMap::iterator BoneParser::findNode(const std::string &name)
{
    return std::find_if(nodeToBoneMap.begin(), nodeToBoneMap.end(), [&name](const NodeToBoneMap::value_type &p)
        { return p.first->getName() == name; });
}

void BoneParser::printBoneHierarchy() const
{
    std::cerr << "Bone hierarchy (" << nodeToBoneMap.size() << " bones):" << std::endl;

    if (nodeToBoneMap.empty())
    {
        std::cerr << "  <empty>" << std::endl;
        return;
    }

    std::unordered_set<const Bone *> visited;
    const auto printBone = [&](const auto &self, const Bone *bone, int depth) -> void
    {
        if (!bone)
            return;

        std::string indent(static_cast<size_t>(depth) * 2, ' ');
        const osg::Node *node = bone->osgNode;

        bool isRoot = (depth == 1);
        std::cerr << indent << (isRoot ? "" : "- ") << (node ? node->getName() : "<unnamed>") << (isRoot ? " (root)" : "") << std::endl;

        if (!visited.insert(bone).second)
        {
            std::cerr << indent << "  <cycle detected>" << std::endl;
            return;
        }

        for (const auto &entry : nodeToBoneMap)
        {
            if (entry.second.parent == bone)
                self(self, &entry.second, depth + 1);
        }
    };

    bool hasRoot = false;
    for (const auto &entry : nodeToBoneMap)
    {
        if (!entry.second.parent)
        {
            hasRoot = true;
            printBone(printBone, &entry.second, 1);
        }
    }

    if (!hasRoot)
    {
        std::cerr << "  <no root bone found; flat listing>" << std::endl;
        for (const auto &entry : nodeToBoneMap)
            printBone(printBone, &entry.second, 1);
    }

    std::cerr << std::endl;
}