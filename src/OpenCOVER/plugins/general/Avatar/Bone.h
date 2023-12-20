#ifndef COVER_PLUGIN_FBXAVATAR_Bone_H
#define COVER_PLUGIN_FBXAVATAR_Bone_H

#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osgAnimation/Animation>
#include <osgAnimation/BasicAnimationManager>
#include <osgAnimation/RigGeometry>
#include <osgAnimation/RigTransformHardware>
#include <osgAnimation/Skeleton>
#include <osgAnimation/StackedQuaternionElement>
#include <osgAnimation/StackedRotateAxisElement>
#include <osgAnimation/StackedTranslateElement>
#include <osgGA/GUIEventHandler>

namespace opencover{
namespace ui{
class Menu;
class Button;
class VectorEditField;
class EditField;
}
}

namespace osgAnimation{
class StackedTransform;
}
struct BoneFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osgAnimation::Bone> bone;
    BoneFinder(const std::string &name);
    void apply(osg::Node& node) override;
private:
    std::string m_nodeName;
};

struct AnimationManagerFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osgAnimation::BasicAnimationManager> m_am;
    AnimationManagerFinder();
    void apply(osg::Node& node) override;
};

struct Bone
{
    Bone(Bone *parent, const std::string &name, osg::Node* model, opencover::ui::Menu *menu);
    osg::ref_ptr<osgAnimation::StackedQuaternionElement> rotation;
    osg::ref_ptr<osgAnimation::Bone> bone;
    float length;
    Bone *parent = nullptr;
    std::string name;
};

#endif // COVER_PLUGIN_FBXAVATAR_Bone_H