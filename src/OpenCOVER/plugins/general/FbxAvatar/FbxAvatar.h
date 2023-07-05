/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Animated_Avatar_Plugin
#define _Animated_Avatar_Plugin

#include "InverseKinematic.h"

#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/FileBrowser.h>
#include <cover/ui/VectorEditField.h>
#include <cover/ui/Owner.h>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osgAnimation/Animation>
#include <osgAnimation/BasicAnimationManager>
#include <osgAnimation/RigGeometry>
#include <osgAnimation/RigTransformHardware>
#include <osgAnimation/Skeleton>
#include <osgAnimation/StackedRotateAxisElement>
#include <osgAnimation/StackedQuaternionElement>
#include <osgAnimation/StackedTranslateElement>
#include <osgGA/GUIEventHandler>
#include <PluginUtil/coVR3DTransInteractor.h>

#include <map>
using namespace covise;
using namespace opencover;
using namespace ui;
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

struct RigGeometryFinder : public osg::NodeVisitor
{
    osg::ref_ptr<osgAnimation::RigGeometry> m_rig;
    RigGeometryFinder();
    void apply(osg::Geode &geode) override;
    void apply(osg::Drawable &geom) override;
};

struct Bone
{
    Bone(const std::string &name, osg::Node* model, ui::Menu *menu);
    osg::ref_ptr<osgAnimation::StackedQuaternionElement> rotation;
    osg::ref_ptr<osgAnimation::StackedTranslateElement> translate;
    osg::ref_ptr<osgAnimation::Bone> bone;
    const osg::Matrix* transform = nullptr;
    osgAnimation::StackedTransform* transforms = nullptr;
    float length;

    opencover::ui::Button *lockRotation;
    opencover::ui::VectorEditField *axis;
    opencover::ui::EditField *angle;
};


struct LoadedAvatar{

    bool loadAvatar(const std::string &filename, ui::Menu *menu);
    void update(const osg::Vec3 &target);
    osgAnimation::AnimationList m_animations;
    AnimationManagerFinder m_animationFinder;
    std::unique_ptr<Bone> hand, forearm, arm;
    bool m_rotateButtonPresed = false;
    IKTwoBoneJob m_rightArmKinematic;
    osg::ref_ptr<osg::Node> model;
    osg::ref_ptr<osg::MatrixTransform> shoulderToBall, shoulderForward;
    opencover::ui::Button *lockA, *lockTheta, *lockB, *flipDeltaW, *flipTheta, *flipA, *flipB;
    opencover::ui::EditField *AOffset, *ThethaOffset, *BOffset;
    osg::ref_ptr<osg::MatrixTransform> shoulderDummy, ellbowDummy;

};



class PLUGINEXPORT FbxAvatarPlugin : public coVRPlugin, public ui::Owner
{
public:

    FbxAvatarPlugin();
private:
    std::unique_ptr<FileBrowserConfigValue> m_avatarFile;
    osg::ref_ptr<osg::MatrixTransform>m_transform;//Position des Avatars
    osg::ref_ptr<osg::MatrixTransform>m_sphereTransform;
    std::shared_ptr<config::File>m_config;
    ui::Menu* m_menu = nullptr;
    std::unique_ptr<LoadedAvatar> m_avatar; 
    std::unique_ptr<coVR3DTransInteractor> m_interactor; 

    void loadAvatar();
    void key(int type, int keySym, int mod) override;
    bool update() override; //wird in jedem Frame aufgerufen
    void preFrame() override;

};

#endif
