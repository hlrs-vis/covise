/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Animated_Avatar_Plugin
#define _Animated_Avatar_Plugin

#include "Bone.h"

#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/FileBrowser.h>
#include <cover/ui/Owner.h>
#include <cover/ui/VectorEditField.h>
#include <ik/ik.h>

#include <PluginUtil/coVR3DTransInteractor.h>

#include <map>
using namespace covise;
using namespace opencover;
using namespace ui;

namespace opencover{
class VRAvatar;
}


class LoadedAvatar{
public:
    bool loadAvatar(const std::string &filename, VRAvatar *partnerAvatar, ui::Menu *menu);
    void update(const osg::Vec3 &target);
    void update();
private:
    void positionAndScaleModel(osg::Node* model, osg::MatrixTransform *pos);
    void initArmJoints(ui::Menu *menu);
    void createAnimationSliders(ui::Menu *menu);
    void buidIkModel();
    void resetIkTransform();
    osg::Vec3 worldPosToShoulder(const osg::Vec3 &pos);
    osgAnimation::AnimationList m_animations;
    AnimationManagerFinder m_animationFinder;
    std::unique_ptr<Bone> wrist, ellbow, shoulder;
    bool m_rotateButtonPresed = false;
    osg::ref_ptr<osg::Node> model;
    osg::ref_ptr<osg::MatrixTransform> modelTrans;
    opencover::ui::Button *lockA, *lockTheta, *lockB, *flipDeltaW, *flipTheta, *flipA, *flipB;
    opencover::ui::EditField *AOffset, *ThethaOffset, *BOffset;
    osg::ref_ptr<osg::MatrixTransform> targetDummy, ellbowDummy;
    ik_solver_t* ikSolver;
    ik_node_t* ikShoulder, *ikEllbow, *ikAnkle;
    ik_effector_t* ikTarget;
    osg::MatrixTransform *m_targetTransform;


};







#endif
