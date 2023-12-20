#include "Avatar.h"
#include <iostream>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/coVRCommunication.h>
#include <cover/coVRPartner.h>
#include <cover/VRAvatar.h>
#include <cover/coVRFileManager.h>
#include <boost/filesystem/path.hpp>
#include <osgDB/ReadFile>
#include <osgAnimation/UpdateBone>
#include <osg/ShapeDrawable>
#include <cover/ui/Slider.h>
#include <osg/Material>
#include <OpenVRUI/osg/mathUtils.h>
#include <ik/ik.h>



static void applyIkToOsgNode(ik_node_t*node)
{
    auto b = (Bone*) node->user_data;
        osg::Quat appliedRotation;
        // oik coords are in root(shoulder) coord system so we have to transform the orientation to local system of the bone
        osg::Quat totalRotation = osg::Quat(node->rotation.x, node->rotation.y, node->rotation.z, node->rotation.w);
        auto parent = b->parent;
        while(parent)
        {
            appliedRotation *= b->parent->rotation->getQuaternion();
            parent = parent->parent;
        }
        b->rotation->setQuaternion(appliedRotation.inverse() * totalRotation);
}

//public

bool LoadedAvatar::loadAvatar(const std::string &filename, ui::Menu *menu)
{
    
    model = osgDB::readNodeFile(filename);  
    if(!model)
        return false;
    positionAndScaleModel(model);
    std::cerr<< "loaded model " << filename << std::endl;
    
    initArmJoints(menu);
    createAnimationSliders(menu);
    buidIkModel();
    
    return true;
}

void LoadedAvatar::update(const osg::Vec3 &targetInWorldCoords)
{
    resetIkTransform(); // not sure why this step is necessary
    auto targetInShoulderCoords = worldPosToShoulder(targetInWorldCoords);
    ikTarget->target_position = ik.vec3.vec3(targetInShoulderCoords.x(), targetInShoulderCoords.y(), targetInShoulderCoords.z());
    // eventually also set rotation -> ikTarget->target_rotation = ik.quat.quat(0.894, 0, 0, 0.447);

    ik.solver.solve(ikSolver);

    //update the osg counterparts of the ik model
    //ik coords are in the shoulder coord system, so they have to be transformed to their local system
    ik.solver.iterate_affected_nodes(ikSolver, applyIkToOsgNode);
}

//private

void LoadedAvatar::positionAndScaleModel(osg::Node* model)
{
    modelTrans = new osg::MatrixTransform;
    cover->getObjectsRoot()->addChild(modelTrans);
    auto scale = osg::Matrix::scale(osg::Vec3f(10, 10, 10));
    auto rot1 = osg::Matrix::rotate(osg::PI / 2, 1,0,0);
    auto rot2 = osg::Matrix::rotate(osg::PI, 0,0,1);
    auto transM = osg::Matrix::translate(osg::Vec3f{0,0,-1300});
    modelTrans->setMatrix(scale * rot1 * rot2 * transM);
    modelTrans->addChild(model);
}

void LoadedAvatar::initArmJoints(ui::Menu *menu)
{
    shoulder = std::make_unique<Bone>(nullptr, "mixamorig:RightArm", model, menu); //z = 0, y = between 0 and 90, x between 280 and 45
    ellbow = std::make_unique<Bone>(shoulder.get(), "mixamorig:RightForeArm", model, menu); // x between 210 and 360
    wrist = std::make_unique<Bone>(ellbow.get(), "mixamorig:RightHand", model, menu);
    // Bone(nullptr, "mixamorig:RightShoulder", model, menu); //just to delete from animator 
}

void LoadedAvatar::createAnimationSliders(ui::Menu *menu)
{
    model->accept(m_animationFinder);
    m_animations = m_animationFinder.m_am->getAnimationList();

    for(const auto & anim : m_animations)
    {
        anim->setPlayMode(osgAnimation::Animation::PlayMode::LOOP);
        auto slider = new ui::Slider(menu, anim->getName());
        slider->setBounds(0,1);
        slider->setCallback([this, &anim](double val, bool x){
            m_animationFinder.m_am->playAnimation(anim, 1, val);
        });
        m_animationFinder.m_am->stopAnimation(anim);
    }
}

void LoadedAvatar::buidIkModel()
{
    ikSolver = ik.solver.create(IK_FABRIK);

    // Create a simple arm-bone structure
    ikShoulder = ikSolver->node->create(0);
    ikEllbow = ikSolver->node->create_child(ikShoulder, 1);
    ikAnkle = ikSolver->node->create_child(ikEllbow, 2);

    // Set node positions in local space so they form a straight line in the Y direction
    ikEllbow->position = ik.vec3.vec3(0, shoulder->length, 0);
    ikAnkle->position = ik.vec3.vec3(0, ellbow->length, 0);

    // set userdata to later update correspondig osg nodes 
    ikShoulder->user_data = shoulder.get();
    ikEllbow->user_data = ellbow.get();
    ikAnkle->user_data = wrist.get();

    // Attach an effector at the end -> ankle will have position and orientation set to this effector
    ikTarget = ikSolver->effector->create();
    ikSolver->effector->attach(ikTarget, ikAnkle);

    /* We want to calculate rotations as well as positions */
    ikSolver->flags |= IK_ENABLE_TARGET_ROTATIONS;

    /* Assign our tree to the solver, rebuild data and calculate solution */
    ik.solver.set_tree(ikSolver, ikShoulder);
    ik.solver.rebuild(ikSolver);
}

void LoadedAvatar::resetIkTransform()
{
    ikShoulder->position =  ik.vec3.vec3(0, 0, 0);
    ikEllbow->position = ik.vec3.vec3(0, shoulder->length, 0);
    ikAnkle->position = ik.vec3.vec3(0, ellbow->length, 0);
    ikShoulder->rotation = ik.quat.quat(0, 0, 0, 1);
    ikEllbow->rotation = ik.quat.quat(0, 0, 0, 1);
    ikAnkle->rotation = ik.quat.quat(0, 0, 0, 1);
}

osg::Vec3 LoadedAvatar::worldPosToShoulder(const osg::Vec3 &pos)
{
    auto shoulderToWorld = shoulder->bone->getBoneParent()->getWorldMatrices(cover->getObjectsRoot())[0];
    auto worldToShoulder = osg::Matrix::inverse(shoulderToWorld);
    return pos * worldToShoulder;
}

