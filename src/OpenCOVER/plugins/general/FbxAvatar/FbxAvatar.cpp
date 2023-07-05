#include "FbxAvatar.h"
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

COVERPLUGIN(FbxAvatarPlugin);

osg::ref_ptr<osg::Geode> createLine()
{
    osg::ref_ptr<osg::Geode> geodeCyl = new osg::Geode;
    geodeCyl->setName("measureLineGeode");
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0, 0, 0.5), 1, 1));
    // sd->setColor(osg::Vec4{1,0,0,1});
    sd->setName("measureLineShape");
    osg::ref_ptr<osg::StateSet> ss = sd->getOrCreateStateSet();
    geodeCyl->addDrawable(sd);
    return geodeCyl;
}

void updateLine(osg::MatrixTransform *m_geo, const osg::Vec3f &p1, const osg::Vec3f &p2)
{
        if(p1 == p2)
            return;
        auto vec = p2 - p1;
        float dist = (vec).length();
        osg::Matrix scale, rot, trans;
        scale.makeIdentity();
        rot.makeIdentity();
        trans.makeIdentity();
        scale.makeScale(osg::Vec3d{1,1,dist});
        osg::Vec3f zAxis{0,0,1};
        rot.makeRotate(zAxis, vec);
        trans.setTrans(p1);
        m_geo->setMatrix(scale * rot * trans);
}

BoneFinder::BoneFinder(const std::string &name) 
: osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
, m_nodeName(name) {}

void BoneFinder::apply(osg::Node& node) {

    std::cerr << "nodename " << node.getName();
    node.getUpdateCallback();
    if(dynamic_cast<osgAnimation::Skeleton*>(&node))
        std::cerr << " is a skeleton";
    if(dynamic_cast<osgAnimation::Bone*>(&node))
        std::cerr << " is a Bone";
    if(dynamic_cast<osgAnimation::RigGeometry*>(&node))
        std::cerr << " is a RigGeometry";
    std::cerr << std::endl;
    if(node.getName() == m_nodeName)
    {
        bone = dynamic_cast<osgAnimation::Bone*>(&node);
        return;
    }
    traverse(node);
}

AnimationManagerFinder::AnimationManagerFinder() 
: osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN) {}

void AnimationManagerFinder::apply(osg::Node& node) {

    if (m_am.valid())
        return;

    if (node.getUpdateCallback()) {       
        m_am = dynamic_cast<osgAnimation::BasicAnimationManager*>(node.getUpdateCallback());
        return;
    }
    
    traverse(node);
}

osg::Vec3f matToEuler(const osg::Matrix &mat)
{
    coCoord offsetCoord(mat);
    return offsetCoord.hpr;
}

osg::Matrix eulerToMat(const osg::Vec3f &angles) 
{
    osg::Matrix offMat;
    coCoord offsetCoord;
    offsetCoord.hpr = angles;
    offsetCoord.makeMat(offMat);
    return offMat;
}

Bone::Bone(const std::string &name, osg::Node* model, ui::Menu *menu)
:rotation(new osgAnimation::StackedQuaternionElement("quaternion", osg::Quat(0, osg::Y_AXIS)))
{
    BoneFinder bf(name);
    model->accept(bf);
    bone = bf.bone;
    auto cb = dynamic_cast<osgAnimation::UpdateBone*>(bone->getUpdateCallback());
    if(cb)
    {
        const std::array<std::string, 3> endings = {"X", "Y", "Z"};
        for (auto i = cb->getStackedTransforms().begin(); i < cb->getStackedTransforms().end(); i++)
        {
            std::cerr << name << " has stakced element " << (*i)->getName() << std::endl;
            // for (size_t j = 0; j < endings.size(); j++)
            // {
            //     if((*i)->getName() == "rotate" + endings[j])
            //         i = cb->getStackedTransforms().erase(i);
            //         // m_rotation[j] = dynamic_cast<osgAnimation::StackedRotateAxisElement*>((*i).get());
            // }
            if((*i)->getName() == "translate")
            {
                translate = dynamic_cast<osgAnimation::StackedTranslateElement*>(i->get());
                auto t = translate->getTranslate();
                auto scale = bone->getWorldMatrices(cover->getObjectsRoot())[0].getScale();
                for (size_t k = 0; k < 3; k++)
                {
                    t[k] *= scale[k];
                }
                length = t.length();
            }else{
                i = cb->getStackedTransforms().erase(i);
            }
        }
        std::cerr << std::endl;
        cb->getStackedTransforms().push_back(rotation);
        transform = &cb->getStackedTransforms().getMatrix();
        transforms = &cb->getStackedTransforms();
    }
    const std::array<std::string, 3> axisNames{"X", "Y", "Z"};
    auto g = new ui::Menu(menu, name);
    // for (size_t i = 0; i < 3; i++)
    // {
    //     auto s = new ui::Slider(g, axisNames[i]);
    //     s->setBounds(0, 360);
    //     s->setCallback([this, i](double val, bool rel)
    //     {
    //         osg::Vec3f v;
    //         v[i] = 1;
    //         osg::Matrix m;
    //         auto curentRot = rotation->getQuaternion();
    //         curentRot.get(m);
    //         auto angles = matToEuler(m);
    //         angles[i] = val;
    //         m = eulerToMat(angles);
    //         curentRot.set(m);
    //         rotation->setQuaternion(curentRot);
    //     });
    // }
    axis = new ui::VectorEditField(g, "axis");
    angle = new ui::EditField(g, "angle");
    lockRotation = new ui::Button(g, "lockRotation");
}

osg::ref_ptr<osg::Geode> createSphere()
{
    osg::ref_ptr<osg::Geode> geodeCyl = new osg::Geode;
    geodeCyl->setName("testSphere");
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0, 0, 0), 100));
    // sd->setColor(osg::Vec4{1,0,0,1});
    sd->setName("testSphereShape");
    osg::ref_ptr<osg::StateSet> ss = sd->getOrCreateStateSet();
    geodeCyl->addDrawable(sd);
    auto mat = new osg::Material;
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 0, 0, 1.0));
    mat->setAmbient(osg::Material::FRONT_AND_BACK,osg::Vec4(0.2, 0, 0, 1.0));
    mat->setColorMode(osg::Material::OFF);
    mat->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mat->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1.0));
    mat->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    ss->setAttributeAndModes(mat, osg::StateAttribute::ON);
    return geodeCyl;
}

bool LoadedAvatar::loadAvatar(const std::string &filename, ui::Menu *menu)
{
    osg::MatrixTransform *trans = new osg::MatrixTransform;
    cover->getObjectsRoot()->addChild(trans);
    auto scale = osg::Matrix::scale(osg::Vec3f(10, 10, 10));
    auto rot1 = osg::Matrix::rotate(osg::PI / 2, 1,0,0);
    auto rot2 = osg::Matrix::rotate(osg::PI, 0,0,1);
    auto transM = osg::Matrix::translate(osg::Vec3f{0,0,-1300});
    trans->setMatrix(scale * rot1 * rot2 * transM);
    model = osgDB::readNodeFile(filename);  
    if(!model)
        return false;
    model->accept(m_animationFinder);
    int i = 0;
    trans->addChild(model);
    std::cerr<< "loaded model " << filename << std::endl;

    hand = std::make_unique<Bone>("mixamorig:RightHand", model, menu);
    forearm = std::make_unique<Bone>("mixamorig:RightForeArm", model, menu); // x between 210 and 360
    arm = std::make_unique<Bone>("mixamorig:RightArm", model, menu); //z = 0, y = between 0 and 90, x between 280 and 45

    auto mm = new ui::Menu(menu, "angle offsets");
    lockA = new opencover::ui::Button(mm, "lockA");
    lockTheta = new opencover::ui::Button(mm, "lockTheta");
    lockB = new opencover::ui::Button(mm, "lockB");
    AOffset = new opencover::ui::EditField(mm, "AOffset");
    ThethaOffset = new opencover::ui::EditField(mm, "ThethaOffset");
    BOffset = new opencover::ui::EditField(mm, "BOffset");

    flipDeltaW = new opencover::ui::Button(mm, "flipDeltaW");
    flipTheta = new opencover::ui::Button(mm, "flipTheta");
    flipA = new opencover::ui::Button(mm, "flipA");
    flipB = new opencover::ui::Button(mm, "flipB");
    
    shoulderDummy = new osg::MatrixTransform;
    auto sphere = createSphere();
    shoulderDummy->addChild(sphere);
    cover->getObjectsRoot()->addChild(shoulderDummy);

    ellbowDummy = new osg::MatrixTransform;
    auto sphere2 = createSphere();
    ellbowDummy->addChild(sphere2);
    cover->getObjectsRoot()->addChild(ellbowDummy);

    m_animations = m_animationFinder.m_am->getAnimationList();

    // for(const auto & anim : m_animations)
    // {
    //     std::cerr << "avatar has animation " << i++ << " " << anim->getName() << std::endl;
    //     anim->setPlayMode(osgAnimation::Animation::PlayMode::LOOP);
    //     auto slider = new ui::Slider(menu, anim->getName());
    //     slider->setBounds(0,1);
    //     slider->setCallback([this, &anim](double val, bool x){
    //         m_animationFinder.m_am->playAnimation(anim, 1, val);
    //     });
    //     m_animationFinder.m_am->stopAnimation(anim);
    // }

    return true;
}

void LoadedAvatar::update(const osg::Vec3 &targetWorld)
{
    arm->transforms->update();
    auto worldToShoulder = arm->bone->getBoneParent()->getWorldMatrices(cover->getObjectsRoot())[0];
    auto forwardArm = osg::Y_AXIS;
    auto forwardShoulder = forwardArm;
    auto targetShoulder = targetWorld * worldToShoulder;
    auto targetShoulderDistanceWorld = (targetWorld - worldToShoulder.getTrans()).length();
    targetShoulderDistanceWorld = (targetWorld - arm->bone->getWorldMatrices(cover->getObjectsRoot())[0].getTrans()).length();
    // osg::Quat armRotation;
    // armRotation.makeRotate(forwardArm, targetArm);
    auto armLenght = arm->length + forearm->length;
    osg::Quat shoulderRotation;
    shoulderRotation.makeRotate(forwardShoulder, targetShoulder);
    arm->rotation->setQuaternion(shoulderRotation);
    if(targetShoulderDistanceWorld > armLenght)
    {
        forearm->rotation->setQuaternion(osg::Quat());
        hand->rotation->setQuaternion(osg::Quat());
        shoulderDummy->setMatrix(osg::Matrix::identity());
        ellbowDummy->setMatrix(osg::Matrix::identity());
    }
    else{
        auto A = arm->bone->getWorldMatrices(cover->getObjectsRoot())[0].getTrans();
        auto B = forearm->bone->getWorldMatrices(cover->getObjectsRoot())[0].getTrans();
        auto C = targetWorld;

        auto mA = osg::Matrix::translate(A);
        auto mB = osg::Matrix::translate(B);

        shoulderDummy->setMatrix(mA);
        ellbowDummy->setMatrix(mB);

        auto AD = A - C;
        AD.y() = 0;
        auto flipAD = flipDeltaW ? -1 : 1;
        auto ACounter = atanf((C.y() - A.y()) / flipAD /AD.length());

        auto theta = atan((C.z() - A.z())/ (C.x() - A.x()));
        auto a = arm->length;
        auto b = forearm->length;
        auto c = std::min(targetShoulderDistanceWorld, a + b);
        auto test1 = (b*b + c*c - a*a)/2/b/c;
        auto test2 = (C.y() - A.y())/sqrt(pow(C.x() - A.x(), 2) + pow(C.z() - A.z(), 2));
        auto test3 = (a*a + c*c - b*b)/2/a/c;
        auto AAngle = acos((b*b + c*c - a*a)/2/b/c) + ACounter; //atan((C.y() - A.y())/sqrt(pow(C.x() - A.x(), 2) + pow(C.z() - A.z(), 2)));
        auto BAngle = osg::PI - acos((a*a + c*c - b*b)/2/a/c);
        // osg::Quat BQuat = osg::Quat(forearm->angle->number() / 180 * osg::PI, forearm->axis->value());
        // osg::Quat AQuat = osg::Quat(arm->angle->number() / 180 * osg::PI, arm->axis->value());
        // if(!forearm->lockRotation->state())
        //     BQuat = osg::Quat(BAngle, osg::X_AXIS) * BQuat;
        // if(!arm->lockRotation->state())
        //     AQuat = osg::Quat(AAngle, osg::X_AXIS) * osg::Quat(theta, osg::Y_AXIS)* AQuat;

        if(lockA->state())
            AAngle = AOffset->number() / 180 * osg::PI;
        else
            AAngle += AOffset->number() / 180 * osg::PI;

        if(lockB->state())
            BAngle = BOffset->number() / 180 * osg::PI;
        else
            BAngle += BOffset->number() / 180 * osg::PI;

        if(lockTheta->state())
            theta = ThethaOffset->number() / 180 * osg::PI;
        else
            theta += ThethaOffset->number() / 180 * osg::PI;

        if(flipTheta->state())
            theta *= -1;

        if(flipB->state())
            BAngle *= -1; 
        if(flipA->state())
            AAngle *= -1;

        forearm->rotation->setQuaternion(osg::Quat(BAngle, osg::X_AXIS));
        arm->rotation->setQuaternion(osg::Quat(AAngle, osg::X_AXIS) * osg::Quat(theta, osg::Z_AXIS));


        // auto worldToAnkle = forearm->bone->getWorldMatrices(cover->getObjectsRoot())[0];
        // //arm->transforms->getMatrix() 
        // auto shouderToAnkle = osg::Matrix::identity(); 
        // forearm->translate->applyToMatrix(shouderToAnkle);
        // forearm->rotation->applyToMatrix(shouderToAnkle);
        // arm->translate->applyToMatrix(shouderToAnkle);
        // arm->rotation->applyToMatrix(shouderToAnkle);

        // auto worldToAnkle2 = worldToShoulder *arm->transforms->getMatrix() * forearm->transforms->getMatrix();
        // auto worldToAnkle3 = worldToShoulder * shouderToAnkle;
    }

    // osg::Matrix shoulderToWorld;
    // shoulderToWorld = osg::Matrix::inverse(worldToShoulder);
    // // auto modelToWorld = osg::Matrix::inverse(worldToModel);
    // auto forward = osg::Y_AXIS;
    // osg::Vec3f t = target * worldToShoulder; 
    // osg::Quat shoulderRotWorld;
    // shoulderRotWorld.makeRotate(forward, t);
    // shoulder->rotation->setQuaternion(shoulderRotWorld);
    // updateLine(shoulderToBall, target, worldToShoulder.getTrans());

    // updateLine(shoulderForward, worldToShoulder.getTrans(), scaledForward);
    // VRAvatar a;
    // m_rightArmKinematic.start_joint = arm->bone->getWorldMatrices(model)[0];
    // m_rightArmKinematic.mid_joint = foreArm->bone->getWorldMatrices(model)[0];
    // m_rightArmKinematic.end_joint = hand->bone->getWorldMatrices(model)[0];


    // m_rightArmKinematic.target = target;
    // if(!m_rightArmKinematic.Run())
    // {
    //     std::cerr << "failed to rn IK" << std::endl;
    //     return;
    // }
    // arm->rotation->setQuaternion(m_rightArmKinematic.start_joint_correction);
    // foreArm->rotation->setQuaternion(m_rightArmKinematic.mid_joint_correction);
    // auto c =  &m_rightArmKinematic.mid_joint_correction;
    // std::cerr << "mid_joint_correction [" << c->x() << ", " << c->y() << ", " << c->z()  << ", " << c->w() << " for target " << target.x() << ", " << target.y() << ", " << target.z() << std::endl;
    // // hand->rotation->setQuaternion(a.handTransform->getMatrix().getRotate());

    // if(m_rotateButtonPresed)
    // {
        
    //     double angle;
    //     osg::Vec3 axis;
    //     hand->rotation->getQuaternion().getRotate(angle, axis);
    //     auto newAngle = angle + osg::PI_4/10;
    //     if(newAngle >= osg::PI * 2)
    //         newAngle = 0;

    //     m_rotateButtonPresed = false;
    //     hand->rotation->setQuaternion(osg::Quat(newAngle, axis));
    //     std::cerr << "updated rotation: " << newAngle << std::endl;
    // }
}



FbxAvatarPlugin::FbxAvatarPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("FbxAvatarPlugin", cover->ui)//braucht man für gui
, m_transform(new osg::MatrixTransform)
, m_sphereTransform(new osg::MatrixTransform)
, m_config(config())
, m_menu(new ui::Menu("FbxAvatar",this))
{
    m_avatarFile = std::make_unique<FileBrowserConfigValue>(m_menu, "avatarFile", "", *m_config, "");
    m_avatarFile->ui()->setFilter("*.fbx");
    m_avatarFile->setUpdater([this](){
        loadAvatar();
    });
    loadAvatar();

    m_interactor .reset(new coVR3DTransInteractor(osg::Vec3{-400,0,0}, 10, vrui::coInteraction::InteractionType::ButtonA, "target", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactor->enableIntersection();
    // auto sphere = createSphere();
    // m_sphereTransform->addChild(sphere);
    // cover->getObjectsRoot()->addChild(m_sphereTransform);
}

void FbxAvatarPlugin::loadAvatar()
{
    m_avatar = std::make_unique<LoadedAvatar>();
    if(m_avatar->loadAvatar(m_avatarFile->getValue(), m_menu))
        m_config->save();
}

void updateMatrixPosition(osg::MatrixTransform *mt, const osg::Vec3f &pos)
{
    auto m = mt->getMatrix();
    m.setTrans(m.getTrans() + pos);
    mt->setMatrix(m);
}

void FbxAvatarPlugin::key(int type, int keySym, int mod)
{
    std::string key = "unknown";
    if (!(keySym & 0xff00))
    {
        char buf[2] = { static_cast<char>(keySym), '\0' };
        key = buf;
    }
    constexpr float speed = 50;
    osg::Vec3f position;
    if(key == "w")
    {
        position = osg::Vec3f(speed,0,0);
    } else if (key == "a")
    {
        position = osg::Vec3f(0, 0, speed);
    }
    else if (key == "s")
    {
        position = osg::Vec3f(-speed,0,0);
    }
    else if (key == "d")
    {
        position = osg::Vec3f(0, 0, -speed);
    }
    else if (key == "e")
    {
        position = osg::Vec3f(0, speed, 0);
    }
    else if (key == "r")
    {
        position = osg::Vec3f(0, -speed, 0);
    }
    updateMatrixPosition(m_sphereTransform, position);

}

bool FbxAvatarPlugin::update(){

    osg::Matrix m = osg::Matrix::identity();
    auto pos = m_interactor->getPos();
    m.setTrans(pos);
    m_sphereTransform->setMatrix(m);
    m_avatar->update(pos);
    return true;
}

void FbxAvatarPlugin::preFrame()
{
    m_interactor->preFrame();
}

