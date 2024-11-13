/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <cover/coIntersection.h>

#include "SteeringWheel.h"
#include "Vehicle.h"
//#include "ITM.h"
//#include "FKFSDynamics.h"
//#include "EinspurDynamik.h"
#include "PorscheRealtimeDynamics.h"
#include "HLRSRealtimeDynamics.h"

#include <OpenVRUI/osg/mathUtils.h>

#ifndef USE_CAR_SOUND
Player *VrmlNodeVehicle::player = NULL;
#endif

osg::Vec2d VrmlNodeVehicle::actual_pos(-1, -1);
double VrmlNodeVehicle::velocity = -1.0;

void playerUnavailableCB()
{
#ifndef USE_CAR_SOUND
    VrmlNodeVehicle::player = NULL;
#endif
}

void VrmlNodeVehicle::initFields(VrmlNodeVehicle *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("carRotation", node->d_carRotation, [node](auto f) {
                         node->recalcMatrix();
                     }),
                     exposedField("carTranslation", node->d_carTranslation, [node](auto f) {
                         node->recalcMatrix();
                     }),
                     exposedField("carBodyRotation", node->d_carBodyRotation),
                     exposedField("carBodyTranslation", node->d_carBodyTranslation),
                     exposedField("axle1Rotation", node->d_axle1Rotation),
                     exposedField("axle1Translation", node->d_axle1Translation),
                     exposedField("axle2Rotation", node->d_axle2Rotation),
                     exposedField("axle2Translation", node->d_axle2Translation),

                     exposedField("wheelFLRotation", node->d_wheelFLRotation),
                     exposedField("wheelFLTranslation", node->d_wheelFLTranslation),
                     exposedField("wheelFRRotation", node->d_wheelFRRotation),
                     exposedField("wheelFRTranslation", node->d_wheelFRTranslation),
                     exposedField("wheelRLRotation", node->d_wheelRLRotation),
                     exposedField("wheelRLTranslation", node->d_wheelRLTranslation),
                     exposedField("wheelRRRotation", node->d_wheelRRRotation),
                     exposedField("wheelRRTranslation", node->d_wheelRRTranslation),

                     exposedField("cameraRotation", node->d_cameraRotation),
                     exposedField("cameraTranslation", node->d_cameraTranslation),
                     exposedField("mirrorLRotation", node->d_mirrorLRotation),
                     exposedField("mirrorMRotation", node->d_mirrorMRotation),
                     exposedField("mirrorRRotation", node->d_mirrorRRotation),

                     exposedField("offsetLoc", node->d_offsetLoc),
                     exposedField("offsetRot", node->d_offsetRot),
    
                     exposedField("numCars", node->d_numCars),
                     exposedField("followTerrain", node->d_followTerrain),

                     exposedField("ffz1Rotation", node->d_ffz1Rotation),
                     exposedField("ffz1Translation", node->d_ffz1Translation),
                     eventInCallBack("vMax", node->d_vMax), 
                     eventInCallBack("aMax", node->d_aMax),
                     eventInCallBack("reset", node->d_reset),
                     eventInCallBack("offsetLoc", node->d_offsetLoc),
                     eventInCallBack("offsetRot", node->d_offsetRot),
                     eventInCallBack("setOffset", node->d_setOffset),
                     eventInCallBack("printTransformation", node->d_printTransformation)
                     );

    for (size_t i = 0; i < NUM_WHEELS; i++)
    {
        initFieldsHelper(node, t,
                         exposedField("wheel" + std::to_string(i) + "Rotation", node->d_wheelRotations[i]),
                         exposedField("wheel" + std::to_string(i) + "Translation", node->d_wheelTranslations[i]));
    }
    
    for (size_t i = 0; i < 15; i++)
    {
        initFieldsHelper(node, t,
            eventInCallBack("carTranslation_" + std::to_string(i), node->d_carTranslations[i]));
    }
    for (size_t i = 0; i < 10; i++)
    {
        initFieldsHelper(node, t,
            eventInCallBack("float_value" + std::to_string(i), node->d_float_values[i]),
            eventInCallBack("int_value" + std::to_string(i), node->d_int_values[i]));
    }
    if(t)
    {
        t->addEventOut("brakePedalRotation", VrmlField::SFROTATION);
        t->addEventOut("clutchPedaRotation", VrmlField::SFROTATION);
        t->addEventOut("gasPedalRotation", VrmlField::SFROTATION);
        t->addEventOut("steeringWheelRotation", VrmlField::SFROTATION);
        t->addEventOut("wheelRotation", VrmlField::SFROTATION);

        t->addEventOut("speed", VrmlField::SFFLOAT);
        t->addEventOut("revs", VrmlField::SFFLOAT);
        t->addEventOut("acceleration", VrmlField::SFFLOAT);
        t->addEventOut("gear", VrmlField::SFINT32);

        t->addEventIn("reset", VrmlField::SFBOOL);
        t->addEventIn("gear", VrmlField::SFINT32);



        
    }
}

const char *VrmlNodeVehicle::name()
{
    return "Vehicle";
}

VrmlNodeVehicle::VrmlNodeVehicle(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_brakePedalRotation(1, 0, 0, 0)
    , d_clutchPedaRotation(1, 0, 0, 0)
    , d_gasPedalRotation(1, 0, 0, 0)
    , d_steeringWheelRotation(1, 0, 0, 0)
    , d_wheelRotation(1, 0, 0, 0)
    , d_carRotation(1, 0, 0, 0)
    , d_carTranslation(0, 0, 0)
    , d_carBodyRotation(1, 0, 0, 0)
    , d_carBodyTranslation(0, 0, 0)
    , d_axle1Rotation(1, 0, 0, 0)
    , d_axle1Translation(0, 0, 0)
    , d_axle2Rotation(1, 0, 0, 0)
    , d_axle2Translation(0, 0, 0)

    , d_wheelFLTranslation(0, 0, 0)
    , d_wheelFLRotation(1, 0, 0, 0)
    , d_wheelFRTranslation(0, 0, 0)
    , d_wheelFRRotation(1, 0, 0, 0)
    , d_wheelRLTranslation(0, 0, 0)
    , d_wheelRLRotation(1, 0, 0, 0)
    , d_wheelRRTranslation(0, 0, 0)
    , d_wheelRRRotation(1, 0, 0, 0)

    //Ghost
    , d_ffz1wheelFLTranslation(0, 0, 0)
    , d_ffz1wheelFLRotation(1, 0, 0, 0)
    , d_ffz1wheelFRTranslation(0, 0, 0)
    , d_ffz1wheelFRRotation(1, 0, 0, 0)
    , d_ffz1wheelRLTranslation(0, 0, 0)
    , d_ffz1wheelRLRotation(1, 0, 0, 0)
    , d_ffz1wheelRRTranslation(0, 0, 0)
    , d_ffz1wheelRRRotation(1, 0, 0, 0)

    , d_cameraRotation(1, 0, 0, 0)
    , d_cameraTranslation(0, 0, 0)
    , d_mirrorLRotation(1, 0, 0, 0)
    , d_mirrorMRotation(1, 0, 0, 0)
    , d_mirrorRRotation(1, 0, 0, 0)

    , d_setOffset(false)
    , d_offsetLoc(0, 0, 0)
    , d_offsetRot(1, 0, 0, 0)
    , d_printTransformation(false)

    , d_numCars(0)
    , d_followTerrain(false)
    , d_speed(0)
    , d_revs(0)
    , d_acceleration(0)
    , d_aMax(20.0)
    , d_vMax(200.0)
    , d_gear(0)
    , d_mirrorLightLeft(0)
    , d_mirrorLightRight(0)

    , d_ffz1Rotation(1, 0, 0, 0)
    , d_ffz1Translation(0, 0, 0)

{
    std::fill(d_wheelRotations.begin(), d_wheelRotations.end(), VrmlSFRotation(1, 0, 0, 0));
    std::fill(d_wheelTranslations.begin(), d_wheelTranslations.end(), VrmlSFVec3f(0, 0, 0));
    std::fill(d_float_values.begin(), d_float_values.end(), 0.0);
    std::fill(d_int_values.begin(), d_int_values.end(), 0);
    
    int i;
    for (i = 0; i < MAX_BODIES; i++)
    {
        bodyTransName[i] = new char[20];
        sprintf(bodyTransName[i], "body%dTranslation", i + 1);
        bodyRotName[i] = new char[20];
        sprintf(bodyRotName[i], "body%dRotation", i + 1);
        d_bodyTranslation[i].set(0, 0, 0);
        d_bodyRotation[i].set(1, 0, 0, 0);
    }
    for (int i = 0; i < 3; i++)
    {
        heading[i] = pitch[i] = 0.0;
    }
    carTrans.makeIdentity();
    setModified();
    init();
}

VrmlNodeVehicle::VrmlNodeVehicle(const VrmlNodeVehicle &n)
    : VrmlNodeChild(n)
    , d_brakePedalRotation(n.d_brakePedalRotation)
    , d_clutchPedaRotation(n.d_clutchPedaRotation)
    , d_gasPedalRotation(n.d_gasPedalRotation)
    , d_steeringWheelRotation(n.d_steeringWheelRotation)
    , d_wheelRotation(n.d_wheelRotation)
    , d_carRotation(n.d_carRotation)
    , d_carTranslation(n.d_carTranslation)
    , d_carBodyRotation(n.d_carBodyRotation)
    , d_carBodyTranslation(n.d_carBodyTranslation)
    , d_axle1Rotation(n.d_axle1Rotation)
    , d_axle1Translation(n.d_axle1Translation)
    , d_axle2Rotation(n.d_axle2Rotation)
    , d_axle2Translation(n.d_axle2Translation)

    , d_wheelFLTranslation(n.d_wheelFLTranslation)
    , d_wheelFLRotation(n.d_wheelFLRotation)
    , d_wheelFRTranslation(n.d_wheelFRTranslation)
    , d_wheelFRRotation(n.d_wheelFRRotation)
    , d_wheelRLTranslation(n.d_wheelRLTranslation)
    , d_wheelRLRotation(n.d_wheelRLRotation)
    , d_wheelRRTranslation(n.d_wheelRRTranslation)
    , d_wheelRRRotation(n.d_wheelRRRotation)

    //Ghost
    , d_ffz1wheelFLTranslation(n.d_ffz1wheelFLTranslation)
    , d_ffz1wheelFLRotation(n.d_ffz1wheelFLRotation)
    , d_ffz1wheelFRTranslation(n.d_ffz1wheelFRTranslation)
    , d_ffz1wheelFRRotation(n.d_ffz1wheelFRRotation)
    , d_ffz1wheelRLTranslation(n.d_ffz1wheelRLTranslation)
    , d_ffz1wheelRLRotation(n.d_ffz1wheelRLRotation)
    , d_ffz1wheelRRTranslation(n.d_ffz1wheelRRTranslation)
    , d_ffz1wheelRRRotation(n.d_ffz1wheelRRRotation)

    , d_wheelRotations(n.d_wheelRotations)
    , d_wheelTranslations(n.d_wheelTranslations)
    , d_cameraRotation(n.d_cameraRotation)
    , d_cameraTranslation(n.d_cameraTranslation)
    , d_mirrorLRotation(n.d_mirrorLRotation)
    , d_mirrorMRotation(n.d_mirrorMRotation)
    , d_mirrorRRotation(n.d_mirrorRRotation)

    , d_setOffset(n.d_setOffset)
    , d_offsetLoc(n.d_offsetLoc)
    , d_offsetRot(n.d_offsetRot)
    , d_printTransformation(n.d_printTransformation)

    , d_numCars(n.d_numCars)
    , d_followTerrain(n.d_followTerrain)
    , d_speed(n.d_speed)
    , d_revs(n.d_revs)
    , d_acceleration(n.d_acceleration)
    , d_aMax(n.d_aMax)
    , d_vMax(n.d_vMax)
    , d_gear(n.d_gear)
    , d_mirrorLightLeft(n.d_mirrorLightLeft)
    , d_mirrorLightRight(n.d_mirrorLightRight)

    , d_ffz1Rotation(n.d_ffz1Rotation)
    , d_ffz1Translation(n.d_ffz1Translation)

    //additional Data
    , d_float_values(n.d_float_values)

    , d_int_values(n.d_int_values)
{
    int i;
    for (i = 0; i < MAX_BODIES; i++)
    {
        bodyTransName[i] = new char[20];
        sprintf(bodyTransName[i], "body%dTranslation", i + 1);
        bodyRotName[i] = new char[20];
        sprintf(bodyRotName[i], "body%dRotation", i + 1);
        d_bodyTranslation[i].set(0, 0, 0);
        d_bodyRotation[i].set(1, 0, 0, 0);
    }
    carTrans.makeIdentity();
    setModified();
    init();
}

float VrmlNodeVehicle::getA(float v, float vmin, float vmax)
{
    float vd = vmax - vmin;
    float vn = (((v - vmin) / vd) * 2) - 1;
    float an = 1 - (vn * vn);
    if (an > 0)
        return an;
    return 0;
}

osg::Matrix *VrmlNodeVehicle::getCarTransMatrix()
{
    return &carTrans;
}

void VrmlNodeVehicle::init()
{
#ifndef USE_CAR_SOUND
    source = NULL;
    gearSound = NULL;
    hornSound = NULL;

    if (player == NULL)
    {
        player = cover->usePlayer(playerUnavailableCB);
        if (player == NULL)
        {
            cover->unusePlayer(playerUnavailableCB);
            cover->addPlugin("Vrml97");
            player = cover->usePlayer(playerUnavailableCB);
            if (player == NULL)
            {
                cerr << "sorry, no VRML, no Sound support " << endl;
            }
        }
    }
    engineSound = new EngineSound(player);
    Audio *engineAudio = new Audio("porsche.wav");
    Audio *gearAudio = new Audio("shift_miss.wav");
    Audio *hornAudio = new Audio("horn.wav");
    if (player)
    {
        source = player->newSource(engineAudio);
        if (source)
        {
            source->setLoop(true);
            source->stop();
            source->setIntensity(1.0);
        }
        gearSound = player->newSource(gearAudio);
        if (gearSound)
        {
            gearSound->setLoop(false);
            gearSound->stop();
            gearSound->setIntensity(1.0);
        }
        hornSound = player->newSource(hornAudio);
        if (hornSound)
        {
            hornSound->setLoop(true);
            hornSound->stop();
            hornSound->setIntensity(1.0);
        }
    }
#else
    carSound = CarSound::instance();
#endif

    cerr << "InputDevice == " << InputDevice::instance()->getName() << endl;

    //   autodetect();

    //double timeStamp = System::the->time();
}

VrmlNodeVehicle::~VrmlNodeVehicle()
{
    /*int i;
    for (i = 0; i < MAX_BODIES; i++)
    {
        delete[] bodyTransName[i];
        delete[] bodyRotName[i];
    }
    cerr << "~VrmlNodeVehicle called" << endl;

#ifdef USE_CAR_SOUND
    delete carSound;
#else
    delete source;
    delete gearSound;
    delete hornSound;
#endif */
}

VrmlNodeVehicle *VrmlNodeVehicle::toVehicleWheel() const
{
    return (VrmlNodeVehicle *)this;
}

void VrmlNodeVehicle::recalcMatrix()
{
    float *ct = d_carTranslation.get();
    osg::Vec3 tr(ct[0], ct[1], ct[2]);
    carTrans.makeTranslate(tr);
    osg::Matrix rot;
    ct = d_carRotation.get();
    tr.set(ct[0], ct[1], ct[2]);
    rot.makeRotate(ct[3], tr);
    carTrans.preMult(rot);
}

void VrmlNodeVehicle::eventIn(double timeStamp,
                              const char *eventName,
                              const VrmlField *fieldValue)
{
    if (strcmp(eventName, "reset") == 0)
    {
        setField(eventName, *fieldValue);
        if (d_reset.get() == true)
        {
            std::cout << "VRML-Script reset" << std::endl;
            d_wheelRotation.set(1, 0, 0, 0);
            d_carRotation.set(1, 0, 0, 0);
            d_carTranslation.set(0, 0, 0);
            d_carBodyRotation.set(1, 0, 0, 0);
            d_carBodyTranslation.set(0, 0, 0);
            d_speed.set(0);
            d_revs.set(0);
            d_mirrorLightLeft.set(0);
            d_mirrorLightRight.set(0);
            d_acceleration.set(0);
            d_ffz1Rotation.set(1, 0, 0, 0);
            d_ffz1Translation.set(0, 0, 0);
            if (SteeringWheelPlugin::plugin->dynamics)
            {
                SteeringWheelPlugin::plugin->dynamics->setVehicleTransformation(carTrans);
                SteeringWheelPlugin::plugin->dynamics->resetState();
            }
            else
                carTrans.makeIdentity();
        }
    }
    else if (strcmp(eventName, "carRotation") == 0)
    {
        setField(eventName, *fieldValue);
        recalcMatrix();
        if (SteeringWheelPlugin::plugin->dynamics)
            SteeringWheelPlugin::plugin->dynamics->setVehicleTransformation(carTrans);
    }
    else if (strcmp(eventName, "carTranslation") == 0)
    {
        setField(eventName, *fieldValue);
        recalcMatrix();
        if (SteeringWheelPlugin::plugin->dynamics)
            SteeringWheelPlugin::plugin->dynamics->setVehicleTransformation(carTrans);
    }
    else if (strcmp(eventName, "numCars") == 0)
    {
        setField(eventName, *fieldValue);
    }
    else if (strcmp(eventName, "followTerrain") == 0)
    {
        setField(eventName, *fieldValue);
    }

    for (size_t i = 0; i < d_carTranslations.size(); i++)
    {
        if(eventName == ("carTranslation_" + std::to_string(i)))
        {
            setField(eventName, *fieldValue);
            break;
        }
    }

    if (strcmp(eventName, "offsetLoc") == 0)
    {
        setField(eventName, *fieldValue);
    }
    else if (strcmp(eventName, "offsetRot") == 0)
    {
        setField(eventName, *fieldValue);
    }
    else if (strcmp(eventName, "setOffset") == 0)
    {
        setField(eventName, *fieldValue);
        if (d_setOffset.get() == true)
        {
            PorscheRealtimeDynamics *dynamics = dynamic_cast<PorscheRealtimeDynamics *>(SteeringWheelPlugin::plugin->dynamics);
            if (dynamics)
            {
                float *loc = d_offsetLoc.get();
                float *rot = d_offsetRot.get();
                if (coVRMSController::instance()->isMaster())
                {
                    cout << "Set Offset Location to: " << loc[0] << " " << loc[1] << " " << loc[2] << endl;
                    cout << "Set Offset Rotation to: (" << rot[0] << " " << rot[1] << " " << rot[2] << "), " << rot[3] << endl;
                }
                osg::Matrix offset;
                offset.setRotate(osg::Quat(rot[0], rot[1], rot[2], rot[3]));
                offset.setTrans(loc[0], loc[1], loc[2]);
                dynamics->setVehicleTransformationOffset(offset);
            }
        }
    }
    else if (strcmp(eventName, "printTransformation") == 0)
    {
        setField(eventName, *fieldValue);
        if (SteeringWheelPlugin::plugin->dynamics)
        {
            if (d_printTransformation.get() == true && coVRMSController::instance()->isMaster())
            {
                osg::Vec3 loc = SteeringWheelPlugin::plugin->dynamics->getVehicleTransformation().getTrans();
                osg::Quat rot = SteeringWheelPlugin::plugin->dynamics->getVehicleTransformation().getRotate();
                cout << "Transformation: " << endl;
                cout << " " << loc[0] << " " << loc[1] << " " << loc[2] << endl;
                cout << " (" << rot[0] << " " << rot[1] << " " << rot[2] << "), " << rot[3] << endl; // axis, angle
            }
        }
    }

    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeVehicle::moveToStreet()
{
    this->moveToStreet(this->carTrans);
}

void VrmlNodeVehicle::moveToStreet(osg::Matrix &carTrans)
{
    //float carHeight=200;
    //  just adjust height here

    osg::Matrix carTransWorld, tmp;
    osg::Vec3 pos, p0, q0;

    osg::Matrix baseMat;
    osg::Matrix invBaseMat;

    baseMat = cover->getObjectsScale()->getMatrix();

    osg::Matrix transformMatrix = cover->getObjectsXform()->getMatrix();

    baseMat.postMult(transformMatrix);
    invBaseMat.invert(baseMat);

    osg::Matrix invVRMLRotMat;
    invVRMLRotMat.makeRotate(-M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    osg::Matrix VRMLRotMat;
    VRMLRotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));

    carTransWorld = carTrans * VRMLRotMat * baseMat;
    osg::Vec3 test_vec = carTrans.getTrans();
    pos = carTransWorld.getTrans();

    //savePos(pos[0],pos[1]);
    //std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [0]: " << test_vec[0] << " | [1]: " << test_vec[1] << " | [2]: " << test_vec[2] << std::endl;
    savePos(test_vec[0], test_vec[2] * -1);

    //pos[2] -= carHeight;
    // down segment
    p0.set(pos[0], pos[1], pos[2] + 1500.0);
    q0.set(pos[0], pos[1], pos[2] - 40000.0);

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector[2];

    intersector[0] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[0]);

    // down segment 2
    p0.set(pos[0], pos[1] + 1000, pos[2] + 1500.0);
    q0.set(pos[0], pos[1] + 1000, pos[2] - 40000.0);
    intersector[1] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[1]);

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Collision);

    cover->getObjectsXform()->accept(visitor);

    bool hit1 = intersector[0]->containsIntersections();
    bool hit2 = intersector[1]->containsIntersections();
    if (hit1 || hit2)
    {
        if (hit1 || hit2)
        {
            float dist = 0.0;
            osg::Vec3 normal(0, 0, 1);
            osg::Vec3 normal2(0, 0, 1);
            osg::Node *geode = NULL;
            if (hit1 && !hit2)
            {
                auto isect = intersector[0]->getFirstIntersection();
                normal = isect.getWorldIntersectNormal();
                dist = pos[2] - isect.getWorldIntersectPoint()[2];
                geode = *isect.nodePath.end();
            }
            else if (!hit1 && hit2)
            {
                auto isect = intersector[1]->getFirstIntersection();
                normal = isect.getWorldIntersectNormal();
                dist = pos[2] - isect.getWorldIntersectPoint()[2];
                geode = *isect.nodePath.end();
            }
            else if (hit1 && hit2)
            {

                auto isect1 = intersector[0]->getFirstIntersection();
                auto isect2 = intersector[1]->getFirstIntersection();
                normal = isect1.getWorldIntersectNormal();
                dist = pos[2] - isect1.getWorldIntersectPoint()[2];
                geode = *isect1.nodePath.end();

                normal2 = isect2.getWorldIntersectNormal();

                normal += normal2;
                normal *= 0.5;

                if (fabs(pos[2] - isect2.getWorldIntersectPoint()[2]) < fabs(dist))
                {
                    dist = pos[2] - isect2.getWorldIntersectPoint()[2];
                    geode = *isect2.nodePath.end();
                }
            }
            if (geode)
            {
                std::string geodeName = geode->getName();
                if (!geodeName.empty())
                {
                    if ((geodeName.find("ROAD")) != std::string::npos)
                    {
                        if (SteeringWheelPlugin::plugin->sitzkiste)
                        {
                            SteeringWheelPlugin::plugin->sitzkiste->setRoadFactor(0.0); //0 == Road 1 == rough
                        }
                        if (SteeringWheelPlugin::plugin->dynamics)
                        {
                            SteeringWheelPlugin::plugin->dynamics->setRoadType(0);
                            lastRoadPos = carTrans;
                        }
                    }
                    else
                    {
                        if (SteeringWheelPlugin::plugin->sitzkiste)
                        {
                            SteeringWheelPlugin::plugin->sitzkiste->setRoadFactor(1.0); //0 == Road 1 == rough
                        }
                        if (SteeringWheelPlugin::plugin->dynamics)
                        {
                            SteeringWheelPlugin::plugin->dynamics->setRoadType(1);
                        }
                    }
                }
            }
            osg::Vec3 carNormal(carTransWorld(1, 0), carTransWorld(1, 1), carTransWorld(1, 2));
            //osg::Vec3 carNormal(carTrans(1,0),carTrans(1,1),carTrans(1,2));
            //osg::Vec3 carNormal(carTrans(1,0),-carTrans(1,2),carTrans(1,1));
            tmp.makeTranslate(0, 0, -dist);
            osg::Matrix rMat;
            carNormal.normalize();
            osg::Vec3 upVec(0.0, 0.0, 1.0);
            float sprod = upVec * normal;
            if (sprod < 0)
                normal *= -1;
            sprod = upVec * normal;
            //cerr <<" carNormal: " << carNormal[0]  << " "<< carNormal[1] << " "<< carNormal[2] << endl;
            //cerr <<" normal: " << normal[0]<< " " << normal[1]<< " " << normal[2] << endl;
            //cerr <<" sprod: " << sprod << endl;
            if (sprod > 0.8) // only rotate the car if the angle is not more the 45 degrees
            {
                rMat.makeRotate(carNormal, normal);
                tmp.preMult(rMat);
                carTrans.postMult(VRMLRotMat * baseMat * tmp * invBaseMat * invVRMLRotMat);
            }
            else
            {

                carTrans.postMult(VRMLRotMat * baseMat * tmp * invBaseMat * invVRMLRotMat);
            }
        }
    }
}

void VrmlNodeVehicle::getGroundDistance(osg::Matrix &wheelFLMatrix, osg::Matrix &wheelFRMatrix, osg::Matrix &wheelRLMatrix, osg::Matrix &wheelRRMatrix, double &distFL, double &distFR, double &distRL, double &distRR)
{
    osg::Vec3 intersectFL;
    osg::Vec3 intersectFR;
    osg::Vec3 intersectRL;
    osg::Vec3 intersectRR;
    getGroundIntersectPoints(wheelFLMatrix, wheelFRMatrix, wheelRLMatrix, wheelRRMatrix, intersectFL, intersectFR, intersectRL, intersectRR);

    distFL = (wheelFLMatrix.getTrans())[1] - intersectFL[1];
    distFR = (wheelFRMatrix.getTrans())[1] - intersectFR[1];
    distRL = (wheelRLMatrix.getTrans())[1] - intersectRL[1];
    distRR = (wheelRRMatrix.getTrans())[1] - intersectRR[1];
}

void VrmlNodeVehicle::getGroundIntersectPoints(osg::Matrix &wheelFLMatrix, osg::Matrix &wheelFRMatrix, osg::Matrix &wheelRLMatrix, osg::Matrix &wheelRRMatrix, osg::Vec3 &intersectFL, osg::Vec3 &intersectFR, osg::Vec3 &intersectRL, osg::Vec3 &intersectRR)
{
    osg::Matrix baseMat = cover->getObjectsScale()->getMatrix();
    osg::Matrix transformMatrix = cover->getObjectsXform()->getMatrix();
    baseMat.postMult(transformMatrix);
    osg::Matrix VRMLRotMat;
    VRMLRotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    osg::Matrix vrmlToBase = VRMLRotMat * baseMat;
    osg::Matrix baseToVrml = osg::Matrix::inverse(vrmlToBase);
    
    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector[4];
    osg::Matrix tmpMat = wheelFLMatrix * vrmlToBase;
    intersector[0] = coIntersection::instance()->newIntersector(tmpMat.preMult(osg::Vec3d(0,1,0)), tmpMat.preMult(osg::Vec3d(0,-1,0)));
    igroup->addIntersector(intersector[0]);
    tmpMat = wheelFRMatrix * vrmlToBase;
    intersector[1] = coIntersection::instance()->newIntersector(tmpMat.preMult(osg::Vec3d(0, 1, 0)), tmpMat.preMult(osg::Vec3d(0, -1, 0)));
    igroup->addIntersector(intersector[1]);
    tmpMat = wheelRLMatrix * vrmlToBase;
    intersector[2] = coIntersection::instance()->newIntersector(tmpMat.preMult(osg::Vec3d(0, 1, 0)), tmpMat.preMult(osg::Vec3d(0, -1, 0)));
    igroup->addIntersector(intersector[2]);
    tmpMat = wheelRRMatrix * vrmlToBase;
    intersector[3] = coIntersection::instance()->newIntersector(tmpMat.preMult(osg::Vec3d(0, 1, 0)), tmpMat.preMult(osg::Vec3d(0, -1, 0)));
    igroup->addIntersector(intersector[3]);

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Collision);
    cover->getObjectsXform()->accept(visitor);
    if (intersector[0]->containsIntersections())
    {
        intersectFL = baseToVrml.preMult(intersector[0]->getFirstIntersection().getWorldIntersectPoint());
    }
    else
    {
        intersectFL = osg::Vec3(0, 0, 0);
    }
    if (intersector[1]->containsIntersections())
    {
        intersectFR = baseToVrml.preMult(intersector[1]->getFirstIntersection().getWorldIntersectPoint());
    }
    else
    {
        intersectFR = osg::Vec3(0, 0, 0);
    }
    if (intersector[2]->containsIntersections())
    {
        intersectRL = baseToVrml.preMult(intersector[2]->getFirstIntersection().getWorldIntersectPoint());
    }
    else
    {
        intersectRL = osg::Vec3(0, 0, 0);
    }
    if (intersector[3]->containsIntersections())
    {
        intersectRR = baseToVrml.preMult(intersector[3]->getFirstIntersection().getWorldIntersectPoint());
    }
    else
    {
        intersectRR = osg::Vec3(0, 0, 0);
    }
}

void VrmlNodeVehicle::render(Viewer *)
{
    InputDevice::instance()->updateInputState();
    float steeringWheelAngle = InputDevice::instance()->getSteeringWheelAngle();
    //float accelerationPedal = InputDevice::instance()->getAccelerationPedal();
    //float breakPedal = InputDevice::instance()->getBrakePedal();
    float clutchPedal = InputDevice::instance()->getClutchPedal();
    int gear = InputDevice::instance()->getGear();
    bool hornButton = InputDevice::instance()->getHornButton();
    bool resetButton = InputDevice::instance()->getResetButton();
    int mirrorLightLeft = InputDevice::instance()->getMirrorLightLeft();
    int mirrorLightRight = InputDevice::instance()->getMirrorLightRight();

    double v = 0.0;
    double a = 0.0;
    double enginespeed = 0.0;

    timeStamp = System::the->time();

    d_steeringWheelRotation.set(0, 0, 1, steeringWheelAngle);

#ifdef USE_CAR_SOUND
    static bool oldHornButton = false;
    if (hornButton != oldHornButton)
    {
        oldHornButton = hornButton;
        if (hornButton)
            carSound->start(CarSound::Horn);
        else
            carSound->stop(CarSound::Horn);
    }
#else
    if (hornSound)
    {
        if (hornButton && !hornSound->isPlaying())
        {
            std::cerr << "TUUUUUUUUUUUUUUUUUUUT" << std::endl;
            hornSound->play();
        }
        else if (!hornButton && hornSound->isPlaying())
        {
            hornSound->stop();
        }
    }
#endif

    /*
	static int loop = 0;
   ++loop;
   if(loop > 19) {
      std::cerr << "Steering Wheel Angle: " << steeringWheelAngle << std::endl;
      std::cerr << "Acceleration Pedal: " << accelerationPedal << std::endl;
      std::cerr << "Brake Pedal: " << breakPedal << std::endl;
      std::cerr << "Clutch Pedal: " << clutchPedal << std::endl;
      std::cerr << "Gear: " << gear << std::endl;
      std::cerr << "Horn Button: " << hornButton << std::endl;
      std::cerr << "Reset Button: " << resetButton << std::endl;
      loop = 0;
   }
   */

    /*
   double dT =cover->frameDuration();
   float wheelBase = 2.5;
   float v = d_speed.get();
   float aDec=-0.2-0.2*v;
   if(v<0)
      aDec=0.2-0.2*v;
   float aAcc=0.0;
   //float vM = d_vMax.get();
   float aM = d_aMax.get();

   float wheelAngle=0.0;
   int gear=d_gear.get();
   float accelerationPedal=0; // Gaspedal 0 - 1
   float breakPedal=0; // Bremspedal 0 - 1
   float clutchPedal=0; // Kupplung 0 - 1
   bool resetButton=false;

   if(inputDevice == VEHICLE_INPUT_PORSCHE_REALTIME_SIM)
   {
      float oldy = carTrans(3,1);
      carTrans.makeIdentity();
      carTrans.setTrans(cover->axes[0][2],cover->axes[0][3],-cover->axes[0][1]);
      carTrans(3,1) = oldy;

      float heading,pitch,roll;
      heading = (-cover->axes[0][4]*180.0/M_PI) +0.0f;
      pitch = cover->axes[0][5]*180.0f/M_PI;
      roll = cover->axes[0][6]*180.0f/M_PI;
   osg::Matrix rot;
   MAKE_EULER_MAT(rot,roll, pitch, heading);
   carTrans.mult(rot,carTrans);

   moveToStreet();

   osg::Quat q;
   q.set(carTrans);
   osg::Quat::value_type orient[4];
   q.getRotate(orient[3], orient[0], orient[1], orient[2]);
   d_carTranslation.set(carTrans(3,0), carTrans(3,1),carTrans(3,2));
   d_carRotation.set(orient[0],orient[1],orient[2],orient[3]);
   double timeStamp = System::the->time();
   d_speed.set(cover->axes[0][31]);
   //d_acceleration.set(a);

   eventOut(timeStamp, "carTranslation", d_carTranslation);
   eventOut(timeStamp, "carRotation", d_carRotation);
   eventOut(timeStamp, "speed", d_speed);
   //eventOut(timeStamp, "acceleration", d_acceleration);
   //eventOut(timeStamp, "steeringWheelRotation", d_steeringWheelRotation);
   return;
   }
	*/

    // ,,,Input device reimplementation,,,

    /*
   VehicleDynamicsInstitutes *vd = SteeringWheelPlugin::plugin->vd;
   if(vd && vd->doRun)
   {
      vd->setSteeringWheelAngle(-(steeringWheelAngle/M_PI)*180);
      vd->setGas(accelerationPedal);
      vd->setBrake(breakPedal);
      vd->setGear(gear);
      vd->setClutch(clutchPedal);
      vd->reset(resetButton);
      osg::Vec3 carPos = carTrans.getTrans();
      coCoord coord=carTrans;
      vd->setHeight(carPos[1]);
      vd->setOrientation(coord.hpr[1],coord.hpr[2]);
   }

   if(gear!= d_gear.get())
   {
      double timeStamp = System::the->time();
      d_gear.set(gear);
      eventOut(timeStamp, "gear", d_gear);
      if(gear != 0 && v > 0.2 && clutchPedal < 0.4)
      {
          if(gearSound)
          {
              gearSound->play();
	  }
      }
   }
   float dz=0;
   if(gear == -1)
   {
      aAcc = -aM * accelerationPedal * getA(v,-10,30);
      dz = -v/30;
   }
   if(gear == 0)
   {
      aAcc = 0;
      dz = 0;
   }
   if(gear == 1)
   {
      aAcc = aM * accelerationPedal * getA(v,-10,30);
      dz = v/30;
   }
   else if(gear == 2)
   {
      aAcc = aM * accelerationPedal * getA(v,0,40);
      dz = (v-0)/40;
   }
   else if(gear == 3)
   {
      aAcc = aM * accelerationPedal * getA(v,10,50);
      dz = (v-10)/40;
   }
   else if(gear == 4)
   {
      aAcc = aM * accelerationPedal * getA(v,15,70);
      dz = (v-15)/55;
   }
   else if(gear == 5)
   {
      aAcc = aM * accelerationPedal * getA(v,20,100);
      dz = (v-20)/80;
   }
   else if(gear == 6) // r
   {
      aAcc = -aM * accelerationPedal * getA(v,-10,20);
      dz = (-v)/20;
   }


   // aDec = -.2 stillstand
   // aDec < -.2 vorwaertsbeschl
   // aDec > -.2 rueckwaerts

   //fprintf(stderr,"aDec %f\n",aDec+.2);
   if(aDec+.2 < 0)
   {
      aDec += aM * -0.5 * breakPedal;
   }
   else
   {
      aDec += aM * 0.5 * breakPedal;
   }


   if(clutchPedal > 0.7)
   {
      aAcc = 0;
   }
   else if(clutchPedal > 0.3)
   {
      aAcc *= (1.0-((clutchPedal-0.3)/0.4));
   }
   static float oldaccelerationPedal=0;
   if(oldaccelerationPedal!=accelerationPedal)
   {
   fprintf(stderr,"accelerationPedal: %f\n",accelerationPedal);
   oldaccelerationPedal=accelerationPedal;
   }
   static float oldbreakPedal=0;
   if(oldbreakPedal!=breakPedal)
   {
   fprintf(stderr,"breakPedal: %f\n",breakPedal);
   oldbreakPedal=breakPedal;
   }
   static float oldclutchPedal=0;
   if(oldclutchPedal!=clutchPedal)
   {
   fprintf(stderr,"clutchPedal: %f\n",clutchPedal);
   oldclutchPedal=clutchPedal;
   }
   static int oldGear=0;
   if(oldGear!=gear)
   {
   fprintf(stderr,"gear: %d\n",gear);
   oldGear=gear;
   }
   //fprintf(stderr,"aAcc: %f\n",aAcc);
   float a;
   a = aAcc+aDec;
   float dV = a*dT;
   if(v>0)
   {
      v = v+dV;
      if(v<0) // stop
         v=0;
   }
   else if(v<0)
   {
      v = v+dV;
      if(v>0) // stop
         v=0;
   }
   else
   {
      if(aAcc > 0.001 || aAcc < -0.001)
      {
         v = v+dV;
      }
   }
   //fprintf(stderr,"v m/s: %f",v);
   //fprintf(stderr,"v km/h: %f",v*60*60/1000);
      if(source)
      {

      if(sdz > 0 || v > 0)
      {
            if(!source->isPlaying())
            {
               source->play();
            }
	    }
	    else
	    {
            if(source->isPlaying())
            {
               source->stop();
            }
	    }

         if(clutchPedal > 0.3) // Kupplung gedrueckt
         {
	    // tf laeuft 3 sekunden vom Druecken der kuplung an von 0 - 1
	    mdz -= (dT/3.0);
	    if(accelerationPedal > 0.01)
	    {
	    mdz += accelerationPedal*dT*4;
	    if(mdz > 1.0)
	        mdz = 1.0;
	    }
         if(clutchPedal <= 0.7) // Kupplung ist teilweise im Eingriff
         {
	    float kf = (clutchPedal-0.3)/0.4;
            //source->setPitch((4000+16000*accelerationPedal)*kf*tf+(6000+(dz)*16000)*(1-(kf*tf)));
	    mdz += ((dz -mdz) * (1.0-kf));
         }
	    sdz = mdz;
         }
         else
         {
	    sdz = mdz = dz;
            //source->setPitch(6000+(dz)*16000);
         }
         //source->setPitch(6000+(sdz)*16000);
      }


   //timeStep();

   float s = v*dT;
   osg::Vec3 V(0,0,-s);

   float rotAngle=0.0;
   //if((vehicleParameters->getWheelAngle()>-0.0001 && vehicleParameters->getWheelAngle()><0.0001 )|| (s < 0.0001 && s > -0.0001)) // straight
   if((s < 0.0001 && s > -0.0001)) // straight
   {
   }
   else
   {
      //float r = tan(M_PI_2-vehicleParameters->getWheelAngle()) * wheelBase;
      float r = tan(M_PI_2-steeringWheelAngle*0.2) * wheelBase;
      float u = 2.0*r*M_PI;
      rotAngle = (s/u)*2.0*M_PI;
      V[2]=-r*sin(rotAngle);
      V[0]=r-r*cos(rotAngle);
   }
   */

    /*
   if(vd && vd->doRun)
   {
      carTrans= vd->getCarTransform();
      v = vd->getVelocity();
      sdz = vd->getRevolution();

      FKFSDynamics *fkfsvd = dynamic_cast<FKFSDynamics*>(SteeringWheelPlugin::plugin->vd);
      if(fkfsvd)
      {
         osg::Matrix bodyTrans;
         osg::Matrix cameraTrans;
         int i;
         cameraTrans = fkfsvd->getCameraTransform();
         osg::Quat q;
         double timeStamp = System::the->time();
         osg::Quat::value_type orient[4];
         for(i=0;i<fkfsvd->appNumObjects;i++)
         {
            bodyTrans = fkfsvd->getBodyTransform(i);
            q.set(bodyTrans);
            q.getRotate(orient[3], orient[0], orient[1], orient[2]);
            d_bodyTranslation[i].set(bodyTrans(3,0), bodyTrans(3,1),bodyTrans(3,2));
            d_bodyRotation[i].set(orient[0],orient[1],orient[2],orient[3]);
            eventOut(timeStamp, bodyTransName[i], d_bodyTranslation[i]);
            eventOut(timeStamp, bodyRotName[i], d_bodyRotation[i]);
         }
         q.set(cameraTrans);
         q.getRotate(orient[3], orient[0], orient[1], orient[2]);
         d_cameraTranslation.set(cameraTrans(3,0), cameraTrans(3,1),cameraTrans(3,2));
         d_cameraRotation.set(orient[0],orient[1],orient[2],orient[3]);
         eventOut(timeStamp, "cameraTranslation", d_cameraTranslation);
         eventOut(timeStamp, "cameraRotation", d_cameraRotation);

      }

      ITM *itm = dynamic_cast<ITM*>(SteeringWheelPlugin::plugin->vd);
      if(itm)
      {
         int i,j;
         osg::Matrix axle1Trans;
         osg::Matrix axle2Trans;
         osg::Matrix wheel1Trans;
         osg::Matrix wheel2Trans;
         osg::Matrix wheel3Trans;
         osg::Matrix wheel4Trans;
         if(itm->haveWheels)
         {
            axle1Trans.makeIdentity();
            axle2Trans.makeIdentity();
            wheel1Trans.makeIdentity();
            wheel2Trans.makeIdentity();
            wheel3Trans.makeIdentity();
            wheel4Trans.makeIdentity();
            for(i=0;i<3;i++)
            {
        	  axle1Trans(3,i)=itm->appReceiveBuffer.axle[0][9+i];
        	  axle2Trans(3,i)=itm->appReceiveBuffer.axle[1][9+i];
        	  wheel1Trans(3,i)=itm->appReceiveBuffer.wheel[0][9+i];
        	  wheel2Trans(3,i)=itm->appReceiveBuffer.wheel[1][9+i];
        	  wheel3Trans(3,i)=itm->appReceiveBuffer.wheel[2][9+i];
        	  wheel4Trans(3,i)=itm->appReceiveBuffer.wheel[3][9+i];
            }
            for(i=0;i<3;i++)
            {
               for(j=0;j<3;j++)
               {
        	  axle1Trans(i,j)=itm->appReceiveBuffer.axle[0][j*3+i];
        	  axle2Trans(i,j)=itm->appReceiveBuffer.axle[1][j*3+i];
        	  wheel1Trans(i,j)=itm->appReceiveBuffer.wheel[0][j*3+i];
        	  wheel2Trans(i,j)=itm->appReceiveBuffer.wheel[1][j*3+i];
        	  wheel3Trans(i,j)=itm->appReceiveBuffer.wheel[2][j*3+i];
        	  wheel4Trans(i,j)=itm->appReceiveBuffer.wheel[3][j*3+i];
               }
            }
   	    osg::Quat q;
   	    double timeStamp = System::the->time();
   	    osg::Quat::value_type orient[4];
   	    q.set(axle1Trans);
   	    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
   	    d_axle1Translation.set(axle1Trans(3,0), axle1Trans(3,1),axle1Trans(3,2));
   	    d_axle1Rotation.set(orient[0],orient[1],orient[2],orient[3]);
   	    eventOut(timeStamp, "axle1Translation", d_axle1Translation);
   	    eventOut(timeStamp, "axle1Rotation", d_axle1Rotation);
   	    q.set(axle2Trans);
   	    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
   	    d_axle2Translation.set(axle2Trans(3,0), axle2Trans(3,1),axle2Trans(3,2));
   	    d_axle2Rotation.set(orient[0],orient[1],orient[2],orient[3]);
   	    eventOut(timeStamp, "axle2Translation", d_axle2Translation);
   	    eventOut(timeStamp, "axle2Rotation", d_axle2Rotation);
   	    q.set(wheel1Trans);
   	    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
   	    d_wheel1Translation.set(wheel1Trans(3,0), wheel1Trans(3,1),wheel1Trans(3,2));
   	    d_wheel1Rotation.set(orient[0],orient[1],orient[2],orient[3]);
   	    eventOut(timeStamp, "wheel1Translation", d_wheel1Translation);
   	    eventOut(timeStamp, "wheel1Rotation", d_wheel1Rotation);
   	    q.set(wheel2Trans);
   	    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
   	    d_wheel2Translation.set(wheel2Trans(3,0), wheel2Trans(3,1),wheel2Trans(3,2));
   	    d_wheel2Rotation.set(orient[0],orient[1],orient[2],orient[3]);
   	    eventOut(timeStamp, "wheel2Translation", d_wheel2Translation);
   	    eventOut(timeStamp, "wheel2Rotation", d_wheel2Rotation);
   	    q.set(wheel3Trans);
   	    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
   	    d_wheel3Translation.set(wheel3Trans(3,0), wheel3Trans(3,1),wheel3Trans(3,2));
   	    d_wheel3Rotation.set(orient[0],orient[1],orient[2],orient[3]);
   	    eventOut(timeStamp, "wheel3Translation", d_wheel3Translation);
	       eventOut(timeStamp, "wheel3Rotation", d_wheel3Rotation);
          q.set(wheel4Trans);
          q.getRotate(orient[3], orient[0], orient[1], orient[2]);
          d_wheel4Translation.set(wheel4Trans(3,0), wheel4Trans(3,1),wheel4Trans(3,2));
	       d_wheel4Rotation.set(orient[0],orient[1],orient[2],orient[3]);
   	    eventOut(timeStamp, "wheel4Translation", d_wheel4Translation);
   	    eventOut(timeStamp, "wheel4Rotation", d_wheel4Rotation);
	      }
         moveToStreet();
      }

     if(source)
     {

        source->setPitch(sdz);
     }
       //fprintf(stderr,"pos: %f %f %f\n",carTrans(3,0),carTrans(3,1),carTrans(3,2));
         //qcarTrans.makeIdentity();
   }
   else if(SteeringWheelPlugin::plugin->dynamics) {
   */
    if (SteeringWheelPlugin::plugin->dynamics)
    {
        if (resetButton)
            SteeringWheelPlugin::plugin->dynamics->resetState();

        HLRSRealtimeDynamics *dyn = dynamic_cast<HLRSRealtimeDynamics *>(SteeringWheelPlugin::plugin->dynamics);
        if (dyn)
        {
            float rotationSpeed = 0.001;
            int js = dyn->getJoystickState() & 0x0f;
            int ms = (dyn->getJoystickState() >> 4) & 0x0f;
            int mirrorNumber = ms - 1;
            int changedMirror = 0;
            if (js == HLRSRealtimeDynamics::JS_RIGHT && heading[mirrorNumber] < 0.7f)
            {
                heading[mirrorNumber] += rotationSpeed;
                changedMirror = ms;
            }
            if (js == HLRSRealtimeDynamics::JS_LEFT && heading[mirrorNumber] > -0.7f)
            {
                heading[mirrorNumber] -= rotationSpeed;
                changedMirror = ms;
            }
            if (js == HLRSRealtimeDynamics::JS_UP && pitch[mirrorNumber] > -0.7f)
            {
                pitch[mirrorNumber] -= rotationSpeed;
                changedMirror = ms;
            }
            if (js == HLRSRealtimeDynamics::JS_DOWN && pitch[mirrorNumber] < 0.7f)
            {
                pitch[mirrorNumber] += rotationSpeed;
                changedMirror = ms;
            }
            if (changedMirror > 0)
            {
                osg::Quat::value_type orient[4];
                osg::Quat q;
                if (changedMirror == HLRSRealtimeDynamics::RIGHT_MIRROR)
                {
                    q.makeRotate(-heading[HLRSRealtimeDynamics::RIGHT_MIRROR - 1], osg::Z_AXIS, pitch[HLRSRealtimeDynamics::RIGHT_MIRROR - 1], osg::X_AXIS, 0.0, osg::Y_AXIS);
                    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
                    d_mirrorRRotation.set(orient[0], orient[1], orient[2], orient[3]);
                    eventOut(timeStamp, "mirrorRRotation", d_mirrorRRotation);
                }

                if (changedMirror == HLRSRealtimeDynamics::LEFT_MIRROR)
                {
                    q.makeRotate(-heading[HLRSRealtimeDynamics::LEFT_MIRROR - 1], osg::Z_AXIS, pitch[HLRSRealtimeDynamics::LEFT_MIRROR - 1], osg::X_AXIS, 0.0, osg::Y_AXIS);
                    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
                    d_mirrorLRotation.set(orient[0], orient[1], orient[2], orient[3]);
                    eventOut(timeStamp, "mirrorLRotation", d_mirrorLRotation);
                }

                if (changedMirror == HLRSRealtimeDynamics::MIDDLE_MIRROR) // take care, rotation axes are different here (who knows why)
                {
                    q.makeRotate(heading[HLRSRealtimeDynamics::MIDDLE_MIRROR - 1], osg::Z_AXIS, pitch[HLRSRealtimeDynamics::MIDDLE_MIRROR - 1], osg::Y_AXIS, 0.0, osg::X_AXIS);
                    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
                    d_mirrorMRotation.set(orient[0], orient[1], orient[2], orient[3]);
                    eventOut(timeStamp, "mirrorMRotation", d_mirrorMRotation);
                }
            }

            static int oldjs = 0;
            if (js != oldjs)
            {

                oldjs = js;
                fprintf(stderr, "js: %d\n", js);
            }
            static int oldms = 0;
            if (ms != oldms)
            {
                oldms = ms;
                fprintf(stderr, "ms: %d\n", ms);
            }
        }

        SteeringWheelPlugin::plugin->dynamics->move(this);

        v = SteeringWheelPlugin::plugin->dynamics->getVelocity();
        velocity = v;
        a = SteeringWheelPlugin::plugin->dynamics->getAcceleration();

        enginespeed = SteeringWheelPlugin::plugin->dynamics->getEngineSpeed();

        carTrans = SteeringWheelPlugin::plugin->dynamics->getVehicleTransformation();

#ifdef USE_CAR_SOUND
        carSound->setSlip((SteeringWheelPlugin::plugin->dynamics->getTyreSlipFL()+SteeringWheelPlugin::plugin->dynamics->getTyreSlipFR()+SteeringWheelPlugin::plugin->dynamics->getTyreSlipRL()+SteeringWheelPlugin::plugin->dynamics->getTyreSlipRR())/4.0);

        //fprintf(stderr,"SlipFL %f",SteeringWheelPlugin::plugin->dynamics->getTyreSlipFL());
        //fprintf(stderr,"SlipRL %f\n",SteeringWheelPlugin::plugin->dynamics->getTyreSlipRL());
        //fprintf(stderr,"Torque %f, v%f\n",SteeringWheelPlugin::plugin->dynamics->getEngineTorque()*1000, v);
        if (v > 0.5)
            carSound->setTorque(SteeringWheelPlugin::plugin->dynamics->getEngineTorque() * 1000);
        else
            carSound->setTorque(1.0);
        carSound->setCarSpeed(v);
        carSound->setSpeed(enginespeed * 60.0);
#else
        engineSound->setSpeed(enginespeed * 60.0);
#endif
        /*
		if(source)
      {
	      if(enginespeed > 0)
 	      {
   	      if(!source->isPlaying())
            {
               source->play();
            }
        		source->setPitch(6000+(enginespeed/(7500.0/60.0))*16000);
	  		}
		   else if(source->isPlaying())
         {
               source->stop();
         }
      }*/
    }
    /*
   else
   {
      osg::Matrix relTrans;
      osg::Matrix relRot;
      relRot.makeRotate(rotAngle,0,1,0);
      relTrans.makeTranslate(V);
      carTrans = relRot * relTrans * carTrans;
      moveToStreet();
   }
   */

    if (SteeringWheelPlugin::plugin->sitzkiste)
        SteeringWheelPlugin::plugin->sitzkiste->carVel = v;

    //osg::Quat q;
    //q.set(carTrans);
    //osg::Quat::value_type orient[4];
    //q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    //d_carTranslation.set(carTrans(3,0), carTrans(3,1),carTrans(3,2));
    //d_carRotation.set(orient[0],orient[1],orient[2],orient[3]);

    //osg::Quat qBody;
    //qBody.set(carBodyTrans);
    //osg::Quat::value_type orientBody[4];
    //qBody.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    //d_carBodyTranslation.set(carBodyTrans(3,0), carBodyTrans(3,1),carBodyTrans(3,2));
    //d_carBodyRotation.set(orientBody[0],orientBody[1],orientBody[2],orientBody[3]);

    double timeStamp = System::the->time();
    d_speed.set(v);
    d_revs.set(enginespeed);
    d_mirrorLightLeft.set(mirrorLightLeft);
    d_mirrorLightRight.set(mirrorLightRight);
    d_acceleration.set(a);

    eventOut(timeStamp, "speed", d_speed);
    eventOut(timeStamp, "revs", d_revs);
    eventOut(timeStamp, "mirrorLightLeft", d_mirrorLightLeft);
    eventOut(timeStamp, "mirrorLightRight", d_mirrorLightRight);
    eventOut(timeStamp, "acceleration", d_acceleration);
    eventOut(timeStamp, "steeringWheelRotation", d_steeringWheelRotation);

    if (gear != d_gear.get())
    {
        //if(gear != 0 && v > 0.2 && clutchPedal < 0.4 && !(InputDevice::instance()->getIsAutomatic()))
        if (gear != 0 && v > 0.2 && clutchPedal < 0.4 && !(InputDevice::instance()->getIsPowerShifting()))
        {
#ifdef USE_CAR_SOUND
            carSound->start(CarSound::GearMiss);
#else
            if (gearSound)
            {
                gearSound->play();
            }
#endif
        }

        d_gear.set(gear);
        eventOut(timeStamp, "gear", d_gear);
    }
    int i = 0;
    int numTrans = d_numCars.get();
    int numextra = 10;

    if (SteeringWheelPlugin::plugin->dataController != NULL)
    {
        SteeringWheelPlugin::plugin->dataController->numIntsOut = 4;
        SteeringWheelPlugin::plugin->dataController->numFloatsOut = numTrans * 3 + numextra;

        //   SteeringWheelPlugin::plugin->dataController->intValuesOut[0]=(int)(cover->frameTime()-1e9);
        //   SteeringWheelPlugin::plugin->dataController->intValuesOut[1]=(int)((cover->frameTime()-1e9-((double)SteeringWheelPlugin::plugin->intValuesOut[0]))*1000.0);
        SteeringWheelPlugin::plugin->dataController->intValuesOut[0] = (int)(((unsigned int)(cover->frameTime() * 1000)) & 0xfffffff);
        SteeringWheelPlugin::plugin->dataController->intValuesOut[1] = 0;
        SteeringWheelPlugin::plugin->dataController->intValuesOut[2] = (int)gear;
        SteeringWheelPlugin::plugin->dataController->intValuesOut[3] = (int)numTrans;

        SteeringWheelPlugin::plugin->dataController->floatValuesOut[0] = (float)(cover->frameDuration() * 1000.0);
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[1] = carTrans(3, 0);
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[2] = carTrans(3, 1);
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[3] = carTrans(3, 2);
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[4] = carTrans(0, 2);
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[5] = carTrans(1, 2);
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[6] = carTrans(2, 2);

        SteeringWheelPlugin::plugin->dataController->floatValuesOut[7] = v;
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[8] = enginespeed;
        SteeringWheelPlugin::plugin->dataController->floatValuesOut[9] = a;
        
        for (size_t i = 0; i < numTrans; i++)
            for (size_t j = 0; j < 3; j++)
                SteeringWheelPlugin::plugin->dataController->floatValuesOut[i * 3 + numextra + j] = d_carTranslations[i].get()[j];
    }
}

void VrmlNodeVehicle::setVRMLVehicleBody(const osg::Matrix &trans)
{
    osg::Quat qBody;
    qBody.set(trans);
    osg::Quat::value_type orientBody[4];
    qBody.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_carBodyRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_carBodyTranslation.set(trans(3, 0), trans(3, 1), trans(3, 2));

    eventOut(timeStamp, "carBodyTranslation", d_carBodyTranslation);
    eventOut(timeStamp, "carBodyRotation", d_carBodyRotation);
}

void VrmlNodeVehicle::setVRMLVehicleFFZBody(const osg::Matrix &trans)
{
    osg::Quat qBody;
    qBody.set(trans);
    osg::Quat::value_type orientBody[4];
    qBody.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_ffz1Rotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_ffz1Translation.set(trans(3, 0), trans(3, 1), trans(3, 2));

    eventOut(timeStamp, "ffz1Translation", d_ffz1Translation);
    eventOut(timeStamp, "ffz1Rotation", d_ffz1Rotation);
}

void VrmlNodeVehicle::setVRMLAdditionalData(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9)
{
    // //
    d_int_values[0] = i0;
    d_int_values[1] = i1;
    d_int_values[2] = i2;
    d_int_values[3] = i3;
    d_int_values[4] = i4;
    d_int_values[5] = i5;
    d_int_values[6] = i6;
    d_int_values[7] = i7;
    d_int_values[8] = i8;
    d_int_values[9] = i9;

    d_float_values[0] = f0;
    d_float_values[1] = f1;
    d_float_values[2] = f2;
    d_float_values[3] = f3;
    d_float_values[4] = f4;
    d_float_values[5] = f5;
    d_float_values[6] = f6;
    d_float_values[7] = f7;
    d_float_values[8] = f8;
    d_float_values[9] = f9;

    for (size_t i = 0; i < NUM_ADDITIONAL_FIELDS; i++)
    {
        eventOut(timeStamp, ("int_value" + std::to_string(i)).c_str(), d_int_values[i]);
        eventOut(timeStamp, ("float_value" + std::to_string(i)).c_str(), d_float_values[i]);
    }
}

void VrmlNodeVehicle::setVRMLVehicleBody(int body, const osg::Matrix &trans)
{
    osg::Quat qBody;
    qBody.set(trans);
    osg::Quat::value_type orientBody[4];
    qBody.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_bodyRotation[body].set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_bodyTranslation[body].set(trans(3, 0), trans(3, 1), trans(3, 2));

    eventOut(timeStamp, bodyTransName[body], d_bodyTranslation[body]);
    eventOut(timeStamp, bodyRotName[body], d_bodyRotation[body]);
}

void VrmlNodeVehicle::setVRMLVehicleCamera(const osg::Matrix &trans)
{
    osg::Quat q;
    q.set(trans);
    osg::Quat::value_type orient[4];
    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    d_cameraTranslation.set(trans(3, 0), trans(3, 1), trans(3, 2));
    d_cameraRotation.set(orient[0], orient[1], orient[2], orient[3]);
    eventOut(timeStamp, "cameraTranslation", d_cameraTranslation);
    //eventOut(timeStamp, "translation", d_translation);
    eventOut(timeStamp, "cameraRotation", d_cameraRotation);
}

void VrmlNodeVehicle::setVRMLVehicleFrontWheels(const osg::Matrix &transFL, const osg::Matrix &transFR)
{
    osg::Quat q;
    q.set(transFL);
    osg::Quat::value_type orientBody[4];
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_wheelFLRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_wheelFLTranslation.set(transFL(3, 0), transFL(3, 1), transFL(3, 2));

    q.set(transFR);
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_wheelFRRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_wheelFRTranslation.set(transFR(3, 0), transFR(3, 1), transFR(3, 2));

    eventOut(timeStamp, "wheelFLTranslation", d_wheelFLTranslation);
    eventOut(timeStamp, "wheelFLRotation", d_wheelFLRotation);
    eventOut(timeStamp, "wheelFRTranslation", d_wheelFRTranslation);
    eventOut(timeStamp, "wheelFRRotation", d_wheelFRRotation);
}

void VrmlNodeVehicle::setVRMLVehicleFFZFrontWheels(const osg::Matrix &transFL, const osg::Matrix &transFR)
{
    osg::Quat q;
    q.set(transFL);
    osg::Quat::value_type orientBody[4];
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_ffz1wheelFLRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_ffz1wheelFLTranslation.set(transFL(3, 0), transFL(3, 1), transFL(3, 2));

    q.set(transFR);
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_ffz1wheelFRRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_ffz1wheelFRTranslation.set(transFR(3, 0), transFR(3, 1), transFR(3, 2));

    eventOut(timeStamp, "ffz1wheelFLTranslation", d_ffz1wheelFLTranslation);
    eventOut(timeStamp, "ffz1wheelFLRotation", d_ffz1wheelFLRotation);
    eventOut(timeStamp, "ffz1wheelFRTranslation", d_ffz1wheelFRTranslation);
    eventOut(timeStamp, "ffz1wheelFRRotation", d_ffz1wheelFRRotation);
}

void VrmlNodeVehicle::setVRMLVehicleRearWheels(const osg::Matrix &transRL, const osg::Matrix &transRR)
{
    osg::Quat q;
    q.set(transRL);
    osg::Quat::value_type orientBody[4];
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_wheelRLRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_wheelRLTranslation.set(transRL(3, 0), transRL(3, 1), transRL(3, 2));

    q.set(transRR);
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_wheelRRRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_wheelRRTranslation.set(transRR(3, 0), transRR(3, 1), transRR(3, 2));

    eventOut(timeStamp, "wheelRLTranslation", d_wheelRLTranslation);
    eventOut(timeStamp, "wheelRLRotation", d_wheelRLRotation);
    eventOut(timeStamp, "wheelRRTranslation", d_wheelRRTranslation);
    eventOut(timeStamp, "wheelRRRotation", d_wheelRRRotation);
}

void VrmlNodeVehicle::setVRMLVehicleFFZRearWheels(const osg::Matrix &transRL, const osg::Matrix &transRR)
{
    osg::Quat q;
    q.set(transRL);
    osg::Quat::value_type orientBody[4];
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_ffz1wheelRLRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_ffz1wheelRLTranslation.set(transRL(3, 0), transRL(3, 1), transRL(3, 2));

    q.set(transRR);
    q.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
    d_ffz1wheelRRRotation.set(orientBody[0], orientBody[1], orientBody[2], orientBody[3]);

    d_ffz1wheelRRTranslation.set(transRR(3, 0), transRR(3, 1), transRR(3, 2));

    eventOut(timeStamp, "ffz1wheelRLTranslation", d_ffz1wheelRLTranslation);
    eventOut(timeStamp, "ffz1wheelRLRotation", d_ffz1wheelRLRotation);
    eventOut(timeStamp, "ffz1wheelRRTranslation", d_ffz1wheelRRTranslation);
    eventOut(timeStamp, "ffz1wheelRRRotation", d_ffz1wheelRRRotation);
}

void VrmlNodeVehicle::setVRMLVehicle(const osg::Matrix &trans)
{
    osg::Quat q;
    q.set(trans);
    osg::Quat::value_type orient[4];
    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    d_carTranslation.set(trans(3, 0), trans(3, 1), trans(3, 2));
    d_carRotation.set(orient[0], orient[1], orient[2], orient[3]);
    //fprintf(stderr,"orientVehicle: %f %f %f %f\n",orient[0], orient[1], orient[2], orient[3]);

    eventOut(timeStamp, "carTranslation", d_carTranslation);
    eventOut(timeStamp, "carRotation", d_carRotation);
}

void VrmlNodeVehicle::setVRMLVehicleAxles(const osg::Matrix &axle1Trans, const osg::Matrix &axle2Trans)
{
    osg::Quat q;
    osg::Quat::value_type orient[4];
    q.set(axle1Trans);
    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    d_axle1Translation.set(axle1Trans(3, 0), axle1Trans(3, 1), axle1Trans(3, 2));
    d_axle1Rotation.set(orient[0], orient[1], orient[2], orient[3]);
    eventOut(timeStamp, "axle1Translation", d_axle1Translation);
    eventOut(timeStamp, "axle1Rotation", d_axle1Rotation);
    q.set(axle2Trans);
    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    d_axle2Translation.set(axle2Trans(3, 0), axle2Trans(3, 1), axle2Trans(3, 2));
    d_axle2Rotation.set(orient[0], orient[1], orient[2], orient[3]);
    eventOut(timeStamp, "axle2Translation", d_axle2Translation);
    eventOut(timeStamp, "axle2Rotation", d_axle2Rotation);
}
void VrmlNodeVehicle::setVRMLVehicleWheels(const osg::Matrix &wheel1Trans, const osg::Matrix &wheel2Trans, const osg::Matrix &wheel3Trans, const osg::Matrix &wheel4Trans)
{
    osg::Quat q;
    osg::Quat::value_type orient[4];

    std::array<const osg::Matrix*, NUM_WHEELS> wheelTranslations = {&wheel1Trans, &wheel2Trans, &wheel3Trans, &wheel4Trans};
    for (size_t i = 0; i < NUM_WHEELS; i++)
    {
        q.set((*wheelTranslations[i]));
        q.getRotate(orient[3], orient[0], orient[1], orient[2]);
        d_wheelTranslations[i].set((*wheelTranslations[i])(3, 0), (*wheelTranslations[i])(3, 1), (*wheelTranslations[i])(3, 2));
        d_wheelRotations[i].set(orient[0], orient[1], orient[2], orient[3]);
        eventOut(timeStamp, ("wheel" + std::to_string(i) + "Translation").c_str(), d_wheelTranslations[i]);
        eventOut(timeStamp, ("wheel" + std::to_string(i) + "Rotation").c_str(), d_wheelRotations[i]);
    }
}

osg::Vec2d VrmlNodeVehicle::getPos()
{
    return actual_pos;
}

void VrmlNodeVehicle::savePos(double x, double y)
{
    //pos[0]=x;
    //pos[1]=y;
    actual_pos[0] = x;
    actual_pos[1] = y;
}

double VrmlNodeVehicle::getV()
{
    return velocity;
    //return PorscheRealtimeDynamics::dSpace_v;
}
