/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Vehicle_NODE_PLUGIN_H
#define _Vehicle_NODE_PLUGIN_H

#include <util/common.h>

#define MAX_BODIES 100
#ifdef WIN32
//#define STRICT
#define DIRECTINPUT_VERSION 0x0800
#include <winsock2.h>
#include <windows.h>
#include <commctrl.h>
#include <basetsd.h>
#include <dinput.h>
#endif

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <array>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>

#include <util/coTypes.h>

#include <vrml97/vrml/Player.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

#include "InputDevice.h"
#define USE_CAR_SOUND
#ifdef USE_CAR_SOUND
#include "CarSound.h"
#else
#include "EngineSound.h"
#endif

using namespace opencover;
using namespace vrml;

class PLUGINEXPORT VrmlNodeVehicle : public VrmlNodeChild
{
public:
#ifdef USE_CAR_SOUND
    CarSound *carSound;
#else
    static Player *player;
    Player::Source *source;
    Player::Source *gearSound;
    Player::Source *hornSound;
    EngineSound *engineSound;
#endif
    // Define the fields of SteeringWheel nodes
    static void initFields(VrmlNodeVehicle *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeVehicle(VrmlScene *scene = 0);
    VrmlNodeVehicle(const VrmlNodeVehicle &n);
    virtual ~VrmlNodeVehicle();


    virtual VrmlNodeVehicle *toVehicleWheel() const;

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    float getA(float v, float vmin, float vmax);

    osg::Matrix *getCarTransMatrix();

    void moveToStreet();
    void moveToStreet(osg::Matrix &carTrans);

    void getGroundDistance(osg::Matrix &wheelFLMatrix, osg::Matrix &wheelFRMatrix, osg::Matrix &wheelRLMatrix, osg::Matrix &wheelRRMatrix, double &distFL, double &distFR, double &distRL, double &distRR);

    void getGroundIntersectPoints(osg::Matrix &wheelFLMatrix, osg::Matrix &wheelFRMatrix, osg::Matrix &wheelRLMatrix, osg::Matrix &wheelRRMatrix, osg::Vec3 &intersectFL, osg::Vec3 &intersectFR, osg::Vec3 &intersectRL, osg::Vec3 &intersectRR);

    virtual void render(Viewer *);

    void setVRMLVehicle(const osg::Matrix &trans);
    void setVRMLVehicleBody(const osg::Matrix &trans);
    void setVRMLVehicleBody(int body, const osg::Matrix &trans);
    void setVRMLVehicleCamera(const osg::Matrix &trans);
    void setVRMLVehicleFrontWheels(const osg::Matrix &transFL, const osg::Matrix &transFR);
    void setVRMLVehicleRearWheels(const osg::Matrix &transRL, const osg::Matrix &transRR);

    //Ghostfahrzeug
    void setVRMLVehicleFFZBody(const osg::Matrix &trans);
    void setVRMLVehicleFFZFrontWheels(const osg::Matrix &transFL, const osg::Matrix &transFR);
    void setVRMLVehicleFFZRearWheels(const osg::Matrix &transRL, const osg::Matrix &transRR);
    //additionalData
    void setVRMLAdditionalData(float, float, float, float, float, float, float, float, float, float, int, int, int, int, int, int, int, int, int, int);
    void setVRMLVehicleAxles(const osg::Matrix &axle1Trans, const osg::Matrix &axle2Trans);
    void setVRMLVehicleWheels(const osg::Matrix &wheel1Trans, const osg::Matrix &wheel2Trans, const osg::Matrix &wheel3Trans, const osg::Matrix &wheel4Trans);

    static osg::Vec2d getPos();
    void savePos(double, double);
    static double getV();

    bool followTerrain()
    {
        return d_followTerrain.get();
    };

private:
    double timeStamp;
    int inputDevice;
    InputDevice *inDevice;
    void init();
    void autodetect();
    void timeStep();

    // Fields
    VrmlSFRotation d_brakePedalRotation;
    VrmlSFRotation d_clutchPedaRotation;
    VrmlSFRotation d_gasPedalRotation;
    VrmlSFRotation d_steeringWheelRotation;
    VrmlSFString d_inputDevice;

    VrmlSFBool d_reset;
    VrmlSFRotation d_wheelRotation;
    VrmlSFRotation d_carRotation;
    VrmlSFVec3f d_carTranslation;
    VrmlSFRotation d_carBodyRotation;
    VrmlSFVec3f d_carBodyTranslation;
    VrmlSFRotation d_axle1Rotation;
    VrmlSFVec3f d_axle1Translation;
    VrmlSFRotation d_axle2Rotation;
    VrmlSFVec3f d_axle2Translation;

    VrmlSFVec3f d_wheelFLTranslation;
    VrmlSFRotation d_wheelFLRotation;
    VrmlSFVec3f d_wheelFRTranslation;
    VrmlSFRotation d_wheelFRRotation;
    VrmlSFVec3f d_wheelRLTranslation;
    VrmlSFRotation d_wheelRLRotation;
    VrmlSFVec3f d_wheelRRTranslation;
    VrmlSFRotation d_wheelRRRotation;

    //Ghost
    VrmlSFVec3f d_ffz1wheelFLTranslation;
    VrmlSFRotation d_ffz1wheelFLRotation;
    VrmlSFVec3f d_ffz1wheelFRTranslation;
    VrmlSFRotation d_ffz1wheelFRRotation;
    VrmlSFVec3f d_ffz1wheelRLTranslation;
    VrmlSFRotation d_ffz1wheelRLRotation;
    VrmlSFVec3f d_ffz1wheelRRTranslation;
    VrmlSFRotation d_ffz1wheelRRRotation;

    //additional Data
    static const size_t NUM_ADDITIONAL_FIELDS = 10;
    std::array<VrmlSFFloat, NUM_ADDITIONAL_FIELDS> d_float_values;
    std::array<VrmlSFInt, NUM_ADDITIONAL_FIELDS> d_int_values;


    static const int NUM_WHEELS = 4;
    std::array<VrmlSFRotation, NUM_WHEELS> d_wheelRotations;
    std::array<VrmlSFVec3f, NUM_WHEELS> d_wheelTranslations;

    VrmlSFRotation d_bodyRotation[MAX_BODIES];
    VrmlSFVec3f d_bodyTranslation[MAX_BODIES];
    char *bodyTransName[MAX_BODIES];
    char *bodyRotName[MAX_BODIES];
    VrmlSFRotation d_cameraRotation;
    VrmlSFVec3f d_cameraTranslation;
    float heading[3];
    float pitch[3];
    VrmlSFRotation d_mirrorLRotation;
    VrmlSFRotation d_mirrorMRotation;
    VrmlSFRotation d_mirrorRRotation;

    std::array<VrmlSFVec3f, 16> d_carTranslations;

    VrmlSFInt d_numCars;
    VrmlSFBool d_followTerrain;
    VrmlSFFloat d_speed;
    VrmlSFFloat d_revs;
    VrmlSFFloat d_acceleration;

    // offset //
    VrmlSFBool d_setOffset;
    VrmlSFVec3f d_offsetLoc;
    VrmlSFRotation d_offsetRot;
    VrmlSFBool d_printTransformation;

    VrmlSFFloat d_aMax;
    VrmlSFFloat d_vMax;
    VrmlSFInt d_gear;

    VrmlSFInt d_mirrorLightLeft;
    VrmlSFInt d_mirrorLightRight;

    VrmlSFVec3f d_ffz1Translation;
    VrmlSFRotation d_ffz1Rotation;
    osg::Matrix carTrans;
    osg::Matrix lastRoadPos;

    void recalcMatrix();
    float tf;
    float mdz;
    float sdz;

    static osg::Vec2d actual_pos;
    static double velocity;
};

#endif
