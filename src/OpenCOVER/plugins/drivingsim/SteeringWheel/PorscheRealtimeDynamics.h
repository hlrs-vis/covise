/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __PorscheRealtimeDynamics_H
#define __PorscheRealtimeDynamics_H

#include <util/common.h>

#include <util/UDPComm.h>

#include <fstream>
#include <array>

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>
#include "VehicleDynamics.h"
#include <net/covise_host.h>
#include <net/covise_connect.h>

#include <osg/MatrixTransform>
#include <osg/LineSegment>

#include <vrml97/vrml/VrmlSFTime.h>

//#include "../TrafficSimulation/PorscheFFZ.h" // TEST 19.04.2011
#include <VehicleUtil/RoadSystem/RoadSystem.h>

#ifndef WIN32
#include <termios.h>
#include <sys/stat.h> /* open */
#include <fcntl.h> /* open */
#include <termios.h> /* tcsetattr */
#include <termio.h> /* tcsetattr */
#include <limits.h> /* sginap */
#endif

#define INDATA 32

using namespace covise;

extern float rayHeight;

enum
{
    VisToDSpace = 1,
    VisToGateway = 2,
    GatewayToVi = 3,
    DSpaceToVis = 513
};

typedef struct
{
    // not active yet
    int MsgType;
    float Hoehe;
    int Reset;
    int Fahrbahnbelag;
    float Kollisionsvektor[3];
} MSG_VisToDSpace;

typedef struct
{
    int MsgType;
    float Simulationszeit;
    float PosX;
    float PosY;
    float PosZ;
    float RollAngle;
    float PitchAngle;
    float YawAngle;
    float RadPosX[4];
    float RadPosY[4];
    float CoGDist;
    float Geschwindigkeit;
    float RadPosZ[4];
    float WheelAngle[4];
    float WheelCamber[4];
    float WheelRotation[4];
    float Motordrehzahl;
    float Lenkwinkel;
    int Gang;
    int shiftmode;
    float ffz1_x;
    float ffz1_y;
    float ffz1_z;
    float ffz1_roll;
    float ffz1_pitch;
    float ffz1_yaw;

    float ffz1_RadPosZ[4];
    float ffz1_WheelRotation;
    float ffz1_WheelAngle;

    //additional data
    float float_value0;
    float float_value1;
    float float_value2;
    float float_value3;
    float float_value4;
    float float_value5;
    float float_value6;
    float float_value7;
    float float_value8;
    float float_value9;

    int int_value0;
    int int_value1;
    int int_value2;
    int int_value3;
    int int_value4;
    int int_value5;
    int int_value6;
    int int_value7;
    int int_value8;
    int int_value9;

    //float coordA[50];
} MSG_DSpaceToVis;

typedef struct
{
    // not active yet
    int MsgType;
    int anzahlFahrzeuge;
    float pos[16][3];
} MSG_VisToGateway;

typedef struct
{
    // not active yet
    int MsgType;
    int SWAInfo;
} MSG_GatewayToVis;

/** VRML node for the Porsche Virtueller Fahrerplatz.
 *
*/
#if 1
class PLUGINEXPORT VrmlNodePorscheVFP : public VrmlNodeChild /* , public coTUIListener*/
{
public:
    // Define the fields of TUI nodes
    static void initFields(VrmlNodePorscheVFP *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodePorscheVFP(VrmlScene *);
    VrmlNodePorscheVFP(const VrmlNodePorscheVFP &);

    virtual void eventIn(double timeStamp, const char *eventName, const VrmlField *fieldValue);
    virtual void render(Viewer *);

    void connectToServer(std::string destinationIP, int port /*, int localPort*/);
    void sendTCPData();
    bool readTCPData(void *intValues, unsigned int numBytes);
    bool getTCPData();
    void syncTCPData();

protected:
    // server for TCP connection
    SimpleClientConnection *clientConn_;
    Host *serverHost_;

    // VRML fields:
    VrmlSFString d_targetIP;
    VrmlSFInt d_targetPort;
    // 	VrmlSFInt d_localPort;

    struct VFPdataPackage
    {
        VFPdataPackage()
            : messageID(0)
        {
            for (int i = 0; i < 16; i++)
            {
                floatValues[i] = 0;
                intValues[i] = 0;
            }
        }
        int messageID;
        float floatValues[16];
        int intValues[16];
    };

    VFPdataPackage fromKMS;
    VFPdataPackage fromMaster;

    VrmlSFInt d_messageID;

    static const int NUM_FIELDS = 16;

    std::array<VrmlSFFloat, NUM_FIELDS> d_floatValues;
    std::array<VrmlSFInt, NUM_FIELDS> d_intValues;

    VrmlSFInt d_messageIDIn;

    std::array<VrmlSFFloat, NUM_FIELDS> d_floatValueIn;
    std::array<VrmlSFInt, NUM_FIELDS> d_intValueIn;


    VrmlSFTime d_lastReceiveTime;
};
#endif

class PLUGINEXPORT PorscheRealtimeDynamics : public VehicleDynamics, public OpenThreads::Thread
{
public:
    PorscheRealtimeDynamics();
    virtual ~PorscheRealtimeDynamics();

    virtual double getVelocity()
    {
        return DSpaceData.Geschwindigkeit;
    }
    //	virtual double					getAcceleration() { return 0.0; } // inherited
    virtual double getEngineSpeed()
    {
        return (int)(DSpaceData.Motordrehzahl*30.0 / (M_PI));
    }
    //	virtual double					getSteeringWheelTorque() { return 0.0; } // inherited

    virtual const osg::Matrix &getVehicleTransformation()
    {
        return chassisTransform;
    }
    virtual void setVehicleTransformation(const osg::Matrix &);
    virtual void setVehicleTransformationOffset(const osg::Matrix &);

    virtual void move(VrmlNodeVehicle *vehicle);
    virtual void resetState();
    virtual void update();

    void moveToStreet(osg::Matrix &carTrans, osg::Matrix &moveMat);
    osg::Vec3 getOldnormal();
    void initOldnormal();
    void setOldnormal(osg::Vec3);
    float oldarr[6];
    virtual void run(); // receiving and sending thread, also does the low level simulation like hard limits
    bool doRun;
    OpenThreads::Barrier endBarrier;
    static double dSpace_v;
    // float									rayHeight[4];

private:
    osg::Matrix chassisTransform;
    osg::Matrix ffz1chassisTransform;

    osg::Matrix targetReference;
    osg::Matrix dSpaceReference;
    osg::Matrix dSpacePose;
    osg::Matrix dSpaceOrigin;

    float oldHeight;
    double inputData[INDATA];
    double visData[INDATA];
    double outputData[1];

    double oldTime;

    Host *serverHost;
    int serverPort;
    int localPort;
    UDPComm *toDSPACE;

    bool readData(); // returns true on success, false if no data has been received.
    bool sendData();
    bool readFile(double *data, unsigned int number);
    std::ifstream inStream;
    std::string filestring;

    MSG_DSpaceToVis threadDSpaceData;
    MSG_DSpaceToVis DSpaceData;
    osg::Vec3 oldnormal;
};

#endif
