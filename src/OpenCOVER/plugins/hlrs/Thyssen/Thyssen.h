/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Thyssen_NODE_PLUGIN_H
#define _Thyssen_NODE_PLUGIN_H

#include <util/common.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <net/covise_connect.h>

#include <util/coTypes.h>

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
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFTime.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlSFVec3f.h>

#include <net/tokenbuffer.h>

using namespace vrui;
using namespace vrml;
using namespace opencover;
using covise::ServerConnection;
using covise::SimpleServerConnection;
using covise::TokenBuffer;

class PLUGINEXPORT VrmlNodeThyssen : public VrmlNodeChild
{
public:
    // Define the fields of Thyssen nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeThyssen(VrmlScene *scene = 0);
    VrmlNodeThyssen(const VrmlNodeThyssen &n);
    virtual ~VrmlNodeThyssen();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeThyssen *toThyssen() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName);

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    // Fields
    VrmlSFBool d_enabled;

    // eventOuts
    VrmlMFFloat d_floats;
    VrmlMFInt d_ints;
    VrmlSFVec3f d_carPos[4];
    VrmlSFTime  d_carDoorClose[4];
    VrmlSFTime  d_carDoorOpen[4];
    VrmlSFFloat d_carAngle[4];
    VrmlSFFloat d_exchangerAngle[4];
    VrmlSFTime  d_landingDoorClose[4];
    VrmlSFTime  d_landingDoorOpen[4];
};

class brakeData
{
public:
    brakeData(int id);
    ~brakeData();
    int brakeID;
    void setData(TokenBuffer &tb);
    int type;
    int status;
};

class carData
{
public:
    carData(int id);
    ~carData();
    int carID;
    void setData(TokenBuffer &tb);
    float posY;
    float posZ;
    float speed;
    float accel;
    int doorState;
    int direction;
    int floor;
    int hzllockState;
    int vtllockState;
    int hzlproxState;
    int vtlproxState;
    int oldDoorState;

    std::vector<brakeData> brakes;
};

class exchangerData
{
public:
    exchangerData(int id);
    ~exchangerData();
    int exID;
    void setData(TokenBuffer &tb);
    float posY;
    float posZ;
    int swvlhzllckStatus;
    int swvlvtllckStatus;
    int swvlproxStatus;
    int cbnlckStatus;
    int cbnlckproxStatus;
    float swvlRotaryMotor; // angle in degrees
    int linkedCar; // carID of car in Exchanger
    float destnPos;

    float oldAngle; // old swvlRotaryMotor

    std::vector<brakeData> brakes;
};
class ThyssenPlugin : public coVRPlugin
{
public:
    ThyssenPlugin();
    ~ThyssenPlugin();
    enum MessageTypes
    {
        CAR_DATA = 0
    };
    enum doorStates
    {
        closed_locked = 1,
        closed_unlocked,
        opening,
        open,
        closing
    };
    enum direction
    {
        direction_down = 0,
        direction_up
    };
    enum lockState
    {
        unlocked = 0,
        locked
    };
    enum proxState
    {
        stateOff = 0,
        stateOn
    };
    bool init();
    int readData(char *buf,unsigned int size);
    ServerConnection *sConn;
    std::unique_ptr<SimpleServerConnection> conn;
    int port;
    double oldTime;
    int numFloats;
    int numInts;
    float *floatValues;
    int *intValues;
    static ThyssenPlugin *plugin;
    float zpos;
    float ypos;
    std::vector<carData> cars;
    std::vector<exchangerData> exchangers;

    // this will be called in PreFrame
    void preFrame();
    bool readVal(void *buf, unsigned int numBytes);

private:
};
#endif
