/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _XCar_NODE_PLUGIN_H
#define _XCar_NODE_PLUGIN_H

#include <util/common.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>

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
#include <vrml97/vrml/VrmlSFRotation.h>
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

class VrmlNodeXElevator;
class VrmlNodeXLanding;

#define Car_WIDTH_2 1.0 // Half with of the XCar
#define Car_HEIGHT_2 1.8 // Half height of the XCar
#define Landing_WIDTH_2 1.8 // Half with of the exchanger or XLanding
#define Landing_HEIGHT_2 1.8 // Half height of the exchanger
#define SAFETY_DISTANCE 0.1

class PLUGINEXPORT VrmlNodeXCar : public VrmlNodeChild,public coTUIListener
{
public:
    enum CarState {Idle=0,DoorOpening, DoorOpen, DoorClosing, Moving, RotatingRight, RotatingLeft, Uninitialized, MoveUp, MoveDown, MoveLeft, MoveRight,StartRotatingRight,StartRotatingLeft};
    // Define the fields of XCar nodes
    static void initFields(VrmlNodeXCar *node, VrmlNodeType *type);
    static const char *name();

    VrmlNodeXCar(VrmlScene *scene = 0);
    VrmlNodeXCar(const VrmlNodeXCar &n);

    virtual VrmlNodeXCar *toXCar() const;

    virtual void render(Viewer *);
    void update();
    int getID(){return ID;};
    void setElevator(VrmlNodeXElevator *);
    enum CarState getState();
    void setState(enum CarState s);
    enum CarState getChassisState();
    void setChassisState(enum CarState s);
    enum CarState getTravelDirection();
    void setTravelDirection(enum CarState t);
    int getLandingNumber(){return LandingNumber;};
    void setDestination(int XLanding);
    void moveToNext(); // move to next station
    void arrivedAtDestination(); // the XCar arrived at its destination
    float getV(){return v;};
    void setAngle(float a);
	void lock();
	void unlock();
    void goTo(int landing);

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

    VrmlSFInt   d_carNumber;
    VrmlSFVec3f d_carPos;
    VrmlSFVec3f d_carTransformPos;
    VrmlSFVec3f d_carOffset;
    VrmlSFTime  d_carDoorClose;
    VrmlSFTime  d_carDoorOpen;
    VrmlSFFloat d_doorTimeout;
    VrmlSFRotation d_carRotation;
    VrmlSFFloat d_carFraction;
    VrmlMFInt d_stationList;
    VrmlSFInt d_currentLanding;
    VrmlMFFloat d_stationOpenTime;
	VrmlSFTime d_lockTime;
    std::set<int>stationList;



private:
    static int IDCounter;
    float v;
    float a;
	float aMax;
	float vMax;
	float ahMax;
	float vhMax;
    float destinationY;
    float startingY;
    double angle;
    float av;
    float aa;
    float avMax;
    float aaMax;
    int doorState;
    int LandingNumber;
    int oldLandingNumber;
    int oldLandingIndex; // is >=0 until we left the station
    int destinationLandingIndex; // is >=0 until we are close to the destination
    double doorTime;
    VrmlNodeXElevator *Elevator;
    enum CarState state;
    enum CarState oldState;
    enum CarState chassisState;
    enum CarState oldChassisState;
    enum CarState travelDirection;
    enum CarState oldTravelDirection;
    double timeoutStart;
    int ID;
    coTUIToggleButton *openButton;
    coTUILabel *CarLabel;
    coTUIEditField *stationListEdit;
    std::list<int> temporaryStationList;
};

#endif
