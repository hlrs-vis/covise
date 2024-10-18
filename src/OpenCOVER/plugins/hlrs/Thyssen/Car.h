/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _Car_NODE_PLUGIN_H
#define _Car_NODE_PLUGIN_H

#include <util/common.h>
#include <Thyssen.h>

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

class VrmlNodeElevator;
class VrmlNodeExchanger;
class VrmlNodeLanding;

#define CAR_WIDTH_2 1.0 // Half with of the car
#define CAR_HEIGHT_2 1.8 // Half height of the car
#define LANDING_WIDTH_2 1.8 // Half with of the exchanger or landing
#define LANDING_HEIGHT_2 1.8 // Half height of the exchanger
#define SAFETY_DISTANCE 0.1

class PLUGINEXPORT VrmlNodeCar : public VrmlNodeChild, public coTUIListener
{
public:
    enum carState {Idle=0,DoorOpening, DoorOpen, DoorClosing, Moving, RotatingRight, RotatingLeft, Uninitialized, MoveUp, MoveDown, MoveLeft, MoveRight,StartRotatingRight,StartRotatingLeft};
    // Define the fields of Car nodes
    static void initFields(VrmlNodeCar *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeCar(VrmlScene *scene = 0);
    VrmlNodeCar(const VrmlNodeCar &n);

    virtual VrmlNodeCar *toCar() const;

    const VrmlField *getField(const char *fieldName) const override;

    void eventIn(double timeStamp, const char *eventName,
        const VrmlField *fieldValue);

    virtual void render(Viewer *);
    void update();
    int getID(){return ID;};
    void setElevator(VrmlNodeElevator *);
    enum carState getState();
    void setState(enum carState s);
    enum carState getChassisState();
    void setChassisState(enum carState s);
    enum carState getTravelDirection();
    void setTravelDirection(enum carState t);
    int getLandingNumber(){return landingNumber;};
    int getShaftNumber(){return shaftNumber;};
    void setDestination(int landing, int shaft);
    void moveToNext(); // move to next station
    void arrivedAtDestination(); // the car arrived at its destination
    bool nextPositionIsEmpty(); // return true if the next station towards the destination landing is empty
    float getV(){return v;};
    void setAngle(float a);
    bool stationListChanged();
    void switchToNewStationList();// try to switch to new stationList
	void lock();
	void unlock();

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

    VrmlSFInt   d_carNumber;
    VrmlSFVec3f d_carPos;
    VrmlSFTime  d_carDoorClose;
    VrmlSFTime  d_carDoorOpen;
    VrmlSFFloat d_doorTimeout;
    VrmlSFRotation d_carRotation;
    VrmlSFFloat d_carFraction;
    VrmlMFInt d_stationList;
    VrmlMFFloat d_stationOpenTime;
    VrmlSFInt d_currentStationIndex;
	VrmlSFTime d_lockTime;



private:
    static int IDCounter;
    float v;
    float a;
	float aMax;
	float vMax;
	float ahMax;
	float vhMax;
    float destinationY;
    float destinationX;
    float startingY;
    float startingX;
    double angle;
    float av;
    float aa;
    float avMax;
    float aaMax;
    int doorState;
    int landingNumber;
    int shaftNumber;
    int oldLandingNumber;
    int oldShaftNumber;
    int oldLandingIndex; // is >=0 until we left the station
    int destinationLandingIndex; // is >=0 until we are close to the destination
    std::list<int>::iterator currentPassingStation;
    double doorTime;
    VrmlNodeElevator *elevator;
    enum carState state;
    enum carState oldState;
    enum carState chassisState;
    enum carState oldChassisState;
    enum carState travelDirection;
    enum carState oldTravelDirection;
    double timeoutStart;
    int ID;
    std::list<VrmlNodeExchanger *> currentExchangers; // list of exchangers we pass
    std::list<int> passingStations; // stations that we pass including the current destination, excluding the start
    std::list<int> occupiedStations; // stations that we occupied and which have not peen released
    coTUIToggleButton *openButton;
    coTUILabel *carLabel;
    coTUIEditField *stationListEdit;
    std::list<int> temporaryStationList;
};

#endif
