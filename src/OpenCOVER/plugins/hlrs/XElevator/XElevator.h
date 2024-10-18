/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _XElevator_NODE_PLUGIN_H
#define _XElevator_NODE_PLUGIN_H

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
class VrmlNodeXCar;
class VrmlNodeXLanding;

class Shaft
{
protected:
    std::list<VrmlNodeXCar *> CarsInShaft;
public:
};

class StationInfo
{
protected:
    float xPos;
    float yPos;
public:
    VrmlNodeXCar *Car;
    float x(){return xPos;};
    float y(){return yPos;};
    void setX(float x){xPos = x;};
    void setY(float y){yPos = y;};

};

class PLUGINEXPORT VrmlNodeXElevator : public VrmlNodeGroup
{
public:
    // Define the fields of XElevator nodes
    static void initFields(VrmlNodeXElevator *node, VrmlNodeType *t = 0);
    static const char *name();

    VrmlNodeXElevator(VrmlScene *scene = 0);
    VrmlNodeXElevator(const VrmlNodeXElevator &n);

    virtual VrmlNodeXElevator *toXElevator() const;
    
    void eventIn(double timeStamp, const char *eventName,
        const VrmlField *fieldValue);

    virtual void render(Viewer *);
    VrmlMFFloat d_landingHeights;
    VrmlMFFloat  d_shaftPositions;
    std::vector<Shaft *> shafts;
    std::vector<VrmlNodeXLanding *> Landings;
    std::vector<VrmlNodeXCar *> Cars;
    std::vector<StationInfo> stations; // stations[i] is set to a XCar if the XCar is currently close to that station
    
    bool occupy(int station,VrmlNodeXCar *XCar); // returns true, if successfull, false if station is occupied by someone else
    void release(int station);
    void goTo(int landing);
    
    coTUITab *XElevatorTab;

private:
    void childrenChanged() override;
};


class XElevatorPlugin : public coVRPlugin
{
public:
    XElevatorPlugin();
    ~XElevatorPlugin();
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
    static XElevatorPlugin* plugin;

    bool update();

private:
};

#endif
