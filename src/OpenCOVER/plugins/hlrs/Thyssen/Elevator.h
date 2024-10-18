/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _Elevator_NODE_PLUGIN_H
#define _Elevator_NODE_PLUGIN_H

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
class VrmlNodeCar;
class VrmlNodeExchanger;
class VrmlNodeLanding;

class Rail
{
protected:
    std::list<VrmlNodeCar *> carsOnRail;
public:
    void putCarOnRail(VrmlNodeCar *);
    void removeCarFromRail(VrmlNodeCar *);
    float getNextCarOnRail(VrmlNodeCar *car, VrmlNodeCar *&closestCar); // returns the distance to the closest car in travel direction on the current rail or -1 if there is none;
};

class StationInfo
{
protected:
    float xPos;
    float yPos;
public:
    VrmlNodeCar *car;
    float x(){return xPos;};
    float y(){return yPos;};
    void setX(float x){xPos = x;};
    void setY(float y){yPos = y;};

};

class PLUGINEXPORT VrmlNodeElevator : public VrmlNodeGroup
{
public:
    static void initFields(VrmlNodeElevator *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeElevator(VrmlScene *scene = 0);
    VrmlNodeElevator(const VrmlNodeElevator &n);

    virtual VrmlNodeElevator *toElevator() const;

    void eventIn(double timeStamp, const char *eventName,
        const VrmlField *fieldValue);

    virtual void render(Viewer *);
    VrmlMFFloat d_landingHeights;
    VrmlMFFloat  d_shaftPositions;
    std::vector<Rail *> shafts;
    std::vector<Rail *> hShafts;
    
    void putCarOnRail(VrmlNodeCar *);
    void removeCarFromRail(VrmlNodeCar *);
    float getNextCarOnRail(VrmlNodeCar *car, VrmlNodeCar *&closestCar); // returns the distance to the closest car in travel direction on the current rail or -1 if there is none;

    std::vector<VrmlNodeLanding *> landings;
    std::vector<VrmlNodeExchanger *> exchangers;
    std::vector<VrmlNodeCar *> cars;
    std::vector<StationInfo> stations; // stations[i] is set to a car if the car is currently close to that station
    
    bool occupy(int station,VrmlNodeCar *car); // returns true, if successfull, false if station is occupied by someone else
    void release(int station);
    
    coTUITab *elevatorTab;

private:
    void childrenChanged() override;
};

#endif
