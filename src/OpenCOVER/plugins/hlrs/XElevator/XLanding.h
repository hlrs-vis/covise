/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _XLanding_NODE_PLUGIN_H
#define _XLanding_NODE_PLUGIN_H

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
class VrmlNodeXCar;

class PLUGINEXPORT VrmlNodeXLanding : public VrmlNodeChild
{
public:
    enum XLandingState {Idle=0,Occupied, Uninitialized};

    static void initFields(VrmlNodeXLanding *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeXLanding(VrmlScene *scene = 0);
    VrmlNodeXLanding(const VrmlNodeXLanding &n);

    virtual VrmlNodeXLanding *toXLanding() const;

    void eventIn(double timeStamp, const char *eventName,
        const VrmlField *fieldValue);

    virtual void render(Viewer *);
    void update();
    void setElevator(VrmlNodeXElevator *);
    enum XLandingState getState(){return state;}
    int getCarNumber();
    void setCar(VrmlNodeXCar *c);
    void openDoor();
    void closeDoor();
    VrmlNodeXCar *getCar(){return currentCar;};
    VrmlSFInt   d_LandingNumber;

    VrmlSFTime  d_callButton;

    VrmlSFTime  d_doorClose;
    VrmlSFTime  d_doorOpen;

private:

    
    
    VrmlNodeXCar *currentCar;
    VrmlNodeXElevator *Elevator;
    
    enum XLandingState state;
    double timeoutStart;
};

#endif
