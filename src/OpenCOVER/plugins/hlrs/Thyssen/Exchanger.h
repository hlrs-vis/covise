/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _Exchanger_NODE_PLUGIN_H
#define _Exchanger_NODE_PLUGIN_H

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
class VrmlNodeCar;

class PLUGINEXPORT VrmlNodeExchanger : public VrmlNodeChild
{
public:
    enum ExchangerState {Idle=0,Occupied, Uninitialized};
    // Define the fields of Exchanger nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeExchanger(VrmlScene *scene = 0);
    VrmlNodeExchanger(const VrmlNodeExchanger &n);
    virtual ~VrmlNodeExchanger();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeExchanger *toExchanger() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName);

    void eventIn(double timeStamp, const char *eventName,
        const VrmlField *fieldValue);

    virtual void render(Viewer *);
    void update();
    void setElevator(VrmlNodeElevator *);
    enum ExchangerState getState(){return state;}
    int getCarNumber();
    void setCar(VrmlNodeCar *c);
    VrmlNodeCar *getCar(){return currentCar;};
    VrmlSFInt   d_LandingNumber;
    VrmlSFFloat   d_Fraction;
    VrmlSFRotation d_Rotation;


private:

    
    float angle;
    float av;
    float aa;
    float avMax;
    float aaMax;
    
    VrmlNodeCar *currentCar;
    VrmlNodeElevator *elevator;
    enum ExchangerState state;
    double timeoutStart;
};

#endif
