/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Simulink_NODE_PLUGIN_H
#define _Simulink_NODE_PLUGIN_H

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
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>
#include <util/UDPComm.h>

using namespace vrml;
using namespace opencover;
using namespace covise;

class PLUGINEXPORT VrmlNodeSimulink : public VrmlNodeChild
{
public:
    // Define the fields of Simulink nodes
    static void initFields(VrmlNodeSimulink *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeSimulink(VrmlScene *scene = 0);
    VrmlNodeSimulink(const VrmlNodeSimulink &n);

    virtual VrmlNodeSimulink *toSimulink() const;

    const VrmlField *getField(const char *fieldName) const override;

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
    VrmlMFFloat d_floatsIn;
    VrmlMFInt d_intsIn;
};

class SimulinkPlugin : public coVRPlugin
{
public:
    SimulinkPlugin();
    ~SimulinkPlugin();
    bool init();
    SimpleClientConnection *conn;
    Host *serverHost;
    Host *localHost;
    int port;
    double oldTime;
    int numFloats;
    int numInts;
    float *floatValues;
    int *intValues;
    static SimulinkPlugin *plugin;

    bool update();
    bool readVal(void *buf, unsigned int numBytes);
    void sendData(int numFloats, float* floats, int numInts, int* ints);

private:
    unsigned short serverPort = 0;
    unsigned short localPort = 0;
    UDPComm* udp;
};
#endif
