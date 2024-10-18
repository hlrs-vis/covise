/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Sensor_NODE_PLUGIN_H
#define _Sensor_NODE_PLUGIN_H

#include <util/common.h>
#include <net/covise_connect.h>

#include <cover/coTabletUI.h>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>

#ifdef WIN32
//#define STRICT
#define DIRECTINPUT_VERSION 0x0800
#include <winsock2.h>
#include <windows.h>
#include <commctrl.h>
#include <basetsd.h>
#include <dinput.h>
#define SAFE_DELETE(p)  \
    {                   \
        if (p)          \
        {               \
            delete (p); \
            (p) = NULL; \
        }               \
    }
#define SAFE_RELEASE(p)     \
    {                       \
        if (p)              \
        {                   \
            (p)->Release(); \
            (p) = NULL;     \
        }                   \
    }
#else
#endif

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <vrml97/vrml/Player.h>
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <config/CoviseConfig.h>

#include <util/coTypes.h>
#include <util/coExport.h>

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
#include <vrml97/vrml/VrmlSFRotation.h>

#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlSFTime.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

#define NUM_BUTTONS 3
#ifdef WIN32
#else
#ifdef __linux__
#include <linux/joystick.h> // das muss nach osg kommen, wegen KEY_F1
#endif
#include <unistd.h>

#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>

#endif
using namespace vrml;
using namespace opencover;

class PLUGINEXPORT VrmlNodeSensor : public VrmlNodeChild
{
public:
    // Define the fields of Sensor nodes
    static void initFields(VrmlNodeSensor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeSensor(VrmlScene *scene = 0);
    VrmlNodeSensor(const VrmlNodeSensor &n);

    static const size_t numSensors = 16;
    std::array<VrmlSFTime, numSensors> d_sensor;

    VrmlSFBool d_enabled;
    virtual ~VrmlNodeSensor();


    virtual VrmlNodeSensor *toSensor() const;


    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    FILE *fp;
    time_t oldTime;
};

class SensorPlugin : public coVRPlugin, public coTUIListener, public OpenThreads::Thread
{
public:
    SensorPlugin();
    ~SensorPlugin();
    bool init();
    volatile bool running;
    void stop();
    bool doStop;

    static SensorPlugin *plugin;
    coTUITab *SensorTab;

    virtual void run();

    void preFrame();
    void key(int type, int keySym, int mod);
    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);
    int initUI();
    void UpdateInputState();

private:
};
#endif
