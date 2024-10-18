/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Bicycle_NODE_PLUGIN_H
#define _Bicycle_NODE_PLUGIN_H

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
#include <config/CoviseConfig.h>

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
#include <vrml97/vrml/VrmlSFRotation.h>

#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

#define NUM_BUTTONS 3
#ifdef WIN32
#else
#include <linux/joystick.h> // das muss nach osg kommen, wegen KEY_F1
#include <unistd.h>

#include <X11/Xlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>

#endif
#include "Tacx.h"
#include "FlightGear.h"
#include "Skateboard.h"

using namespace vrml;
using namespace opencover;

class PLUGINEXPORT VrmlNodeBicycle : public VrmlNodeChild
{
public:
    // Define the fields of Bicycle nodes
    static void initFields(VrmlNodeBicycle *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeBicycle(VrmlScene *scene = 0);
    VrmlNodeBicycle(const VrmlNodeBicycle &n);

    VrmlSFRotation d_bikeRotation;
    VrmlSFVec3f d_bikeTranslation;
    VrmlSFBool d_thermal;

    virtual VrmlNodeBicycle *toBicycle() const;

    const VrmlField *getField(const char *fieldName) const override;

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    virtual void render(Viewer *);
    void recalcMatrix();

    void moveToStreet();

    void moveToStreet(osg::Matrix &carTrans);

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    osg::Matrix bikeTrans;
    osg::Matrix fgPlaneRot;
    osg::Vec3d fgPlaneZeroPos;
    // Fields
    VrmlSFBool d_enabled;
    VrmlSFInt d_joystickNumber;
    VrmlSFInt d_button;

    // State
    VrmlMFFloat d_axes;
    VrmlMFInt d_buttons;
    int oldButtonState;
};

class BicyclePlugin : public coVRPlugin, public coTUIListener, public OpenThreads::Thread
{
public:
    BicyclePlugin();
    ~BicyclePlugin();
    bool init();
    unsigned char speedCounter;
    volatile bool running;
    void stop();
    bool doStop;

    static BicyclePlugin *plugin;
    Tacx *tacx;
    FlightGear* flightgear;
    Skateboard* skateboard;
    bool isPlane;
    bool isBike;
    bool isParaglider;
    bool isSkateboard;
    coTUITab *BicycleTab;
    coTUIEditFloatField *velocityFactor;
    coTUILabel *velocityFactorLabel;
    coTUIEditFloatField *forceFactor;
    coTUILabel *forceFactorLabel;
    coTUIEditFloatField *wingArea;
    coTUILabel *wingAreaLabel;



    virtual void run();

    int numFloats;
    void preFrame();
    void key(int type, int keySym, int mod);
    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);
    int initUI();
    void UpdateInputState();

    int buttonPressEventType;
    int buttonReleaseEventType;

    char readps2(int fd);
    int mouse1; // file descriptor for the opened device
    int mouse2; // file descriptor for the opened device
    char buffer[8]; // [0] = buttonmask, [1] = dx, [2] = dy, [3] = dz (wheel)
    int counters[2];
    int angleCounter;
    float angle;
    float speed;
    float power;
    
private:
};
#endif
