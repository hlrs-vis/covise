/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SteeringWheel_NODE_PLUGIN_H
#define _SteeringWheel_NODE_PLUGIN_H

#include <util/common.h>
#include <net/covise_connect.h>

#include <cover/coTabletUI.h>

#ifdef WIN32
//#define STRICT
#include <InitGuid.h>
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
#include "Porsche.h"
#include "FKFS.h"
#include "PorscheController.h"
#include "VehicleDynamics.h"
#include "CAN.h"

#include <vrml97/vrml/Player.h>
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

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
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

#ifdef WIN32
#include "TempWindow.h"
#else
#include <linux/joystick.h> // das muss nach osg kommen, wegen KEY_F1
#include <unistd.h>

#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>

#endif

#define NUM_BUTTONS 3

enum
{
    JOYSTICK_BUTTON_EVENTS = 1,
    JOYSTICK_AXES_EVENTS = 2
};

class PLUGINEXPORT VrmlNodeSteeringWheel : public VrmlNodeChild
{
public:
    // Define the fields of SteeringWheel nodes
    static void initFields(VrmlNodeSteeringWheel *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeSteeringWheel(VrmlScene *scene = 0);
    VrmlNodeSteeringWheel(const VrmlNodeSteeringWheel &n);

    virtual VrmlNodeSteeringWheel *toSteeringWheel() const;

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
    VrmlSFInt d_joystickNumber;

    // State
    VrmlMFFloat d_axes;
    VrmlMFInt d_buttons;
};

class SteeringWheelPlugin : public coVRPlugin, public coTUIListener
{
public:
    SteeringWheelPlugin();
    ~SteeringWheelPlugin();
    bool init();

    static SteeringWheelPlugin *plugin;
    int fd[MAX_NUMBER_JOYSTICKS];
    int version;
    Porsche *porsche;
    FFWheel *sitzkiste;
    VehicleDynamics *dynamics;
    PorscheController *dataController;

    ServerConnection *serverConn;
    std::unique_ptr<SimpleServerConnection> toClientConn;

    coTUITab *SteeringWheelTab;
    //coTUIToggleButton *showSky;
    coTUIEditFloatField *blockAngle;
    coTUIEditFloatField *springConstant;
    coTUIEditFloatField *dampingConstant;
    coTUIEditFloatField *rumbleFactor;
    coTUIEditFloatField *drillingFrictionConstant;
    coTUIEditFloatField *velocityImpactFactor;
    coTUIEditFloatField *velocityImpactFactorRumble;
    coTUILabel *blockAngleLabel;
    coTUILabel *springConstantLabel;
    coTUILabel *dampingConstantLabel;
    coTUILabel *rumbleFactorLabel;
    coTUILabel *drillingFrictionConstantLabel;
    coTUILabel *velocityImpactFactorLabel;
    coTUILabel *velocityImpactFactorRumbleLabel;

    coTUIButton *softResetWheelButton;
    coTUIButton *cruelResetWheelButton;
    coTUIButton *shutdownWheelButton;

    coTUIButton *platformToGroundButton;
    coTUIButton *platformReturnToActionButton;

    SimpleClientConnection *conn;
    Host *serverHost;
    Host *localHost;
    int serverPort;
    int port;
    int numFloats;
    int numInts;
    int numLocalJoysticks;
    int simulatorJoystick;
    int oldSimulatorJoystick;
    float *floatValues;
    int *intValues;
    double oldTime;

    float *floatValuesOut;
    int *intValuesOut;
    int numFloatsOut;
    int numIntsOut;
    float updateRate;
#ifdef WIN32
    static BOOL CALLBACK EnumObjectsCallback(const DIDEVICEOBJECTINSTANCE *pdidoi, VOID *pContext);
    static BOOL CALLBACK EnumJoysticksCallback(const DIDEVICEINSTANCE *pdidInstance, VOID *pContext);
    LPDIRECTINPUT8 g_pDI;
    LPDIRECTINPUTDEVICE8 g_pJoystick[MAX_NUMBER_JOYSTICKS];
    TemporaryWindow window;
#endif

    bool sendValues();
    bool readVal(void *buf, unsigned int numBytes);
    bool sendValuesToClient();
    bool readClientVal(void *buf, unsigned int numBytes);
    // this will be called in PreFrame
    void preFrame();
    void key(int type, int keySym, int mod);
    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);
    int initUI();
    void UpdateInputState();

    // this function is called if a message arrives
    virtual void message(int toWhom, int type, int length, const void *data) override;

    bool initJoystick(int joystickNumber);
    int getData(int number_joystick);
    void printData(int number_joystick);

    int buttonPressEventType;
    int buttonReleaseEventType;
#if !defined(_WIN32)
    Display *display;
    XEventClass eventClasses[50];
    int eventTypes[50];
    int buttonPressMask;
#endif
    int wheelcounter;

    char readps2(int fd);
    int device; // file descriptor for the opened device
    int horndevice;
    char buffer[4]; // [0] = buttonmask, [1] = dx, [2] = dy, [3] = dz (wheel)
    bool haveMouse;
    bool haveHorn;

private:
};
#endif
