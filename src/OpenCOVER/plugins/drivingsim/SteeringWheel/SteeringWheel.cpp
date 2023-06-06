/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "SteeringWheel.h"
#include "Vehicle.h"
#include "RemoteVehicle.h"
#include "FKFSDynamics.h"
#include "TestDynamics.h"
#include "EinspurDynamik.h"
#ifdef __XENO__
#ifdef HAVE_CARDYNAMICSCGA
#include "CarDynamicsCGARealtime.h"
#endif

#include "FourWheelDynamicsRealtime.h"
#include "EinspurDynamikRealtime.h"
#endif
#include "ITMDynamics.h"
#include "PorscheRealtimeDynamics.h"
#include "HLRSRealtimeDynamics.h"

// #include "ITM.h"
#include "Keyboard.h"
#include <VehicleUtil/fasiUpdateManager.h>

#ifdef HAVE_CARDYNAMICSCGA
#include "CarDynamicsCGA.h"
#include "CarDynamicsRtus.h"
#endif

#include <cover/coVRTui.h>
#include <cover/coVRCommunication.h>

#include <net/covise_host.h>
#include <net/covise_socket.h>

#include <util/unixcompat.h>

using namespace vehicleUtil;

#if !defined(_WIN32) && !defined(__APPLE__)
//#define USE_XINPUT
#define USE_LINUX
#endif

#ifdef USE_XINPUT
#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <X11/cursorfont.h>
#endif

#include <PluginUtil/PluginMessageTypes.h>
#include <util/UDPComm.h>

SteeringWheelPlugin *SteeringWheelPlugin::plugin = NULL;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSteeringWheel(scene);
}

// Define the built in VrmlNodeType:: "SteeringWheel" fields

VrmlNodeType *VrmlNodeSteeringWheel::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("SteeringWheel", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addEventIn("set_time", VrmlField::SFTIME);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("joystickNumber", VrmlField::SFINT32);
    t->addEventOut("buttons_changed", VrmlField::MFINT32);
    t->addEventOut("axes_changed", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeSteeringWheel::nodeType() const
{
    return defineType(0);
}

VrmlNodeSteeringWheel::VrmlNodeSteeringWheel(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
    , d_joystickNumber(-1)
{
    setModified();
}

VrmlNodeSteeringWheel::VrmlNodeSteeringWheel(const VrmlNodeSteeringWheel &n)
    : VrmlNodeChild(n.d_scene)
    , d_enabled(n.d_enabled)
    , d_joystickNumber(n.d_joystickNumber)
{

    setModified();
}

VrmlNodeSteeringWheel::~VrmlNodeSteeringWheel()
{
}

VrmlNode *VrmlNodeSteeringWheel::cloneMe() const
{
    return new VrmlNodeSteeringWheel(*this);
}

VrmlNodeSteeringWheel *VrmlNodeSteeringWheel::toSteeringWheel() const
{
    return (VrmlNodeSteeringWheel *)this;
}

ostream &VrmlNodeSteeringWheel::printFields(ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!d_joystickNumber.get())
        PRINT_FIELD(joystickNumber);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeSteeringWheel::setField(const char *fieldName,
                                     const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(joystickNumber, SFInt)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeSteeringWheel::getField(const char *fieldName)
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "joystickNumber") == 0)
        return &d_joystickNumber;
    else if (strcmp(fieldName, "axes_changed") == 0)
        return &d_axes;
    else if (strcmp(fieldName, "buttons_changed") == 0)
        return &d_buttons;
    else
        cout << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeSteeringWheel::eventIn(double timeStamp,
                                    const char *eventName,
                                    const VrmlField *fieldValue)
{
    if (strcmp(eventName, "set_time") == 0)
    {
    }
    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

char SteeringWheelPlugin::readps2(int fd)
{
    char ch;

    while (read(fd, &ch, 1) && (ch == (char)0xfa || ch == (char)0xaa))
    {
        //fprintf(stderr,"<%02X>",ch&0xff);
    }
    //fprintf(stderr,"[%02X]",ch&0xff);
    return (ch);
}

bool SteeringWheelPlugin::initJoystick(int joystickNumber)
{
    if (joystickNumber > MAX_NUMBER_JOYSTICKS)
        return false;
    if (coVRMSController::instance()->isMaster())
    {
#ifdef WIN32
        return false;
#else
#ifdef HAVE_NO_DEV_INPUT_JOYSTICK
        const char *devName = "/dev/js";
#else
        const char *devName = "/dev/input/js";
#endif
#define DEVICE_LEN 512
        string joystickDev = coCoviseConfig::getEntry("value", "COVER.Plugin.Joystick.Device", devName);
        char device[DEVICE_LEN];
        strcpy(device, joystickDev.c_str());
        int deviceLen = strlen(device);
        sprintf(device + deviceLen, "%d", joystickNumber);
        if ((fd[joystickNumber] = open(device, O_RDONLY)) < 0)
        {
            fprintf(stderr, "joystickdevice %s ", device);
            perror("joystick initialisation failed: ");
            return false;
        }

        ioctl(fd[joystickNumber], JSIOCGVERSION, &version);
        ioctl(fd[joystickNumber], JSIOCGAXES, &cover->number_axes[joystickNumber]);
        if (cover->number_axes[joystickNumber] > 0)
        {
            cover->axes[joystickNumber] = new float[cover->number_axes[joystickNumber]];
            for (int i = 0; i < cover->number_axes[joystickNumber]; i++)
                cover->axes[joystickNumber][i] = 0;
        }
        ioctl(fd[joystickNumber], JSIOCGBUTTONS, &cover->number_buttons[joystickNumber]);
        if (cover->number_buttons[joystickNumber] > 0)
        {
            cover->buttons[joystickNumber] = new int[cover->number_buttons[joystickNumber]];
            for (int i = 0; i < cover->number_buttons[joystickNumber]; i++)
                cover->buttons[joystickNumber][i] = 0;
        }
        //   ioctl(fd[joystickNumber], JSIOCGNAME(NAME_LENGTH), name);
        fcntl(fd[joystickNumber], F_SETFL, O_NONBLOCK);
    }
    else
    {
        int i;
        for (i = 0; i < MAX_NUMBER_JOYSTICKS; i++)
        {
            if (cover->number_axes[i] > 0)
            {
                cover->axes[i] = NULL;
            }
            if (cover->number_buttons[i] > 0)
            {
                cover->buttons[i] = NULL;
            }
        }
        return false;
#endif
    }
    return true;
}

void VrmlNodeSteeringWheel::render(Viewer *)
{
    if (!d_enabled.get())
        return;

    int joystickNumber = d_joystickNumber.get();
    if (((joystickNumber >= SteeringWheelPlugin::plugin->numLocalJoysticks) && SteeringWheelPlugin::plugin->haveMouse) || (joystickNumber >= SteeringWheelPlugin::plugin->numLocalJoysticks + 1))
        return;
    double timeStamp = System::the->time();
    //if (eventType & JOYSTICK_AXES_EVENTS)
    {
        if (cover->number_axes[joystickNumber] && cover->axes[joystickNumber])
        {
            d_axes.set(cover->number_axes[joystickNumber], cover->axes[joystickNumber]);
            // Send the new value
            eventOut(timeStamp, "axes_changed", d_axes);
        }
    }

    //if (eventType & JOYSTICK_BUTTON_EVENTS)
    {
        if (cover->number_buttons[joystickNumber] && cover->buttons[joystickNumber])
        {
            d_buttons.set(cover->number_buttons[joystickNumber],
                          cover->buttons[joystickNumber]);
            // Send the new value
            eventOut(timeStamp, "buttons_changed", d_buttons);
        }
    }
}

int SteeringWheelPlugin::getData(int joystickNumber)
{
    int ret = 0;
    if (((joystickNumber >= numLocalJoysticks) && haveMouse) || (joystickNumber >= numLocalJoysticks + 1))
        return 0;
#ifndef WIN32
    int button_pressed = -1;
    int button_released = -1;
    struct js_event js;
    if (fd[joystickNumber] == -1)
        if (!initJoystick(joystickNumber))
        {
            fd[joystickNumber] = -1;
            return 0;
        }
    while (read(fd[joystickNumber], &js,
                sizeof(struct js_event)) == sizeof(struct js_event))
    {
#ifdef DEBUG
        fprintf(stderr, "Event: type %d, time %d, number %d, value %d\n",
                js.type, js.time, js.number, js.value);
#endif
        switch (js.type & ~JS_EVENT_INIT)
        {
        case JS_EVENT_BUTTON:
            ret = ret | JOYSTICK_BUTTON_EVENTS;
            cout << "js.value" << js.value << endl;
            cout << "js.number" << js.number << endl;
            cout << "joystickNumber" << joystickNumber << endl;
            if (js.value)
            {
                cover->buttons[joystickNumber][js.number] = 1;
                button_pressed = js.number;
            }
            else
            {
                cover->buttons[joystickNumber][js.number] = 0;
                button_released = js.number;
            }
            break;
        case JS_EVENT_AXIS:
            ret = ret | JOYSTICK_AXES_EVENTS;
            cover->axes[joystickNumber][js.number] = js.value / 32767.0f;
            break;
        }
        if ((button_pressed != -1) || (button_released != -1))
            break;
    }
#endif
    return ret;
}
#ifdef WIN32

//-----------------------------------------------------------------------------
// Name: EnumJoysticksCallback()
// Desc: Called once for each enumerated joystick. If we find one, create a
//       device interface on it so we can play with it.
//-----------------------------------------------------------------------------
BOOL CALLBACK SteeringWheelPlugin::EnumJoysticksCallback(const DIDEVICEINSTANCE *pdidInstance,
                                                         VOID *pContext)
{
    HRESULT hr;
    // Obtain an interface to the enumerated joystick.
    hr = SteeringWheelPlugin::plugin->g_pDI->CreateDevice(pdidInstance->guidInstance, &SteeringWheelPlugin::plugin->g_pJoystick[SteeringWheelPlugin::plugin->numLocalJoysticks], NULL);

    // If it failed, then we can't use this joystick. (Maybe the user unplugged
    // it while we were in the middle of enumerating it.)
    if (FAILED(hr))
        return DIENUM_CONTINUE;

    // Set the data format to "simple joystick" - a predefined data format
    //
    // A data format specifies which controls on a device we are interested in,
    // and how they should be reported. This tells DInput that we will be
    // passing a DIJOYSTATE2 structure to IDirectInputDevice::GetDeviceState().
    if (FAILED(hr = SteeringWheelPlugin::plugin->g_pJoystick[SteeringWheelPlugin::plugin->numLocalJoysticks]->SetDataFormat(&c_dfDIJoystick2)))
        return DIENUM_CONTINUE;

    // Set the cooperative level to let DInput know how this device should
    // interact with the system and with other DInput applications.
    if (FAILED(hr = SteeringWheelPlugin::plugin->g_pJoystick[SteeringWheelPlugin::plugin->numLocalJoysticks]->SetCooperativeLevel(SteeringWheelPlugin::plugin->window.handle_, DISCL_EXCLUSIVE | DISCL_BACKGROUND)))
        return DIENUM_CONTINUE;

    // Enumerate the joystick objects. The callback function enabled user
    // interface elements for objects that are found, and sets the min/max
    // values property for discovered axes.
    if (FAILED(hr = SteeringWheelPlugin::plugin->g_pJoystick[SteeringWheelPlugin::plugin->numLocalJoysticks]->EnumObjects(EnumObjectsCallback,
                                                                                                                          NULL, DIDFT_ALL)))
        return DIENUM_CONTINUE;
    if ((cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] > 0) || (cover->number_buttons[SteeringWheelPlugin::plugin->numLocalJoysticks] > 0))
    {
        cover->numJoysticks++;
        SteeringWheelPlugin::plugin->numLocalJoysticks++;
    }
    return DIENUM_CONTINUE;
}

//-----------------------------------------------------------------------------
// Name: EnumObjectsCallback()
// Desc: Callback function for enumerating objects (axes, buttons, POVs) on a
//       joystick. This function enables user interface elements for objects
//       that are found to exist, and scales axes min/max values.
//-----------------------------------------------------------------------------
BOOL CALLBACK SteeringWheelPlugin::EnumObjectsCallback(const DIDEVICEOBJECTINSTANCE *pdidoi,
                                                       VOID *pContext)
{
    //HWND hDlg = (HWND)pContext;

    // For axes that are returned, set the DIPROP_RANGE property for the
    // enumerated axis in order to scale min/max values.
    if (pdidoi->dwType & DIDFT_AXIS)
    {
        DIPROPRANGE diprg;
        diprg.diph.dwSize = sizeof(DIPROPRANGE);
        diprg.diph.dwHeaderSize = sizeof(DIPROPHEADER);
        diprg.diph.dwHow = DIPH_BYID;
        diprg.diph.dwObj = pdidoi->dwType; // Specify the enumerated axis
        diprg.lMin = -1000;
        diprg.lMax = +1000;

        // Set the range for the axis
        if (FAILED(SteeringWheelPlugin::plugin->g_pJoystick[SteeringWheelPlugin::plugin->numLocalJoysticks]->SetProperty(DIPROP_RANGE, &diprg.diph)))
            return DIENUM_STOP;
    }

    if (pdidoi->guidType == GUID_Button)
    {
        cover->number_buttons[SteeringWheelPlugin::plugin->numLocalJoysticks]++;
    }
    if (pdidoi->guidType == GUID_XAxis)
    {
        if (cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] < 1)
            cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] = 1;
    }
    if (pdidoi->guidType == GUID_YAxis)
    {
        if (cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] < 2)
            cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] = 2;
    }
    if (pdidoi->guidType == GUID_ZAxis)
    {
        if (cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] < 3)
            cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] = 3;
    }
    if (pdidoi->guidType == GUID_RxAxis)
    {
        if (cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] < 4)
            cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] = 4;
    }
    if (pdidoi->guidType == GUID_RyAxis)
    {
        if (cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] < 5)
            cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] = 5;
    }
    if (pdidoi->guidType == GUID_RzAxis)
    {
        if (cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] < 6)
            cover->number_axes[SteeringWheelPlugin::plugin->numLocalJoysticks] = 6;
    }
    if (pdidoi->guidType == GUID_Slider)
    {
        cover->number_sliders[SteeringWheelPlugin::plugin->numLocalJoysticks]++;
    }
    if (pdidoi->guidType == GUID_POV)
    {
        cover->number_POVs[SteeringWheelPlugin::plugin->numLocalJoysticks]++;
    }

    return DIENUM_CONTINUE;
}

#endif

//-----------------------------------------------------------------------------
// Name: UpdateInputState()
// Desc: Get the input device's state and display it.
//-----------------------------------------------------------------------------
void SteeringWheelPlugin::UpdateInputState()
{
    int i;

#ifdef WIN32
    if (coVRMSController::instance()->isMaster())
    {
        HRESULT hr;
        DIJOYSTATE2 js; // DInput joystick state

        for (i = 0; i < numLocalJoysticks; i++)
        {

            // Poll the device to read the current state

            hr = g_pJoystick[i]->Poll();
            if (FAILED(hr))
            {
                // DInput is telling us that the input stream has been
                // interrupted. We aren't tracking any state between polls, so
                // we don't have any special reset that needs to be done. We
                // just re-acquire and try again.
                hr = g_pJoystick[i]->Acquire();
                while (hr == DIERR_INPUTLOST)
                    hr = g_pJoystick[i]->Acquire();

                // hr may be DIERR_OTHERAPPHASPRIO or other errors.  This
                // may occur when the app is minimized or in the process of
                // switching, so just try again later
                return;
            }

            // Get the input's device state
            if (FAILED(hr = g_pJoystick[i]->GetDeviceState(sizeof(DIJOYSTATE2), &js)))
                return; // The device should have been acquired during the Poll()
            int n;

            for (n = 0; n < cover->number_buttons[i]; n++)
            {
                if (js.rgbButtons[n] & 0x80)
                    cover->buttons[i][n] = 1;
                else
                    cover->buttons[i][n] = 0;
            }
            for (n = 0; n < cover->number_sliders[i]; n++)
            {
                cover->sliders[i][n] = js.rglSlider[n] / 1000.0f;
            }
            if (cover->number_axes[i] > 0)
                cover->axes[i][0] = js.lX / 1000.0f;
            if (cover->number_axes[i] > 1)
                cover->axes[i][1] = js.lY / 1000.0f;
            if (cover->number_axes[i] > 2)
                cover->axes[i][2] = js.lZ / 1000.0f;
            if (cover->number_axes[i] > 3)
                cover->axes[i][3] = js.lRx / 1000.0f;
            if (cover->number_axes[i] > 4)
                cover->axes[i][4] = js.lRy / 1000.0f;
            if (cover->number_axes[i] > 5)
                cover->axes[i][5] = js.lRz / 1000.0f;

            for (n = 0; n < cover->number_POVs[i]; n++)
            {
                cover->POVs[i][n] = js.rgdwPOV[n];
            }
            coVRMSController::instance()->sendSlaves((char *)&cover->number_axes[i], sizeof(int));
            if (cover->number_axes[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->axes[i][0]), cover->number_axes[i] * sizeof(float));
            }
            coVRMSController::instance()->sendSlaves((char *)&cover->number_buttons[i], sizeof(int));
            if (cover->number_buttons[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->buttons[i][0]), cover->number_buttons[i] * sizeof(int));
            }
            coVRMSController::instance()->sendSlaves((char *)&cover->number_sliders[i], sizeof(int));
            if (cover->number_sliders[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->sliders[i][0]), cover->number_sliders[i] * sizeof(int));
            }
            coVRMSController::instance()->sendSlaves((char *)&cover->number_POVs[i], sizeof(int));
            if (cover->number_POVs[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->POVs[i][0]), cover->number_POVs[i] * sizeof(int));
            }
        }
    }
    else
    {
        for (i = 0; i < numLocalJoysticks; i++)
        {
            coVRMSController::instance()->readMaster((char *)&cover->number_axes[i], sizeof(int));
            if (cover->number_axes[i])
            {
                if (cover->axes[i] == NULL)
                {
                    cover->axes[i] = new float[cover->number_axes[i]];
                    for (int n = 0; n < cover->number_axes[i]; n++)
                        cover->axes[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->axes[i][0]), cover->number_axes[i] * sizeof(float));
            }
            coVRMSController::instance()->readMaster((char *)&cover->number_buttons[i], sizeof(int));

            if (cover->number_buttons[i])
            {
                if (cover->buttons[i] == NULL)
                {
                    cover->buttons[i] = new int[cover->number_buttons[i]];
                    for (int n = 0; n < cover->number_buttons[i]; n++)
                        cover->buttons[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->buttons[i][0]), cover->number_buttons[i] * sizeof(int));
            }
            coVRMSController::instance()->readMaster((char *)&cover->number_sliders[i], sizeof(int));

            if (cover->number_sliders[i])
            {
                if (cover->sliders[i] == NULL)
                {
                    cover->sliders[i] = new float[cover->number_sliders[i]];
                    for (int n = 0; n < cover->number_sliders[i]; n++)
                        cover->sliders[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->sliders[i][0]), cover->number_sliders[i] * sizeof(int));
            }
            coVRMSController::instance()->readMaster((char *)&cover->number_POVs[i], sizeof(int));

            if (cover->number_POVs[i])
            {
                if (cover->POVs[i] == NULL)
                {
                    cover->POVs[i] = new float[cover->number_POVs[i]];
                    for (int n = 0; n < cover->number_POVs[i]; n++)
                        cover->POVs[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->POVs[i][0]), cover->number_POVs[i] * sizeof(int));
            }
        }
    }
#else
    int eventType = 0;
    if (coVRMSController::instance()->isMaster())
    {
        for (i = 0; i < numLocalJoysticks; i++)
        {

            eventType = getData(i);
            coVRMSController::instance()->sendSlaves((char *)&eventType, sizeof(int));
            if (eventType & JOYSTICK_AXES_EVENTS)
            {
                coVRMSController::instance()->sendSlaves((char *)&cover->number_axes[i], sizeof(int));
                if (cover->number_axes[i])
                {
                    coVRMSController::instance()->sendSlaves((char *)&(cover->axes[i][0]), cover->number_axes[i] * sizeof(float));
                }
            }

            if (eventType & JOYSTICK_BUTTON_EVENTS)
            {
                coVRMSController::instance()->sendSlaves((char *)&cover->number_buttons[i], sizeof(int));
                if (cover->number_buttons[i])
                {
                    coVRMSController::instance()->sendSlaves((char *)&(cover->buttons[i][0]), cover->number_buttons[i] * sizeof(int));
                }
            }
        }
    }
    else
    {

        for (i = 0; i < numLocalJoysticks; i++)
        {

            coVRMSController::instance()->readMaster((char *)&eventType, sizeof(int));
            if (eventType & JOYSTICK_AXES_EVENTS)
            {
                coVRMSController::instance()->readMaster((char *)&cover->number_axes[i], sizeof(int));
                if (cover->number_axes[i])
                {
                    if (cover->axes[i] == NULL)
                    {
                        cover->axes[i] = new float[cover->number_axes[i]];
                        for (int n = 0; n < cover->number_axes[i]; n++)
                            cover->axes[i][n] = 0;
                    }
                    coVRMSController::instance()->readMaster((char *)&(cover->axes[i][0]), cover->number_axes[i] * sizeof(float));
                }
            }

            if (eventType & JOYSTICK_BUTTON_EVENTS)
            {
                coVRMSController::instance()->readMaster((char *)&cover->number_buttons[i], sizeof(int));

                if (cover->number_buttons[i])
                {
                    if (cover->buttons[i] == NULL)
                    {
                        cover->buttons[i] = new int[cover->number_buttons[i]];
                        for (int n = 0; n < cover->number_buttons[i]; n++)
                            cover->buttons[i][n] = 0;
                    }
                    coVRMSController::instance()->readMaster((char *)&(cover->buttons[i][0]), cover->number_buttons[i] * sizeof(int));
                }
            }
        }
    }
#ifdef USE_XINPUT
    XEvent event;
    if (haveMouse)
    {
        while (XCheckTypedEvent(display, buttonPressEventType, &event))
        {
            XDeviceButtonEvent *bev = (XDeviceButtonEvent *)&event;
            if (bev->button == 4)
            {
                wheelcounter++;
            }
            else if (bev->button == 5)
            {
                wheelcounter--;
            }
            else
            {
                buttonPressMask |= 1 << (bev->button - 1);
                cout << "buttonPressMask: " << buttonPressMask << endl;
            }
            if (buttonPressMask == 1)
                wheelcounter = 0;
            cover->axes[numLocalJoysticks][0] = wheelcounter / 116.0;
            //cout << "WheelCounter: " << wheelcounter << endl;
        }
    }
    while (XCheckTypedEvent(display, buttonReleaseEventType, &event))
    {
        XDeviceButtonEvent *bev = (XDeviceButtonEvent *)&event;
        if (bev->button != 4 && bev->button != 5)
        {
            buttonPressMask &= ~(1L << (bev->button - 1));
        }
        cout << "buttonPressMask: " << buttonPressMask << endl;
    }
#endif
#ifdef USE_LINUX
    if (device != -1)
    {
        while (read(device, buffer, 4) == 4)
        {
            wheelcounter -= buffer[3];
            buffer[0] &= (1 << NUM_BUTTONS) - 1;
            if (buffer[0] & 2)
            {
                wheelcounter = 0;
                cout << "reset wheelcounter: " << (int)buffer[0] << endl;
            }
            //cout << "button: " << (int)buffer[0] << endl;
        }
        //printf("buffer: %d %d %d %d\n", buffer[0], buffer[1], buffer[2], buffer[3]);
        //printf("wheelcounter: %d\n", wheelcounter);
        cover->axes[numLocalJoysticks][0] = wheelcounter / 116.0;
    }
    if (horndevice != -1 && device != -1)
    {
        while (read(horndevice, buffer, 3) == 3)
        {
            buffer[0] &= (1 << NUM_BUTTONS) - 1;
            cout << "hornbutton: " << (int)buffer[0] << endl;
            if ((buffer[0] & 1) != 0)
            {
                cover->buttons[numLocalJoysticks][3] = 1;
            }
            else
            {
                cover->buttons[numLocalJoysticks][3] = 0;
            }
        }
        //printf("buffer: %d %d %d %d\n", buffer[0], buffer[1], buffer[2], buffer[3]);
        //printf("wheelcounter: %d\n", wheelcounter);
    }
#endif

    if (haveMouse)
    {
        if (coVRMSController::instance()->isMaster())
            coVRMSController::instance()->sendSlaves((char *)&(cover->axes[numLocalJoysticks][0]), sizeof(float));
        else
            coVRMSController::instance()->readMaster((char *)&(cover->axes[numLocalJoysticks][0]), sizeof(float));
    }

#endif
    return;
}

void SteeringWheelPlugin::printData(int joystickNumber)
{
    for (int i = 0; i < cover->number_axes[joystickNumber]; i++)
    {
        printf("%d: %f ", i, cover->axes[joystickNumber][i]);
        if (((i + 1) % 4 == 0))
            printf("\n");
    }
    printf("\n");
    for (int i = 0; i < cover->number_buttons[joystickNumber]; i++)
    {
        printf("%d: %d ", i, cover->buttons[joystickNumber][i]);
        if (((i + 1) % 5 == 0))
            printf("\n");
    }
    printf("\n");
}

void SteeringWheelPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (tUIItem == softResetWheelButton)
        {
            std::cout << "Soft reset of steering wheel..." << std::endl;
            if (sitzkiste)
                sitzkiste->softResetWheel();
        }

        if (tUIItem == cruelResetWheelButton)
        {
            std::cout << "Cruel reset steering wheel..." << std::endl;
            if (sitzkiste)
                sitzkiste->cruelResetWheel();
        }

        if (tUIItem == shutdownWheelButton)
        {
            std::cout << "Shutting down steering wheel..." << std::endl;
            if (sitzkiste)
                sitzkiste->shutdownWheel();
        }

#ifdef __XENO__
        if (tUIItem == platformToGroundButton)
        {
            EinspurDynamikRealtime *einspur = dynamic_cast<EinspurDynamikRealtime *>(dynamics);
            FourWheelDynamicsRealtime *fourwheel = dynamic_cast<FourWheelDynamicsRealtime *>(dynamics);
#ifdef HAVE_CARDYNAMICSCGA
            CarDynamicsCGARealtime *cardynCGA = dynamic_cast<CarDynamicsCGARealtime *>(SteeringWheelPlugin::plugin->dynamics);
#endif
            if (einspur)
            {
                einspur->platformToGround();
            }
            else if (fourwheel)
            {
                fourwheel->platformToGround();
            }
#ifdef HAVE_CARDYNAMICSCGA
            else if (cardynCGA)
            {
                cardynCGA->platformToGround();
            }
#endif
        }
        else if (tUIItem == platformReturnToActionButton)
        {
            EinspurDynamikRealtime *einspur = dynamic_cast<EinspurDynamikRealtime *>(dynamics);
            FourWheelDynamicsRealtime *fourwheel = dynamic_cast<FourWheelDynamicsRealtime *>(dynamics);
#ifdef HAVE_CARDYNAMICSCGA
            CarDynamicsCGARealtime *cardynCGA = dynamic_cast<CarDynamicsCGARealtime *>(SteeringWheelPlugin::plugin->dynamics);
#endif
            if (einspur)
            {
                einspur->platformReturnToAction();
            }
            else if (fourwheel)
            {
                fourwheel->platformReturnToAction();
            }
#ifdef HAVE_CARDYNAMICSCGA
            else if (cardynCGA)
            {
                cardynCGA->platformToGround();
            }
#endif
        }
#endif
    }
}

void SteeringWheelPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == springConstant)
    {
    }
}

SteeringWheelPlugin::SteeringWheelPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool SteeringWheelPlugin::init()
{
    fprintf(stderr, "SteeringWheelPlugin::SteeringWheelPlugin\n");
    if (plugin)
        return false;
    plugin = this;
    SteeringWheelTab = new coTUITab("SteeringWheel", coVRTui::instance()->mainFolder->getID());
    SteeringWheelTab->setPos(0, 0);

    blockAngle = new coTUIEditFloatField("block angle", SteeringWheelTab->getID());
    blockAngle->setEventListener(this);
    blockAngle->setValue(6);
    blockAngle->setPos(1, 0);

    blockAngleLabel = new coTUILabel("block angle:", SteeringWheelTab->getID());
    blockAngleLabel->setPos(0, 0);

    springConstant = new coTUIEditFloatField("spring constant", SteeringWheelTab->getID());
    springConstant->setEventListener(this);
    springConstant->setValue(0.6);
    springConstant->setPos(1, 1);

    springConstantLabel = new coTUILabel("spring constant:", SteeringWheelTab->getID());
    springConstantLabel->setPos(0, 1);

    dampingConstant = new coTUIEditFloatField("damping constant", SteeringWheelTab->getID());
    dampingConstant->setEventListener(this);
    dampingConstant->setValue(0.2);
    dampingConstant->setPos(1, 2);

    dampingConstantLabel = new coTUILabel("damping constant:", SteeringWheelTab->getID());
    dampingConstantLabel->setPos(0, 2);

    rumbleFactor = new coTUIEditFloatField("rumble factor", SteeringWheelTab->getID());
    rumbleFactor->setEventListener(this);
    rumbleFactor->setValue(3);
    rumbleFactor->setPos(1, 3);

    rumbleFactorLabel = new coTUILabel("rumble factor:", SteeringWheelTab->getID());
    rumbleFactorLabel->setPos(0, 3);

    drillingFrictionConstant = new coTUIEditFloatField("drilling friction constant", SteeringWheelTab->getID());
    drillingFrictionConstant->setEventListener(this);
    drillingFrictionConstant->setValue(3);
    drillingFrictionConstant->setPos(1, 4);

    drillingFrictionConstantLabel = new coTUILabel("drilling friction constant:", SteeringWheelTab->getID());
    drillingFrictionConstantLabel->setPos(0, 4);

    velocityImpactFactor = new coTUIEditFloatField("velocity impact factor", SteeringWheelTab->getID());
    velocityImpactFactor->setEventListener(this);
    velocityImpactFactor->setValue(0.3);
    velocityImpactFactor->setPos(1, 5);

    velocityImpactFactorLabel = new coTUILabel("velocity impact factor:", SteeringWheelTab->getID());
    velocityImpactFactorLabel->setPos(0, 5);

    velocityImpactFactorRumble = new coTUIEditFloatField("rumble velocity impact factor", SteeringWheelTab->getID());
    velocityImpactFactorRumble->setEventListener(this);
    velocityImpactFactorRumble->setValue(0.07);
    velocityImpactFactorRumble->setPos(1, 6);

    velocityImpactFactorRumbleLabel = new coTUILabel("rumble velocity impact factor:", SteeringWheelTab->getID());
    velocityImpactFactorRumbleLabel->setPos(0, 6);

    softResetWheelButton = new coTUIButton("Soft Reset", SteeringWheelTab->getID());
    softResetWheelButton->setEventListener(this);
    softResetWheelButton->setPos(0, 7);

    cruelResetWheelButton = new coTUIButton("Cruel Reset", SteeringWheelTab->getID());
    cruelResetWheelButton->setEventListener(this);
    cruelResetWheelButton->setPos(0, 8);

    shutdownWheelButton = new coTUIButton("Shutdown Wheel", SteeringWheelTab->getID());
    shutdownWheelButton->setEventListener(this);
    shutdownWheelButton->setPos(1, 7);

    platformToGroundButton = new coTUIButton("Platform to ground", SteeringWheelTab->getID());
    platformToGroundButton->setEventListener(this);
    platformToGroundButton->setPos(0, 10);
    platformReturnToActionButton = new coTUIButton("Platform return to action", SteeringWheelTab->getID());
    platformReturnToActionButton->setEventListener(this);
    platformReturnToActionButton->setPos(1, 10);

    simulatorJoystick = -1;
    oldSimulatorJoystick = -1;
    /* porsche = new Porsche("COM1",9600); //115200

   if(porsche->deviceOpen())
   {
   porsche->writeChars((unsigned char *)"init\n",5);
   // char buf[11];
   // buf[0]='a';
   // while(buf[0]!='x')
   //  {
   //    int num = porsche->readBlock(buf,10);
   //    buf[num]='\0';
   //    cout << "num: " << num << " :";
   //    cout << buf << endl;
   // }
   }*/
    conn = NULL;

    numFloatsOut = 62;
    numIntsOut = 61;

    floatValuesOut = new float[numFloatsOut];
    intValuesOut = new int[numIntsOut];

    updateRate = coCoviseConfig::getFloat("COVER.Plugin.SteeringWheel.UpdateRate", 0.1);

    if (coCoviseConfig::isOn("COVER.Plugin.SteeringWheel.CAN", false) == true)
    {
        std::cout << "Using CAN (digital) wheel control..." << std::endl;
#ifdef HAVE_PCAN
        sitzkiste = new CAN();
#endif
    }
    else if (coCoviseConfig::isOn("COVER.Plugin.SteeringWheel.FKFS", false) == true)
    {
        sitzkiste = new FKFS();
        std::cout << "Using FKFS (analog) wheel control..." << std::endl;
    }
    else
    {
        sitzkiste = new Keyboard();
        std::cout << "Trallala! No force feedback wheel control defined... using Keyboard" << std::endl;
    }

    /*
	vd = NULL;
   if(coCoviseConfig::isOn("COVER.Plugin.SteeringWheel.ITM",false)==true)
   {
      std::cout << "Using ITM vehicle dynamics (obsolete)..." << std::endl;
      vd = new ITM();
   }
   else if(coCoviseConfig::isOn("COVER.Plugin.SteeringWheel.FKFSDynamics",false)==true)
   {
      vd = new FKFSDynamics();
      std::cout << "Using FKFS vehicle dynamics (obsolete)..." << std::endl;
   }
   else {
   */
    std::string dynString = coCoviseConfig::getEntry("value", "COVER.Plugin.SteeringWheel.Dynamics", "EinspurDynamik");
    if (dynString == "TestDynamics")
    {
        dynamics = new TestDynamics(); //mass, moment of inertia, front to point of mass, rear to point of mass
        std::cout << "Using test vehicle dynamics..." << std::endl;
    }
	else if (dynString == "ITMDynamics")
    {
        dynamics = new ITMDynamics(); //mass, moment of inertia, front to point of mass, rear to point of mass
        std::cout << "Using ITM vehicle dynamics..." << std::endl;
    }
    else if (dynString == "FKFSDynamics")
    {
        dynamics = new FKFSDynamics(); //mass, moment of inertia, front to point of mass, rear to point of mass
        std::cout << "Using FKFS vehicle dynamics..." << std::endl;
    }
    else if (dynString == "PorscheRealtimeDynamics")
    {
        dynamics = new PorscheRealtimeDynamics();
        std::cout << "Using Porsche realtime vehicle dynamics..." << std::endl;
    }
    else if (dynString == "HLRSRealtimeDynamics")
    {
        dynamics = new HLRSRealtimeDynamics();
        std::cout << "Using HLRS realtime vehicle dynamics..." << std::endl;
    }  
#ifdef __XENO__
    else if (dynString == "FourWheelDynamicsRealtime")
    {
        dynamics = new FourWheelDynamicsRealtime();
        std::cout << "Using four wheel vehicle dynamics..." << std::endl;
    }
    else if (dynString == "EinspurDynamikRealtime")
    {
        dynamics = new EinspurDynamikRealtime();
        std::cout << "Using Einspur realtime vehicle dynamics..." << std::endl;
    }
#endif
#ifdef HAVE_CARDYNAMICSCGA
    else if (dynString == "CarDynamicsCGA")
    {
        dynamics = new CarDynamicsCGA();
        std::cout << "Using CarDynamicsCGA dynamics..." << std::endl;
    }
    else if (dynString == "CarDynamicsRtus")
    {
        dynamics = new CarDynamicsRtus();
        std::cout << "Using CarDynamicsRtus dynamics..." << std::endl;
    }
#ifdef __XENO__
    else if (dynString == "CarDynamicsCGARealtime")
    {
        dynamics = new CarDynamicsCGARealtime();
        std::cout << "Using CarDynamicsCGARealtime dynamics..." << std::endl;
    }
#endif
#endif
    else
    {
        dynamics = new EinspurDynamik(); //mass, moment of inertia, front to point of mass, rear to point of mass
        std::cout << "Using Einspur vehicle dynamics..." << std::endl;
    }
    // }

    if (coCoviseConfig::isOn("COVER.Plugin.SteeringWheel.PorscheServer", false) == true)
    {
        std::cout << "Using Porsche server data controller..." << std::endl;
        dataController = new PorscheController();
    }
    else
    {
        dataController = NULL;
    }

    serverHost = NULL;
    localHost = new Host("localhost");
    port = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.Server", 1001);
    serverPort = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.SimServer", 1002);
    toClientConn = NULL;
    serverConn = new ServerConnection(serverPort, 1234, 0);
    if (!serverConn->getSocket())
    {
        cout << "tried to open server Port " << serverPort << endl;
        cout << "Creation of server failed!" << endl;
        cout << "Port-Binding failed! Port already bound?" << endl;
        delete serverConn;
        serverConn = NULL;
    }

    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    cout << "Set socket options..." << endl;
    if (serverConn)
    {
        setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

        cout << "Set server to listen mode..." << endl;
        serverConn->listen();
        if (!serverConn->is_connected()) // could not open server port
        {
            fprintf(stderr, "Could not open server port %d\n", serverPort);
            delete serverConn;
            serverConn = NULL;
        }
    }

    std::string line = coCoviseConfig::getEntry("COVER.Plugin.SteeringWheel.Server");
    if (!line.empty())
    {
        if (strcasecmp(line.c_str(), "NONE") == 0)
            serverHost = NULL;
        else
            serverHost = new Host(line.c_str());
        cout << serverHost->getName() << endl;
    }

    numFloats = 62;
    numInts = 61;

    floatValues = new float[numFloats];
    intValues = new int[numInts];

    buttonPressEventType = -1;
    haveMouse = false;
    haveHorn = false;
    buttonReleaseEventType = -1;
    plugin = this;
#ifndef WIN32
    std::string MouseDev = coCoviseConfig::getEntry("COVER.Plugin.SteeringWheel.PorscheMouse");
    cout << "MouseDev :" << MouseDev << endl;
    std::string HornDev = coCoviseConfig::getEntry("COVER.Plugin.SteeringWheel.PorscheHupe");
    cout << "HornDev  : " << HornDev << endl;

#ifdef USE_LINUX
    device = -1;
    horndevice = -1;
    wheelcounter = 0;

    if (!MouseDev.empty())
    {
        memset(buffer, 0, 4);
        device = open(MouseDev.c_str(), O_RDWR | O_NONBLOCK);
        if (device == -1)
        {
            fprintf(stderr, "SteeringWheel: could not open Mouse device %s\n\n", MouseDev.c_str());
        }
        else
        {

            char ch;
            char getdevtype = 0xf2, disableps2 = 0xf5, imps2[6] = { (char)0xf3, (char)200, (char)0xf3, (char)100, (char)0xf3, (char)80 }, resetps2 = 0xff;

            fprintf(stderr, "write disable\n");
            ssize_t iret = write(device, &disableps2, 1);
            if (iret != 1)
                fprintf(stderr, "SteeringWheel: error reading 'disableps2', wrong no. of arguments\n");

            tcflush(device, TCIFLUSH);
            iret = write(device, &getdevtype, 1);
            if (iret != 1)
                fprintf(stderr, "SteeringWheel: error reading 'getdevtype', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(device);

            iret = write(device, &resetps2, 1);
            if (iret != 1)
                fprintf(stderr, "SteeringWheel: error reading 'resetps2', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(device);
            tcflush(device, TCIFLUSH);
            iret = write(device, &getdevtype, 1);
            if (iret != 1)
                fprintf(stderr, "SteeringWheel: error reading 'getdevtype', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(device);

            iret = write(device, imps2, 6);
            if (iret != 6)
                fprintf(stderr, "SteeringWheel: error reading 'imps2',wrong no. of arguments\n");
            haveMouse = true;
        }
    }
    if (!HornDev.empty())
    {
        memset(buffer, 0, 4);
        horndevice = open(HornDev.c_str(), O_RDWR | O_NONBLOCK);
        if (horndevice == -1)
        {
            fprintf(stderr, "SteeringWheel: could not open Horn device %s\n\n", HornDev.c_str());
        }
        else
        {
            haveHorn = true;
            haveMouse = true;
        }
    }
#endif
    wheelcounter = 0;
#ifdef USE_XINPUT
    if (MouseDev)
    {
        display = XOpenDisplay(cover->inputDisplay);
        int numDevices = 0;
        XDeviceInfoPtr list = XListInputDevices(display, &numDevices);
        XDevice *mouseDevice = NULL;
        for (int i = 0; i < numDevices; i++)
        {
            if (!strcmp(list[i].name, MouseDev))
            {
                mouseDevice = XOpenDevice(display, list[i].id);
                if (mouseDevice)
                {
                    fprintf(stderr, "using %s as wheel mouse\n", MouseDev);
                    break;
                }
            }
        }

        cout << "mouseDevice" << mouseDevice << endl;
        int numEventClasses = 0;
        uint32_t eventClass;
        if (mouseDevice)
        {
            Window window = DefaultRootWindow(display);
            XEventClass *oldEventClasses, *oldAllClasses;
            int allEvents;
            XGetSelectedExtensionEvents(display, window, &numEventClasses, &oldEventClasses, &allEvents, &oldAllClasses);
            int i;
            for (i = 0; i < numEventClasses; i++)
            {
                eventClasses[i] = oldEventClasses[i];
            }
            cout << "numEventClasses" << numEventClasses << endl;
            DeviceButtonPress(mouseDevice, buttonPressEventType, eventClass);
            eventClasses[numEventClasses] = eventClass;
            eventTypes[numEventClasses] = buttonPressEventType;
            numEventClasses++;
            cout << "eventClass" << eventClass << endl;
            cout << "eventbuttonPressEventType" << buttonPressEventType << endl;

            DeviceButtonRelease(mouseDevice, buttonReleaseEventType, eventClass);
            eventClasses[numEventClasses] = eventClass;
            eventTypes[numEventClasses] = buttonReleaseEventType;
            numEventClasses++;

            cout << "eventClass" << eventClass << endl;
            cout << "buttonReleaseEventType" << buttonReleaseEventType << endl;
            XSelectExtensionEvent(display, window, eventClasses, numEventClasses);

            cout << "numEventClasses" << numEventClasses << endl;
            buttonPressMask = 0;
            wheelcounter = 0;
            if (buttonPressEventType != -1)
                haveMouse = true;
        }
        else
        {
            fprintf(stderr, "SteeringWheelPlugin: X Input device %s not found\n", MouseDev);
        }
    }
#endif
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((char *)&haveMouse, sizeof(bool));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&haveMouse, sizeof(bool));
    }
#endif

    cover->numJoysticks = numLocalJoysticks = 0;
    int i;
    for (i = 0; i < MAX_NUMBER_JOYSTICKS; i++)
    {
        cover->number_buttons[i] = 0;
        cover->number_sliders[i] = 0;
        cover->number_axes[i] = 0;
        cover->number_POVs[i] = 0;
        cover->buttons[i] = NULL;
        cover->sliders[i] = NULL;
        cover->axes[i] = NULL;
        cover->POVs[i] = NULL;
        fd[i] = -1;
    }
    for (i = 0; i < MAX_NUMBER_JOYSTICKS; i++)
    {
        if (initJoystick(i))
        {
            fprintf(stderr, "Found Joystick %d\n", i);
            cover->numJoysticks++;
            numLocalJoysticks++;
        }
        else
            break;
    }

    initUI();

    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
SteeringWheelPlugin::~SteeringWheelPlugin()
{
    fprintf(stderr, "SteeringWheelPlugin::~SteeringWheelPlugin\n");
    delete dynamics;
    delete dataController;
    delete conn;
    delete serverHost;
    delete[] floatValues;
    delete[] intValues;
    int i;
    for (i = 0; i < numLocalJoysticks; i++)
    {
#ifdef WIN32
        // Unacquire the device one last time just in case
        // the app tried to exit while the device is still acquired.
        if (g_pJoystick[i])
            g_pJoystick[i]->Unacquire();

        // Release any DirectInput objects.
        SAFE_RELEASE(g_pJoystick[i]);
#else
        close(fd[i]);
#endif
        delete cover->buttons[i];
        delete cover->sliders[i];
        delete cover->axes[i];
        delete cover->POVs[i];

        cover->number_buttons[i] = 0;
        cover->number_sliders[i] = 0;
        cover->number_axes[i] = 0;
        cover->number_POVs[i] = 0;
        cover->buttons[i] = NULL;
        cover->sliders[i] = NULL;
        cover->axes[i] = NULL;
        cover->POVs[i] = NULL;
        fd[i] = -1;
    }
    cover->numJoysticks = 0;
    numLocalJoysticks = 0;

    delete blockAngle;
    delete SteeringWheelTab;
    delete springConstant;
    delete drillingFrictionConstant;
    delete dampingConstant;
    delete rumbleFactor;
    delete velocityImpactFactor;
    delete velocityImpactFactorRumble;
    delete blockAngleLabel;
    delete springConstantLabel;
    delete drillingFrictionConstantLabel;
    delete dampingConstantLabel;
    delete rumbleFactorLabel;
    delete velocityImpactFactorLabel;
    delete velocityImpactFactorRumbleLabel;
    delete softResetWheelButton;
    delete cruelResetWheelButton;
    delete shutdownWheelButton;

    InputDevice::instance()->destroy();
    //delete sitzkiste;
}

bool SteeringWheelPlugin::readVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    if (conn == NULL)
        return false;
    while (numRead < numBytes)
    {
        readBytes = conn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes < 0)
        {
            cout << "error reading data from socket" << endl;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}
bool SteeringWheelPlugin::readClientVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    if (toClientConn == NULL)
        return false;
    while (numRead < numBytes)
    {
        readBytes = toClientConn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes < 0)
        {
            cout << "error reading data from socket" << endl;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}

bool SteeringWheelPlugin::sendValuesToClient()
{
    if (toClientConn == NULL)
        return false;
    int written;
    written = toClientConn->getSocket()->write(&numFloatsOut, sizeof(numFloats));
    if (written < 0)
        return false;
    written = toClientConn->getSocket()->write(&numIntsOut, sizeof(numInts));
    if (written < 0)
        return false;
    if (numFloatsOut > 0)
    {
        written = toClientConn->getSocket()->write(floatValuesOut, numFloatsOut * sizeof(float));
        if (written < numFloatsOut * sizeof(float))
            cout << "short write" << endl;
        return false;
    }
    if (numIntsOut > 0)
    {
        written = toClientConn->getSocket()->write(intValuesOut, numIntsOut * sizeof(int));
        if (written < 0)
            return false;
    }
    return true;
}

bool SteeringWheelPlugin::sendValues()
{
    if (conn == NULL)
        return false;
    int written;

    //std::cout << "Sending: ";

    written = conn->getSocket()->write(&numFloatsOut, sizeof(numFloats));
    if (written < 0)
        return false;
    //for(int i=0; i<sizeof(numFloats); ++i)
    //   std::cout << (int)(((char*)&numFloatsOut)[i]) << " ";

    written = conn->getSocket()->write(&numIntsOut, sizeof(numInts));
    if (written < 0)
        return false;
    //for(int i=0; i<sizeof(numInts); ++i)
    //   std::cout << (int)(((char*)&numIntsOut)[i]) << " ";

    if (numFloatsOut > 0)
    {
        written = conn->getSocket()->write(floatValuesOut, numFloatsOut * sizeof(float));
        if (written < numFloatsOut * sizeof(float))
        {
            cout << "float short write" << endl;
            return false;
        }
    }
    //for(int i=0; i<numFloatsOut*sizeof(float); ++i)
    //   std::cout << (int)(((char*)floatValuesOut)[i]) << " ";

    if (numIntsOut > 0)
    {
        written = conn->getSocket()->write(intValuesOut, numIntsOut * sizeof(int));
        if (written < numIntsOut * sizeof(int))
        {
            cout << "integer short write" << endl;
            return false;
        }
    }
    //for(int i=0; i<numIntsOut*sizeof(int); ++i)
    //   std::cout << (int)(((char*)intValuesOut)[i]) << " ";

    //std::cout << std::endl;

    return true;
}

int SteeringWheelPlugin::initUI()
{

    VrmlNamespace::addBuiltIn(VrmlNodeSteeringWheel::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeVehicle::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeRemoteVehicle::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodePorscheVFP::defineType());

#ifdef WIN32

    g_pDI = NULL;
    HRESULT hr;

    if (coVRMSController::instance()->isMaster())
    {

        // Register with the DirectInput subsystem and get a pointer
        // to a IDirectInput interface we can use.
        // Create a DInput object
        if (FAILED(hr = DirectInput8Create(GetModuleHandle(NULL), DIRECTINPUT_VERSION,
                                           IID_IDirectInput8, (VOID **)&g_pDI, NULL)))
            return hr;

        // Look for a simple joystick we can use for this sample program.
        if (FAILED(hr = g_pDI->EnumDevices(DI8DEVCLASS_GAMECTRL,
                                           EnumJoysticksCallback,
                                           this, DIEDFL_ATTACHEDONLY)))
            return hr;
    }

#endif

    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((char *)&numLocalJoysticks, sizeof(int));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&numLocalJoysticks, sizeof(int));
        if (numLocalJoysticks > 0)
        {
            cover->numJoysticks = numLocalJoysticks;
        }
    }
    int i;
    for (i = 0; i < numLocalJoysticks; i++)
    {
        if (cover->number_buttons[i] > 0)
        {
            cover->buttons[i] = new int[cover->number_buttons[i]];
            for (int n = 0; n < cover->number_buttons[i]; n++)
                cover->buttons[i][n] = 0;
        }
        else
            cover->buttons[i] = NULL;
        if (cover->number_sliders[i] > 0)
        {
            cover->sliders[i] = new float[cover->number_sliders[i]];
            for (int n = 0; n < cover->number_sliders[i]; n++)
                cover->sliders[i][n] = 0;
        }
        else
            cover->sliders[i] = NULL;
        if (cover->number_axes[i] > 0)
        {
            cover->axes[i] = new float[cover->number_axes[i]];
            for (int n = 0; n < cover->number_axes[i]; n++)
                cover->axes[i][n] = 0;
        }
        else
            cover->axes[i] = NULL;
        if (cover->number_POVs[i] > 0)
        {
            cover->POVs[i] = new float[cover->number_POVs[i]];
            for (int n = 0; n < cover->number_POVs[i]; n++)
                cover->POVs[i][n] = 0;
        }
        else
            cover->POVs[i] = NULL;

        if (coVRMSController::instance()->isMaster())
        {
            coVRMSController::instance()->sendSlaves((char *)&cover->number_axes[i], sizeof(int));
            if (cover->number_axes[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->axes[i][0]), cover->number_axes[i] * sizeof(float));
            }
            coVRMSController::instance()->sendSlaves((char *)&cover->number_buttons[i], sizeof(int));
            if (cover->number_buttons[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->buttons[i][0]), cover->number_buttons[i] * sizeof(int));
            }
            coVRMSController::instance()->sendSlaves((char *)&cover->number_sliders[i], sizeof(int));
            if (cover->number_sliders[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->sliders[i][0]), cover->number_sliders[i] * sizeof(float));
            }
            coVRMSController::instance()->sendSlaves((char *)&cover->number_POVs[i], sizeof(int));
            if (cover->number_POVs[i])
            {
                coVRMSController::instance()->sendSlaves((char *)&(cover->POVs[i][0]), cover->number_POVs[i] * sizeof(float));
            }
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)&cover->number_axes[i], sizeof(int));
            if (cover->number_axes[i])
            {
                if (cover->axes[i] == NULL)
                {
                    cover->axes[i] = new float[cover->number_axes[i]];
                    for (int n = 0; n < cover->number_axes[i]; n++)
                        cover->axes[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->axes[i][0]), cover->number_axes[i] * sizeof(float));
            }
            coVRMSController::instance()->readMaster((char *)&cover->number_buttons[i], sizeof(int));

            if (cover->number_buttons[i])
            {
                if (cover->buttons[i] == NULL)
                {
                    cover->buttons[i] = new int[cover->number_buttons[i]];
                    for (int n = 0; n < cover->number_buttons[i]; n++)
                        cover->buttons[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->buttons[i][0]), cover->number_buttons[i] * sizeof(int));
            }
            coVRMSController::instance()->readMaster((char *)&cover->number_sliders[i], sizeof(int));

            if (cover->number_sliders[i])
            {
                if (cover->sliders[i] == NULL)
                {
                    cover->sliders[i] = new float[cover->number_sliders[i]];
                    for (int n = 0; n < cover->number_sliders[i]; n++)
                        cover->sliders[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->sliders[i][0]), cover->number_sliders[i] * sizeof(int));
            }
            coVRMSController::instance()->readMaster((char *)&cover->number_POVs[i], sizeof(int));

            if (cover->number_POVs[i])
            {
                if (cover->POVs[i] == NULL)
                {
                    cover->POVs[i] = new float[cover->number_POVs[i]];
                    for (int n = 0; n < cover->number_POVs[i]; n++)
                        cover->POVs[i][n] = 0;
                }
                coVRMSController::instance()->readMaster((char *)&(cover->POVs[i][0]), cover->number_POVs[i] * sizeof(int));
            }
        }
    }
    if (haveMouse)
    {
        int i = numLocalJoysticks;
        cover->number_buttons[i] = 4;
        cover->number_sliders[i] = 0;
        cover->number_axes[i] = 1;
        cover->number_POVs[i] = 0;
        cover->buttons[i] = new int[3];
        cover->sliders[i] = NULL;
        cover->axes[i] = new float[1];
        cover->POVs[i] = NULL;
        fd[i] = -1;
    }
    return 1;
}

// this function is called if a message arrives
void SteeringWheelPlugin::message(int toWhom, int type, int length, const void *data)
{
    if (type == PluginMessageTypes::HLRS_SteeringWheelRemoteVehiclePosition)
    {
        osg::Matrix vehicleMat;
        TokenBuffer tb((const char *)data, length);
        int RemoteID;
        tb >> RemoteID;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                tb >> vehicleMat(i, j);
        if (VrmlNodeRemoteVehicle::instance())
            VrmlNodeRemoteVehicle::instance()->setVRMLVehicle(vehicleMat);
    }
}

void
SteeringWheelPlugin::preFrame()
{
    if (dataController)
        dataController->update();

    fasiUpdateManager::instance()->update();

    if (sitzkiste)
        sitzkiste->update();
    //if(vd)
    // vd->update();
    if (dynamics)
    {
        dynamics->update();
        if (VrmlNodeRemoteVehicle::instance())
        {
            TokenBuffer tb;
            osg::Matrix vehicleMat = dynamics->getVehicleTransformation();
            tb << coVRCommunication::instance()->getID();
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    tb << vehicleMat(i, j);

            cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS,
                               PluginMessageTypes::HLRS_SteeringWheelRemoteVehiclePosition,
                               tb.getData().length(), tb.getData().data());
        }
    }
    UpdateInputState();

    //std::cout << "Data controller: " << dataController << std::endl;
    if (dataController)
    {
        //std::cout << "Trying dataController..." << std::endl;
        if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
        {
            //   std::cout << "Trying serverConn..." << std::endl;
            toClientConn = serverConn->spawnSimpleConnection();
            if (toClientConn && toClientConn->is_connected())
            {
                //int id=2;
                //int ret =toClientConn->getSocket()->read(&id,sizeof(id));
                //if(ret < sizeof(id))
                //{
                //   fprintf(stderr,"reading Porsche id failed\n");
                //}
                //if(id==1)
                //{
                if (simulatorJoystick == -1)
                {
                    simulatorJoystick = cover->numJoysticks++;
                }
                fprintf(stderr, "Connected to RealtimeSimulatorHardware, storing Data as Joystick  %d\n", simulatorJoystick);
                cover->number_buttons[simulatorJoystick] = 3;
                cover->number_sliders[simulatorJoystick] = 0;
                cover->number_axes[simulatorJoystick] = 50;
                cover->number_POVs[simulatorJoystick] = 0;
                cover->buttons[simulatorJoystick] = new int[3];
                cover->sliders[simulatorJoystick] = NULL;
                cover->axes[simulatorJoystick] = new float[50];
                cover->POVs[simulatorJoystick] = NULL;
                fd[simulatorJoystick] = -1;
                //}
            }
        }

        if (toClientConn)
        {
            if (coVRMSController::instance()->isMaster())
            {
                if ((cover->frameTime() - oldTime) > updateRate)
                {
                    //sendValuesToClient();
                    oldTime = cover->frameTime();
                }
            }
            while (toClientConn && toClientConn->check_for_input())
            {
                int newNumFloats = 6; // should read these numbers from the server!!
                int newNumInts = 1; // should read these numbers from the server!!
                if (!readClientVal(&newNumFloats, sizeof(int)))
                {
                    toClientConn.reset(nullptr);
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cout << "Creset " << newNumInts << endl;
                }
                if (!readClientVal(&newNumInts, sizeof(int)))
                {
                    toClientConn.reset(nullptr);
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cout << "Creseti " << newNumInts << endl;
                }
                if (newNumFloats > 0 && newNumFloats != numFloats)
                {
                    numFloats = (int)newNumFloats;
                    delete[] floatValues;
                    floatValues = new float[numFloats];
                }
                if (newNumInts > 0 && newNumInts != numInts)
                {
                    numInts = (int)newNumInts;
                    delete[] intValues;
                    intValues = new int[numInts];
                }
                if (!readClientVal(floatValues, numFloats * sizeof(float)))
                {
                    toClientConn.reset(nullptr);
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cout << "Creseti2 " << newNumInts << endl;
                }
                if (!readClientVal(intValues, numInts * sizeof(int)))
                {
                    toClientConn.reset(nullptr);
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cout << "Creseti2 " << newNumInts << endl;
                }
                int i;

                for (i = 0; i < numFloats; i++)
                {
                    if (i >= cover->number_axes[simulatorJoystick])
                    {
                        cout << endl << "more floats than axes " << endl;
                        cout << endl << "simulatorJoystick " << endl;
                        cout << endl << "numFloats " << endl;
                    }
                    else
                    {
                        cover->axes[simulatorJoystick][i] = floatValues[i];
                    }
                }
                for (i = 0; i < (int)(cover->number_buttons[simulatorJoystick]); i++)
                {
                    if (intValues[0] & (1 << i))
                        cover->buttons[simulatorJoystick][i] = 1;
                    else
                        cover->buttons[simulatorJoystick][i] = 0;
                }
                cout << "Creceived numFloats: " << numFloats << " numInts " << numInts << endl;
                cout << "Creceived: ";
                for (i = 0; i < numFloats; i++)
                {
                    cout << floatValues[i] << " ";
                }
                cout << endl;

                for (i = 0; i < numInts; i++)
                {
                    cout << intValues[i] << " ";
                }
                cout << endl;
            }
        }
    }

    /*
   if(conn)
   {
      if(coVRMSController::instance()->isMaster())
      {
         if((cover->frameTime() - oldTime)>updateRate)
         {
            sendValues();
            oldTime = cover->frameTime();
         }
      }
      while(conn && conn->check_for_input())
      {
         int newNumFloats=6; // should read these numbers from the server!!
         int newNumInts=1; // should read these numbers from the server!!
         if(!readVal(&newNumFloats,sizeof(int)))
         {
            delete conn;
            conn = NULL;
            newNumFloats = 0;
            newNumInts = 0;
            numFloats = 0;
            numInts = 0;
            cout << "reset " << newNumInts << endl;
         }
         if(!readVal(&newNumInts,sizeof(int)))
         {
            delete conn;
            conn = NULL;
            newNumFloats = 0;
            newNumInts = 0;
            numFloats = 0;
            numInts = 0;
            cout << "reseti " << newNumInts << endl;
         }
         if(newNumFloats > 0 && newNumFloats != numFloats)
         {
            numFloats=(int)newNumFloats;
            delete[] floatValues;
            floatValues = new float[numFloats];
         }
         if(newNumInts > 0 && newNumInts != numInts)
         {
            numInts=(int)newNumInts;
            delete[] intValues;
            intValues = new int[numInts];
         }
         if(!readVal(floatValues,numFloats*sizeof(float)))
         {
            delete conn;
            conn = NULL;
            newNumFloats = 0;
            newNumInts = 0;
            numFloats = 0;
            numInts = 0;
            cout << "reseti2 " << newNumInts << endl;
         }
         if(!readVal(intValues,numInts*sizeof(int)))
         {
            delete conn;
            conn = NULL;
            newNumFloats = 0;
            newNumInts = 0;
            numFloats = 0;
            numInts = 0;
            cout << "reseti2 " << newNumInts << endl;
         }
         int i;

         for(i=0;i<numFloats;i++)
         {
                if(i>=cover->number_axes[simulatorJoystick])
                {
                   cout << endl << "more floats than axes " << endl;
                   cout << endl << "simulatorJoystick " << endl;
                   cout << endl << "numFloats " << endl;
                }
                else
                {
                   cover->axes[simulatorJoystick][i]=floatValues[i];
                }
         }
         for(i=0;i<(int)(cover->number_buttons[simulatorJoystick]);i++)
         {
            if(intValues[0] & (1 << i))
               cover->buttons[simulatorJoystick][i]=1;
            else
               cover->buttons[simulatorJoystick][i]=0;
         }
         cout << "received numFloats: " << numFloats << endl;
         for(i=0;i<numFloats;i++)
         {
            cout << floatValues[i] << " ";
         }
         cout << endl;
         cout << "received numInts: " << numInts << endl;

         for(i=0;i<numInts;i++)
         {
            cout << intValues[i] << " ";
         }
         cout << endl;
      }
   }
   else if((coVRMSController::instance()->isMaster()) && (serverHost!=NULL))
   {
      // try to connect to server every 2 secnods
      if((cover->frameTime() - oldTime)>2)
      {
         conn = new SimpleClientConnection(serverHost,port,0);

         if(!conn->is_connected())                // could not open server port
         {
#ifndef _WIN32
            if(errno!=ECONNREFUSED)
            {
               fprintf(stderr,"Could not connect to Porsche on %s; port %d\n",serverHost->getName(),port);
               delete serverHost;
               serverHost = NULL;
            }
#endif
            delete conn;
            conn=NULL;
            conn = new SimpleClientConnection(localHost,port,0);
            if(!conn->is_connected())             // could not open server port
            {
#ifndef _WIN32
               if(errno!=ECONNREFUSED)
               {
                  fprintf(stderr,"Could not connect to Porsche on %s; port %d\n",localHost->getName(),port);
               }
#endif
               delete conn;
               conn=NULL;

            }
            else
            {
               fprintf(stderr,"Connected to Porsche on %s; port %d\n",localHost->getName(),port);
            }
         }
         else
         {
            fprintf(stderr,"Connected to Porsche on %s; port %d\n",serverHost->getName(),port);
         }
         if(conn && conn->is_connected())
         {
            int id=2;
            int ret =conn->getSocket()->read(&id,sizeof(id));
            if(ret < sizeof(id))
            {
               fprintf(stderr,"reading Porsche id failed\n");
            }
            if(id==1)
            {
               if(simulatorJoystick == -1)
               {
               simulatorJoystick = cover->numJoysticks++;
               }
               fprintf(stderr,"Connected to SimulatorHardware, storing Data as Joystick  %d\n",simulatorJoystick);
               cover->number_buttons[simulatorJoystick]=3;
               cover->number_sliders[simulatorJoystick]=0;
               cover->number_axes[simulatorJoystick]=3;
               cover->number_POVs[simulatorJoystick]=0;
               cover->buttons[simulatorJoystick]=new int[3];
               cover->sliders[simulatorJoystick]=NULL;
               cover->axes[simulatorJoystick]=new float[3];
               cover->POVs[simulatorJoystick]=NULL;
               fd[simulatorJoystick]=-1;
            }
         }
         oldTime = cover->frameTime();
      }
   }
   */
    /*
   if(coVRMSController::instance()->isMaster())
   {
      //cout << numFloats << endl;
      //cout << numInts << endl;
      coVRMSController::instance()->sendSlaves((char *)&numFloats,sizeof(int));
      coVRMSController::instance()->sendSlaves((char *)&numInts,sizeof(int));
      if(numFloats)
         coVRMSController::instance()->sendSlaves((char *)floatValues,numFloats*sizeof(float));
      if(numInts)
         coVRMSController::instance()->sendSlaves((char *)intValues,numInts*sizeof(int));
   }
   else
   {
      int newNumFloats=0;
      int newNumInts=0;
      coVRMSController::instance()->readMaster((char *)&newNumFloats,sizeof(int));
      coVRMSController::instance()->readMaster((char *)&newNumInts,sizeof(int));
      //cout << newNumFloats << endl;
      //cout << newNumInts << endl;
               if(newNumFloats == 3 && (simulatorJoystick == -1))
               {
                   simulatorJoystick = cover->numJoysticks++;
               fprintf(stderr,"Connected to SimulatorHardware, storing Data as Joystick  %d\n",simulatorJoystick);
               cover->number_buttons[simulatorJoystick]=3;
               cover->number_sliders[simulatorJoystick]=0;
               cover->number_axes[simulatorJoystick]=3;
               cover->number_POVs[simulatorJoystick]=0;
               cover->buttons[simulatorJoystick]=new int[3];
               cover->sliders[simulatorJoystick]=NULL;
               cover->axes[simulatorJoystick]=new float[3];
               cover->POVs[simulatorJoystick]=NULL;
               fd[simulatorJoystick]=-1;
               }
      if(newNumFloats>0 && newNumFloats != numFloats)
      {
         cout << "resize" << endl;
         numFloats=newNumFloats;
         delete[] floatValues;
         floatValues = new float[numFloats];
      }
      if(newNumInts > 0 && newNumInts != numInts)
      {
         cout << "resize" << endl;
         numInts=newNumInts;
         delete[] intValues;
         intValues = new int[numInts];
      }
      if(newNumFloats>0 && numFloats)
      {
         //cout << "rf" << endl;
         coVRMSController::instance()->readMaster((char *)floatValues,numFloats*sizeof(float));
      }
      if(newNumFloats>0 && numInts)
      {
         //cout << "ri" << endl;
         coVRMSController::instance()->readMaster((char *)intValues,numInts*sizeof(int));
      }
      if(newNumFloats>0 && numFloats && simulatorJoystick != -1)
      {
         for(int i=0;i<numFloats;i++)
         {
                if(i>=cover->number_axes[simulatorJoystick])
                {
                   cout << endl << "more floats than axes " << endl;
                   cout << endl << "simulatorJoystick " << endl;
                   cout << endl << "numFloats " << endl;
                }
                else
                {
                   cover->axes[simulatorJoystick][i]=floatValues[i];
                }
         }
         for(int i=0;i<(int)(cover->number_buttons[simulatorJoystick]);i++)
         {
            if(intValues[0] & (1 << i))
               cover->buttons[simulatorJoystick][i]=1;
            else
               cover->buttons[simulatorJoystick][i]=0;
         }
         cout << "Creceived numFloats: " << numFloats << " numInts " << numInts<< endl;
         cout << "Creceived: ";
         for(int i=0;i<numFloats;i++)
         {
            cout << floatValues[i] << " ";
         }
         cout << endl;

         for(int i=0;i<numInts;i++)
         {
            cout << intValues[i] << " ";
         }
         cout << endl;
}
   }
   */
    /*if(coVRMSController::instance()->isMaster())
   {
      coVRMSController::instance()->sendSlaves((char *)&simulatorJoystick,sizeof(int));
      if(simulatorJoystick != -1)
      {
          coVRMSController::instance()->sendSlaves((char *)cover->buttons[simulatorJoystick],3*sizeof(int));
          coVRMSController::instance()->sendSlaves((char *)cover->axes[simulatorJoystick],50*sizeof(float));
      }
   }
   else
   {
      coVRMSController::instance()->readMaster((char *)&simulatorJoystick,sizeof(int));

               if(simulatorJoystick != oldSimulatorJoystick)
               {
                   cover->numJoysticks++;
                   oldSimulatorJoystick=simulatorJoystick;
               fprintf(stderr,"Connected to RealtimeSimulatorHardware, storing Data as Joystick  %d\n",simulatorJoystick);
               cover->number_buttons[simulatorJoystick]=3;
               cover->number_sliders[simulatorJoystick]=0;
               cover->number_axes[simulatorJoystick]=50;
               cover->number_POVs[simulatorJoystick]=0;
               cover->buttons[simulatorJoystick]=new int[3];
               cover->sliders[simulatorJoystick]=NULL;
               cover->axes[simulatorJoystick]=new float[50];
               cover->POVs[simulatorJoystick]=NULL;
               fd[simulatorJoystick]=-1;
               }
      if(simulatorJoystick != -1)
      {
          coVRMSController::instance()->readMaster((char *)cover->buttons[simulatorJoystick],3*sizeof(int));
          coVRMSController::instance()->readMaster((char *)cover->axes[simulatorJoystick],50*sizeof(float));
      }
   }*/

    /*  if(porsche->deviceOpen())
   {
   while(porsche->bufferempty==false)
   {
   if(porsche->readBlock())
   {
   int i;
   for(i=0;i<porsche->numFloats;i++)
   {
   cout << i << ": " << porsche->floats[i] << " ";
   }
   cout << (int)porsche->numFloats << endl;
   }
   else
   {
   cout << "oops" << endl;
   }
   }
   porsche->bufferempty = false;
   }*/
}

void SteeringWheelPlugin::key(int type, int keySym, int /*mod*/)
{

    Keyboard *keyb = dynamic_cast<Keyboard *>(SteeringWheelPlugin::plugin->sitzkiste);
    if (keyb != NULL)
    {
        //if(mod) {} //Otherwise warning: "mod not used" -> with compiler flag: "Treat warnings as errors" -> Compiler-TERROR!!!
        cerr << "keySym " << keySym << endl;
        if (type == osgGA::GUIEventAdapter::KEYDOWN)
        {
            switch (keySym)
            {
            case 65361:
                keyb->leftKeyDown();
                break;
            case 65363:
                keyb->rightKeyDown();
                break;
            case 65362:
                keyb->foreKeyDown();
                break;
            case 65364:
                keyb->backKeyDown();
                break;
            case 103:
                keyb->gearShiftUpKeyDown();
                break;
            case 102:
                keyb->gearShiftDownKeyDown();
                break;
            case 104:
                keyb->hornKeyDown();
                break;
            case 114:
                keyb->resetKeyDown();
                break;
            case 101:
                UDPComm::errorStatus_SW();
                if (coVRMSController::instance()->isMaster())
                {
                    if (UDPComm::getError_SW() == true)
                        cout << "'e' pressed (Steering Wheel) | UDP-Errors: ON" << endl;
                    else
                        cout << "'e' pressed (Steering Wheel) | UDP-Errors: OFF" << endl;
                }
                break;
            case 107:
                if ((Carpool::Instance()->getPoolVector()).size() > 0)
                {
                    std::cout << endl << "Show actual position"
                              << " w: " << RoadSystem::Instance()->current_tile_x << " h: " << RoadSystem::Instance()->current_tile_y << std::endl;
                    // system = RoadSystem::Instance();

                    if ((RoadSystem::_tiles_y >= 200) || (RoadSystem::_tiles_x >= 200))
                    {
                        // Für eine bessere Übersicht wird bei großen Straßennetzen nur ein Auschnitt um das Eigenfahrzeug herum dargestellt
                        int x_min = RoadSystem::Instance()->current_tile_x - 100;
                        if (x_min < 0)
                            x_min = 0;
                        int x_max = RoadSystem::Instance()->current_tile_x + 100;
                        if (x_max > RoadSystem::_tiles_x)
                            x_max = RoadSystem::_tiles_x;

                        int y_min = RoadSystem::Instance()->current_tile_y - 100;
                        if (y_min < 0)
                            y_min = 0;
                        int y_max = RoadSystem::Instance()->current_tile_y + 100;
                        if (y_max > RoadSystem::_tiles_y)
                            y_max = RoadSystem::_tiles_y;

                        std::cout << "Strassennetz zu gross - aktueller Ausschnitt: x: " << x_min << "-" << x_max << " ; y: " << y_min << "-" << y_max << std::endl;
                        //for (int i=y_min; i<y_max;i++) {
                        for (int i = y_max; i >= y_min; i--)
                        {
                            for (int j = x_min; j <= x_max; j++)
                            {
                                if (RoadSystem::Instance()->current_tile_x == j && RoadSystem::Instance()->current_tile_y == i)
                                    std::cout << ".";
                                //std::cout << "TEST3" << std::endl;
                                else if ((RoadSystem::Instance()->getRLS_List(j, i)).size() == 0)
                                    std::cout << "-";
                                //std::cout << "TEST4" << std::endl;
                                else
                                    std::cout << (RoadSystem::Instance()->getRLS_List(j, i)).size();
                            }
                            std::cout << std::endl;
                        }
                    }
                    else
                    {
                        /*for (int i=0; i<RoadSystem::_tiles_y+1;i++) {
							for (int j=0; j<RoadSystem::_tiles_x+1;j++) {
								if (system->current_tile_x == j && system->current_tile_y == i) std::cout << ".";
								else if ((system->getRLS_List(j,i)).size() == 0) std::cout << "-";
								else std::cout << (system->getRLS_List(j,i)).size();
							}
							std::cout << std::endl;
						}*/
                        for (int i = RoadSystem::_tiles_y; i >= 0; i--)
                        {
                            for (int j = 0; j <= RoadSystem::_tiles_x; j++)
                            {
                                if (RoadSystem::Instance()->current_tile_x == j && RoadSystem::Instance()->current_tile_y == i)
                                    std::cout << ".";
                                else if ((RoadSystem::Instance()->getRLS_List(j, i)).size() == 0)
                                    std::cout << "-";
                                else
                                    std::cout << (RoadSystem::Instance()->getRLS_List(j, i)).size();
                            }
                            std::cout << std::endl;
                        }
                    }
                }
                break;
            }
        }
        else if (type == osgGA::GUIEventAdapter::KEYUP)
        {
            std::cout << "KEYUP EVENT!!!" << std::endl;
            switch (keySym)
            {
            case 65361:
                keyb->leftKeyUp();
                break;
            case 65363:
                keyb->rightKeyUp();
                break;
            case 65362:
                keyb->foreKeyUp();
                break;
            case 65364:
                keyb->backKeyUp();
                break;
            case 103:
                keyb->gearShiftUpKeyUp();
                break;
            case 102:
                keyb->gearShiftDownKeyUp();
                break;
            case 104:
                keyb->hornKeyUp();
                break;
            case 114:
                keyb->resetKeyUp();
                break;
            }
        }
    }
}

COVERPLUGIN(SteeringWheelPlugin)

/*
// C plugin interface, don't do any coding down here, do it in the C++ Class!

int coVRInit(coVRPlugin *m)
{
   (void) m;
   SteeringWheelPlugin::plugin = new SteeringWheelPlugin();
   return SteeringWheelPlugin::plugin->init();
}


void coVRDelete(coVRPlugin *m)
{
   (void) m;
   delete SteeringWheelPlugin::plugin;
}


void coVRPreFrame()
{
   SteeringWheelPlugin::plugin->preFrame();
}

void coVRKey(int type,int keySym,int mod)
{
   //std::cout << "KEYEVENT --- Type: " << type << ", keySym: " << keySym << ", mod: " << mod << std::endl;
   SteeringWheelPlugin::plugin->keyEvent(type,keySym,mod);
}
*/
