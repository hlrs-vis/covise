/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\ 
**                                                            (C)2008 HLRS  **
**                                                                          **
** Description: Input Plugin for SpaceNavigator/SpaceMouse/Joysticks        **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                             **
**                                                                          **
** History:  			                                             **
** June-08  v1	    				                             **
**                                                                          **
**                                                                          **
\****************************************************************************/

/* for serial SpaceMouse:
   - run e.g. 'inputattach --magellan /dev/ttyUSB0&'
   - point COVER.Input.SpaceNavigator to the created event device */

#include "SpaceNavigatorDriver.h"
#include <config/CoviseConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/input/input.h>
#include <cover/coVRPluginSupport.h>
#include <algorithm>

#ifdef USE_HIDAPI
#include <hidapi.h>

// from Virtual Reality Peripheral Network - vrpn_3DConnexion.C
// and https://www.3dconnexion.com/forum/viewtopic.php?t=1610&start=15
std::vector<int> hidapi_vendors{
    0x046d, // 1133 // Logitech (3Dconnexion is made by Logitech)
    0x256f, // 9583 // 3Dconnexion
};

std::vector<int> hidapi_products{
    0xC623, // 50723 // SpaceTraveler
    0xC626, // 50726 // SpaceNavigator
    0xc628, // 50728 // SpaceNavigator for Notebooks
    0xc627, // 50727 // SpaceExplorer
    0xC603, // 50691 // SpaceMouse
    0xC62B, // 50731 // SpaceMouse Pro
    0xc621, // 50721 // Spaceball 5000
    0xc625, // 50725 // SpacePilot
    0xc629, // 50729 // SpacePilot Pro
};

enum HidEventType
{
    Translation = 1,
    Rotation    = 2,
    Button      = 3
};
#endif

const int MaxButtons = 32;
const int NumValuators = 6;
const int SpaceNavZero = 5;

using namespace opencover;
using covise::coCoviseConfig;

// see <linux/input.h>
struct InputEvent
{
    struct timeval time;
    uint16_t type;
    uint16_t code;
    int32_t value;
};

enum InputEventType
{
    EV_SYN = 0,
    EV_KEY = 1,
    EV_REL = 2,
    EV_ABS = 3,
    EV_MSC = 4,
};

bool SpaceNavigatorDriver::hidapi_init()
{
#ifndef USE_HIDAPI
    return false;
#else
    unsigned short current_vendor_id = 0x0;
    unsigned short current_product_id = 0x0;

    struct hid_device_info *devs = hid_enumerate(0x0, 0x0);

    for (struct hid_device_info *cur_dev = devs;
            cur_dev;
            cur_dev = cur_dev->next)
    {
        if (std::find(hidapi_vendors.begin(), hidapi_vendors.end(), cur_dev->vendor_id) != hidapi_vendors.end()
                && std::find(hidapi_products.begin(), hidapi_products.end(), cur_dev->product_id) != hidapi_products.end())
        {
            //setup the values with the founded vendor id.s
            current_vendor_id = cur_dev->vendor_id;
            current_product_id = cur_dev->product_id;

            break; //break when first comaptaible device is found.
        }
    }
    hid_free_enumeration(devs);

    if (current_vendor_id == 0x0 || current_product_id == 0x0)
    {
        std::cerr << "SpaceNavigatorDriver hidapi: no compatible SpaceMouse device found\n";
        return false;
    }

    //open device with current_vendor_id and current_product_id
    errno = 0;
    m_hidapiHandle = hid_open(current_vendor_id, current_product_id, NULL);
    if (!m_hidapiHandle)
    {
        std::cerr << "SpaceNavigatorDriver hidapi: Unable to open SpaceMouse: ";
        if (errno)
            std::cerr << errno << " " << strerror(errno) << ": ";
        std::cerr << std::hex << "Manufacturer: 0x"<< current_vendor_id << ",  Product: 0x" << current_product_id << std::dec << "\n";
        return false;
    }

    std::cerr << "SpaceNavigatorDriver hidapi: using SpaceMouse: ";
    std::cerr << std::hex << "Manufacturer: 0x"<< current_vendor_id << ",  Product: 0x" << current_product_id << std::dec << "\n";

    m_spacemouse = false;

    return true;
#endif
}

void SpaceNavigatorDriver::hidapi_finish()
{
#ifdef USE_HIDAPI
    //only try closing when the handle is open.
    if (!m_hidapi)
        return;

    hid_close(m_hidapiHandle);
    m_hidapiHandle = nullptr;
    hid_exit();

    m_hidapi = false;
#endif
}

bool SpaceNavigatorDriver::hidapi_recalibrate()
{
#ifndef USE_HIDAPI
    return false;
#else
    if (!m_hidapi)
        return false;

    unsigned char data[] = {0x07, 0x00}; // This proprietary(?) feature report will rezero the device.
    int res = hid_send_feature_report(m_hidapiHandle, data, sizeof(data));
    return res == sizeof(data);
#endif
}

bool SpaceNavigatorDriver::hidapi_poll()
{
#ifndef USE_HIDAPI
    return false;
#else
    unsigned char data[7] = { 0 };

    int result = hid_read_timeout(m_hidapiHandle, data, sizeof(data), 400); //when timeout
    if (result < 0)
    {
        std::cout << "SpaceNavigatorDriver hidapi: Unable to read(), stopping thread\n";
        return false;
    }
    if (result == 0)
    {
        return true;
    }

    auto value = [](int v0, int v1){
        int16_t v = (v0 & 0xff) | ((v1 & 0xff)<<8);
        return v;
    };

    switch (data[0])
    {
    case Translation:
    {
        for (int i=0; i<3; ++i)
            m_raw[i] = value(data[1+i*2], data[2+i*2]);
        m_mutex.lock();
        for (int i=0; i<3; ++i)
            processRaw(i, m_raw[i]);
        m_mutex.unlock();
        break;
    }
    case Rotation:
    {
        for (int i=0; i<3; ++i)
            m_raw[i+3] = value(data[1+i*2], data[2+i*2]);
        m_mutex.lock();
        for (int i=0; i<3; ++i)
            processRaw(i+3, m_raw[i+3]);
        m_mutex.unlock();
        break;
    }
    case Button:
    {
        m_mutex.lock();
        int b=1;
        for (int i=0; i<8; ++i)
        {
            if (data[1] & b)
                m_buttonStates[i+6] = true;
            else
                m_buttonStates[i+6] = false;
            b <<= 1;
        }
        m_mutex.unlock();
        break;
    }
    default:
        break;
    }

    return true;
#endif
}


SpaceNavigatorDriver::SpaceNavigatorDriver(const std::string &config)
: InputDevice(config)
{
    m_valuatorValues.resize(NumValuators);
    m_valuatorRanges.resize(NumValuators);
    m_buttonStates.resize(MaxButtons);

    for (int i = 0; i < NumValuators; i++)
    {
        m_valuatorValues[i] = 0.;
        m_valuatorRanges[i].first = -4.;
        m_valuatorRanges[i].second = 4.;
    }

    m_raw.resize(NumValuators);

    bool tryHidapi = coCoviseConfig::isOn("hidapi", configPath(), true);
    if (tryHidapi && hidapi_init())
    {
        m_hidapi = true;
        m_valid = true;
        return;
    }

    m_hidapi = false;

    std::vector<std::string> tried;
    std::string deviceFile = coCoviseConfig::getEntry("device", configPath(), "");
    int flags = O_RDONLY | O_NONBLOCK;
    if (!deviceFile.empty())
    {
        tried.push_back(deviceFile);
        m_fd = open(deviceFile.c_str(), flags);
    }
    else
    {
        deviceFile = "/dev/input/spacemouse";
        tried.push_back(deviceFile);
        m_spacemouse = true;
        m_fd = open(deviceFile.c_str(), flags);
        if (m_fd < 0)
        {
            deviceFile = "/dev/input/spacenavigator";
            tried.push_back(deviceFile);
            m_spacemouse = false;
            m_fd = open(deviceFile.c_str(), flags);
        }
    }
    if (m_fd < 0)
    {
        std::cerr << "Input: SpaceNavigator: failed to open device, tried:";
        for (auto t: tried)
            std::cerr << " " << t;
        std::cerr << std::endl;
        return;
    }

    if (Input::instance()->debug(Input::Driver) && Input::instance()->debug(Input::Config))
    {
        std::cerr << "Input: SpaceNavigator: opened " << configPath() << ": " << deviceFile << std::endl;
    }
#ifndef WIN32
    struct stat statbuf;
    statbuf.st_blksize = 4096;
    fstat(m_fd, &statbuf);
    int bufSize = statbuf.st_blksize;
#else
    int bufSize = 4096;
#endif
    m_buf.resize(bufSize);

    m_valid = true;
}

SpaceNavigatorDriver::~SpaceNavigatorDriver()
{
    if (m_hidapi)
    {
        hidapi_finish();
        return;
    }

    if (m_fd >= 0)
    {
        close(m_fd);
    }
}

void SpaceNavigatorDriver::processRaw(int axis, int value)
{
    double fvalue = 0.;
    if (abs(value) < SpaceNavZero)
        value = 0;
    else if (value < 0)
        value += SpaceNavZero;
    else if (value > 0)
        value -= SpaceNavZero;
    if (value > 400)
        value = 400;
    if (value < -400)
        value = -400;

    fvalue = value / 400.;

    if (axis < NumValuators)
        m_valuatorValues[axis] = fvalue;
    if (Input::instance()->debug(Input::Driver) && Input::instance()->debug(Input::Valuators))
    {
        std::cerr << "Input: SpaceNavigator: axis " << axis << ": " << fvalue << std::endl;
    }
}
bool SpaceNavigatorDriver::poll()
{
    if (m_hidapi)
    {
        return hidapi_poll();
    }

#ifdef WIN32
    MSG Message;
    while (::PeekMessage(&Message, hWnd, NULL, NULL, PM_REMOVE))
    {
        //if ( TranslateAccelerator(MainhWnd, hAccel, &Message) )
        //continue;

        TranslateMessage(&Message);
        DispatchMessage(&Message);
    };

    smd.tx = 0.0;
    smd.ty = 0.0;
    smd.tz = 0.0;
    smd.h = 0.0;
    smd.p = 0.0;
    smd.r = 0.0;
    if (g3DKeyboard)
    {
        // Check if any change to the keyboard state
        try
        {
            long nKeys;
            nKeys = g3DKeyboard->Keys;
            long i;
            for (i = 1; i <= nKeys; i++)
            {
                __int64 mask = (__int64)1 << (i - 1);
                VARIANT_BOOL isPressed;
                isPressed = g3DKeyboard->IsKeyDown(i);
                if (isPressed == VARIANT_TRUE)
                {
                    smd.buttonStatus |= mask;
                }
                else
                {
                    smd.buttonStatus &= ~mask;
                }
            }
            // Test the special keys
            for (i = 30; i <= 31; i++)
            {
                __int64 mask = (__int64)1 << (i - 1);
                VARIANT_BOOL isPressed;
                isPressed = g3DKeyboard->IsKeyDown(i);
                if (isPressed == VARIANT_TRUE)
                {
                    smd.buttonStatus |= mask;
                }
                else
                {
                    smd.buttonStatus &= ~mask;
                }
            }
        }
        catch (...)
        {
            // Some sort of exception handling
        }
    }
    if (g3DSensor)
    {
        try
        {

            CComPtr<IAngleAxis> pRotation = g3DSensor->Rotation;
            CComPtr<IVector3D> pTranslation = g3DSensor->Translation;

            // Check if the cap is still displaced
            if (pRotation->Angle > 0. || pTranslation->Length > 0.)
            {
                spacedMouseEvent(pTranslation->X, pTranslation->Y, pTranslation->Z, pRotation->X, pRotation->Y, pRotation->Z, pRotation->Angle);
            }

            pRotation.Release();
            pTranslation.Release();
        }
        catch (...)
        {
            // Some sort of exception handling
        }
    }
#else

    errno = 0;
    int numRead = read(m_fd, m_buf.data(), m_buf.size());
    if (numRead < 0)
    {
        if (m_evdevRel)
        {
            double t = coVRPluginSupport::currentTime();
            // no update seems to sent for zeroing EV_REL - clear it after some time
            if (t - m_lastUpdate > 0.05)
            {
                //std::cerr << "Input: SpaceNavigator: zeroing..." << std::endl;
                m_mutex.lock();
                for (int i=0; i<NumValuators; ++i)
                {
                    m_raw[i] = 0;
                    processRaw(i, m_raw[i]);
                }
                m_mutex.unlock();
                m_lastUpdate = t;
            }
        }

        if (errno == EAGAIN || errno == EINTR)
            return  true;

        return false;
    }

    m_lastUpdate = coVRPluginSupport::currentTime();

    int i = 0;
    while (i < numRead)
    {
        if (i+sizeof(InputEvent) > numRead)
        {
            break;
        }

        InputEvent *message = (InputEvent *)(m_buf.data() + i);
#if 0
        std::cerr << "Input: SpaceNavigator: event: type=" << message->type << ", code=" << message->code << ", value=" << message->value << std::endl;
#endif
        switch (message->type) {
        case EV_SYN: // data complete
        {
            m_mutex.lock();
            for (int axis=0; axis<NumValuators; ++axis)
                processRaw(axis, m_raw[axis]);
            m_mutex.unlock();
#if 0
            std::cerr << "Input: SpaceNavigator: SYN: ";
            for (int i=0; i<NumValuators; ++i)
                std::cerr << " " << m_raw[i];
            std::cerr << std::endl;
#endif
            break;
        }
        case EV_KEY: // button press
        {
            int buttonNumber = message->code % MaxButtons;
            if (!m_spacemouse && buttonNumber <= 1)
            {
                // make side buttons 6 & 7 also on SpaceNavigator
                buttonNumber += 6;
            }

            m_mutex.lock();
            m_buttonStates[buttonNumber] = message->value!=0;
            m_mutex.unlock();
            if (Input::instance()->debug(Input::Driver) && Input::instance()->debug(Input::Buttons))
            {
                std::cerr << "Input: SpaceNavigator: button " << buttonNumber << " " << (m_buttonStates[buttonNumber] ? "pressed" : "released") << std::endl;
            }
            break;
        }
        case EV_REL: // motion on SpaceNavigator
        {
            m_spacemouse = false;
            m_evdevRel = true;

            int axis = message->code;
            axis %= NumValuators;
            int value = message->value;
            m_raw[axis] = value;

            if (Input::instance()->debug(Input::Driver) && Input::instance()->debug(Input::Valuators))
            {
                std::cerr << "Input: SpaceNavigator: relative axis " << message->code << ": raw: " << message->value << std::endl;
            }
            break;
        }
        case EV_ABS: // motion on SpaceMouse
        {
            m_spacemouse = true;
            m_evdevRel = false;

            int value = message->value;
            int axis = message->code;
            axis %= NumValuators;

            // shuffle axis to match SpaceNavigator
            if (axis%3 == 2)
            {
                axis -= 1;
            }
            else if (axis%3 == 1)
            {
                axis += 1;
                value = -value;
            }

            m_raw[axis] = value;

            if (Input::instance()->debug(Input::Driver) && Input::instance()->debug(Input::Valuators))
            {
                std::cerr << "Input: SpaceNavigator: absolute axis " << message->code << ": raw: " << message->value << std::endl;
            }
            break;
        }
        case EV_MSC: {
            if (Input::instance()->debug(Input::Driver))
            {
                std::cerr << "Input: SpaceNavigator: EV_MSC: code=" << message->code << ", value=" << message->value << std::endl;
            }
            break;
        }
        default:
        {
            if (Input::instance()->debug(Input::Driver))
            {
                std::cerr << "Input: SpaceNavigator: unhandled event: type=" << message->type << ", code=" << message->code << ", value=" << message->value << std::endl;
            }
            break;
        }
        }
        i += sizeof(InputEvent);
    }
#endif

    return true;
}

INPUT_PLUGIN(SpaceNavigatorDriver)
