/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputhdw.cpp
 *
 *  Created on: Dec 9, 2014
 *      Author: svnvlad
 */
#include "Joystick.h"

#include <util/unixcompat.h>

#if !defined(_WIN32) && !defined(__APPLE__)
 //#define USE_XINPUT
#define USE_LINUX
#include <fcntl.h>
#include <linux/joystick.h> // das muss nach osg kommen, wegen KEY_F1
#include <unistd.h>
#include <sys/select.h>

#endif

#ifdef USE_XINPUT
#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <X11/cursorfont.h>
#endif

#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#include <iostream>

static OpenThreads::Mutex JoystickMutex; // Joystick is not thread-safe

using namespace std;

Joystick::Joystick(const std::string &config)
    : InputDevice(config)
{
    //cerr << "test " << numLocalJoysticks << endl;
    JoystickMutex.lock();
	numLocalJoysticks = 0;
	for (int i = 0; i < MAX_NUMBER_JOYSTICKS; i++)
	{
		number_buttons[i] = 0;
		number_sliders[i] = 0;
		number_axes[i] = 0;
		number_POVs[i] = 0;
		buttons[i] = NULL;
		sliders[i] = NULL;
		axes[i] = NULL;
		POVs[i] = NULL;
	}

	joystickNumber = covise::coCoviseConfig::getInt("joystickNumber", configPath(), 0);
#ifdef WIN32
	g_pDI = NULL;
	HRESULT hr;

	// Register with the DirectInput subsystem and get a pointer
	// to a IDirectInput interface we can use.
	// Create a DInput object
	if (FAILED(hr = DirectInput8Create(GetModuleHandle(NULL), DIRECTINPUT_VERSION,
		IID_IDirectInput8, (VOID **)&g_pDI, NULL)))
		return;

	// Look for a simple joystick we can use for this sample program.
	if (FAILED(hr = g_pDI->EnumDevices(DI8DEVCLASS_GAMECTRL,
		EnumJoysticksCallback,
		this, DIEDFL_ATTACHEDONLY)))
		return;

	for (int i = 0; i < numLocalJoysticks; i++)
	{
		if (buttons[i] == NULL)
		{
			buttons[i] = new int[number_buttons[i]];
			for (int n = 0; n < number_buttons[i]; n++)
				buttons[i][n] = 0;
		}
		if (sliders[i] == NULL)
		{
			sliders[i] = new float[number_sliders[i]];
			for (int n = 0; n < number_sliders[i]; n++)
				sliders[i][n] = 0;
		}

		if (axes[i] == NULL)
		{
			axes[i] = new float[number_axes[i]];
			for (int n = 0; n < number_axes[i]; n++)
				axes[i][n] = 0;
		}
		if (POVs[i] == NULL)
		{
			POVs[i] = new float[number_POVs[i]];
			for (int n = 0; n < number_POVs[i]; n++)
				POVs[i][n] = 0;
		}
	}
#else
#ifdef HAVE_NO_DEV_INPUT_JOYSTICK
        const char *devName = "/dev/js";
#else
        const char *devName = "/dev/input/js";
#endif
#define DEVICE_LEN 512
        string joystickDev = covise::coCoviseConfig::getEntry("device", configPath(), devName);
    int version=0;
        numLocalJoysticks=0;
	for (int deviceNumber  = 0; deviceNumber < MAX_NUMBER_JOYSTICKS; deviceNumber++)
	{
        std::string devName = joystickDev+to_string(deviceNumber);
        if ((fd[numLocalJoysticks] = open(devName.c_str(), O_RDONLY)) < 0)
        {
            if(numLocalJoysticks == 0)
            {
                cerr << "failed to opent " << devName << endl;
                perror("joystick initialisation failed: ");
	        if(deviceNumber > 4)
	        {
                    break;
		}
            }
	    else if((deviceNumber - numLocalJoysticks) > 4)
	    {
                break; //stop if we already found one
	    }
        }
	else
	{

        ioctl(fd[numLocalJoysticks], JSIOCGVERSION, &version);
        char joyname[128];
        std::string jName;
        if (ioctl(fd[numLocalJoysticks], JSIOCGNAME(128), joyname) < 0) {
            jName = "Unknown Joystick";
        } else {
            jName = joyname;
        }
        cerr << "found Joystick " << numLocalJoysticks << " name:" << jName << endl;
        names[numLocalJoysticks] = jName;
        ioctl(fd[numLocalJoysticks], JSIOCGAXES, &number_axes[numLocalJoysticks]);
        if (number_axes[numLocalJoysticks] > 0)
        {
            axes[numLocalJoysticks] = new float[number_axes[numLocalJoysticks]];
            for (int i = 0; i < number_axes[numLocalJoysticks]; i++)
                axes[numLocalJoysticks][i] = 0;
        }
        ioctl(fd[numLocalJoysticks], JSIOCGBUTTONS, &number_buttons[numLocalJoysticks]);
        if (number_buttons[numLocalJoysticks] > 0)
        {
            buttons[numLocalJoysticks] = new int[number_buttons[numLocalJoysticks]];
            for (int i = 0; i < number_buttons[numLocalJoysticks]; i++)
                buttons[numLocalJoysticks][i] = 0;
        }
	numLocalJoysticks++;
	}
     }
#endif
	if(numLocalJoysticks > joystickNumber)
	{
		m_valuatorRanges.resize(number_axes[joystickNumber]);
		m_valuatorValues.resize(number_axes[joystickNumber]);
		for (size_t i = 0; i < number_axes[joystickNumber]; i++)
		{
			m_valuatorValues[i] = axes[joystickNumber][i];
			m_valuatorRanges[i].first = -1.0;
			m_valuatorRanges[i].second = 1.0;
		}

		m_buttonStates.resize(number_buttons[joystickNumber]);
		for (size_t i = 0; i < number_buttons[joystickNumber]; i++)
		{
			m_buttonStates[i] = buttons[joystickNumber][i];
		}
	}
    cerr << "numLocalJoysticks " << numLocalJoysticks << endl;
    JoystickMutex.unlock();
}

//====================END of init section============================


Joystick::~Joystick()
{
	
    stopLoop();
    JoystickMutex.lock();
    JoystickMutex.unlock();
}

bool Joystick::needsThread() const 
{ 
return true;
}
#ifdef WIN32
//-----------------------------------------------------------------------------
// Name: EnumJoysticksCallback()
// Desc: Called once for each enumerated joystick. If we find one, create a
//       device interface on it so we can play with it.
//-----------------------------------------------------------------------------
BOOL CALLBACK Joystick::EnumJoysticksCallback(const DIDEVICEINSTANCE *pdidInstance,
	VOID *pContext)
{
	Joystick *joystick=(Joystick *)pContext;
	return joystick->EnumJoysticks(pdidInstance);
}

BOOL Joystick::EnumJoysticks(const DIDEVICEINSTANCE *pdidInstance)
{
	HRESULT hr;
	// Obtain an interface to the enumerated joystick.
	hr = g_pDI->CreateDevice(pdidInstance->guidInstance, &g_pJoystick[numLocalJoysticks], NULL);

	// If it failed, then we can't use this joystick. (Maybe the user unplugged
	// it while we were in the middle of enumerating it.)
	if (FAILED(hr))
		return DIENUM_CONTINUE;

	// Set the data format to "simple joystick" - a predefined data format
	//
	// A data format specifies which controls on a device we are interested in,
	// and how they should be reported. This tells DInput that we will be
	// passing a DIJOYSTATE2 structure to IDirectInputDevice::GetDeviceState().
	if (FAILED(hr = g_pJoystick[numLocalJoysticks]->SetDataFormat(&c_dfDIJoystick2)))
		return DIENUM_CONTINUE;

	// Set the cooperative level to let DInput know how this device should
	// interact with the system and with other DInput applications.
	if (FAILED(hr = g_pJoystick[numLocalJoysticks]->SetCooperativeLevel(window.handle_, DISCL_EXCLUSIVE | DISCL_BACKGROUND)))
		return DIENUM_CONTINUE;

	// Enumerate the joystick objects. The callback function enabled user
	// interface elements for objects that are found, and sets the min/max
	// values property for discovered axes.
	if (FAILED(hr = g_pJoystick[numLocalJoysticks]->EnumObjects(EnumObjectsCallback, this, DIDFT_ALL)))
		return DIENUM_CONTINUE;
	if ((number_axes[numLocalJoysticks] > 0) || (number_buttons[numLocalJoysticks] > 0))
	{
		fprintf(stderr, "Joystick %zd %s:%s\n", numLocalJoysticks, pdidInstance->tszProductName, pdidInstance->tszInstanceName);
        names[numLocalJoysticks] = pdidInstance->tszInstanceName;
		numLocalJoysticks++;
	}
	return DIENUM_CONTINUE;
}
BOOL CALLBACK Joystick::EnumObjectsCallback(const DIDEVICEOBJECTINSTANCE *pdidoi, VOID *pContext)
{
	Joystick *joystick = (Joystick *)pContext;
	return joystick->EnumObjects(pdidoi);
}

//-----------------------------------------------------------------------------
// Name: EnumObjectsCallback()
// Desc: Callback function for enumerating objects (axes, buttons, POVs) on a
//       joystick. This function enables user interface elements for objects
//       that are found to exist, and scales axes min/max values.
//-----------------------------------------------------------------------------
BOOL Joystick::EnumObjects(const DIDEVICEOBJECTINSTANCE *pdidoi)
{

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
		if (FAILED(g_pJoystick[numLocalJoysticks]->SetProperty(DIPROP_RANGE, &diprg.diph)))
			return DIENUM_STOP;
	}

	if (pdidoi->guidType == GUID_Button)
	{
		number_buttons[numLocalJoysticks]++;
	}
	if (pdidoi->guidType == GUID_XAxis)
	{
		if (number_axes[numLocalJoysticks] < 1)
			number_axes[numLocalJoysticks] = 1;
	}
	if (pdidoi->guidType == GUID_YAxis)
	{
		if (number_axes[numLocalJoysticks] < 2)
			number_axes[numLocalJoysticks] = 2;
	}
	if (pdidoi->guidType == GUID_ZAxis)
	{
		if (number_axes[numLocalJoysticks] < 3)
			number_axes[numLocalJoysticks] = 3;
	}
	if (pdidoi->guidType == GUID_RxAxis)
	{
		if (number_axes[numLocalJoysticks] < 4)
			number_axes[numLocalJoysticks] = 4;
	}
	if (pdidoi->guidType == GUID_RyAxis)
	{
		if (number_axes[numLocalJoysticks] < 5)
			number_axes[numLocalJoysticks] = 5;
	}
	if (pdidoi->guidType == GUID_RzAxis)
	{
		if (number_axes[numLocalJoysticks] < 6)
			number_axes[numLocalJoysticks] = 6;
	}
	if (pdidoi->guidType == GUID_Slider)
	{
		number_sliders[numLocalJoysticks]++;
	}
	if (pdidoi->guidType == GUID_POV)
	{
		number_POVs[numLocalJoysticks]++;
	}

	return DIENUM_CONTINUE;
}

#endif


//==========================main loop =================

/**
 * @brief Joystick::run ImputHdw main loop
 *
 * Gets the status of the input devices
 */
bool Joystick::poll()
{
    #ifdef WIN32
	HRESULT hr;
	DIJOYSTATE2 js; // DInput joystick state
	usleep(0.02);
	int bs = 0;
	for (int i = 0; i < numLocalJoysticks; i++)
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
			return true;
		}

		// Get the input's device state
		if (FAILED(hr = g_pJoystick[i]->GetDeviceState(sizeof(DIJOYSTATE2), &js)))
			return true; // The device should have been acquired during the Poll()

		for (int n = 0; n < number_buttons[i]; n++)
		{
			if (js.rgbButtons[n] & 0x80)
				buttons[i][n] = 1;
			else
				buttons[i][n] = 0;
		}
		for (int n = 0; n < number_sliders[i]; n++)
		{
			sliders[i][n] = js.rglSlider[n] / 1000.0f;
		}
		if (number_axes[i] > 0)
			axes[i][0] = js.lX / 1000.0f;
		if (number_axes[i] > 1)
			axes[i][1] = js.lY / 1000.0f;
		if (number_axes[i] > 2)
			axes[i][2] = js.lZ / 1000.0f;
		if (number_axes[i] > 3)
			axes[i][3] = js.lRx / 1000.0f;
		if (number_axes[i] > 4)
			axes[i][4] = js.lRy / 1000.0f;
		if (number_axes[i] > 5)
			axes[i][5] = js.lRz / 1000.0f;
		if (number_axes[i] > 6)
			number_axes[i] = 6;

		for (int n = 0; n < number_POVs[i]; n++)
		{
			POVs[i][n] = js.rgdwPOV[n];
		}
	}

	m_mutex.lock();
	bs = 0;
	int vs = 0;
	for (int i = 0; i < numLocalJoysticks; i++)
	{

		for (int n = 0; n < number_buttons[i]; n++)
		{
			while (bs >= m_buttonStates.size())
				m_buttonStates.push_back(0);
			m_buttonStates[bs] = buttons[i][n];
			bs++;
		}
		for (int n = 0; n < number_axes[i]; n++)
		{
			while (vs >= m_valuatorValues.size())
			{
				m_valuatorValues.push_back(0);
				m_valuatorRanges.push_back(std::pair<double,double>(0.0,1.0));
			}
			m_valuatorValues[vs] = axes[i][n];
			vs++;
		}
		
	}
    m_valid = true;
    m_mutex.unlock();
    #else
    fd_set read_fds;      // Set of file descriptors to monitor
    int max_fd = -1;      // Highest file descriptor value

    FD_ZERO(&read_fds);   // Clear the fd set

    // Add valid joystick file descriptors to the set
    for (int joystickNumber = 0; joystickNumber < numLocalJoysticks; joystickNumber++) {
        if (fd[joystickNumber] != -1) {
            FD_SET(fd[joystickNumber], &read_fds);
            if (fd[joystickNumber] > max_fd) {
                max_fd = fd[joystickNumber];
            }
        }
    }
    // Use select to wait for input on any of the file descriptors
    int activity = select(max_fd + 1, &read_fds, NULL, NULL, NULL);

    if (activity < 0) {
        perror("select");
    }

    for (int joystickNumber = 0; joystickNumber < numLocalJoysticks; joystickNumber++)
    {
       int button_pressed = -1;
       int button_released = -1;
       struct js_event js;
       if (fd[joystickNumber] != -1 && FD_ISSET(fd[joystickNumber], &read_fds))
       {
        if(read(fd[joystickNumber], &js,
                sizeof(struct js_event)) == sizeof(struct js_event))
        {
#ifdef DEBUG
            fprintf(stderr, "Event: type %d, time %d, number %d, value %d\n",
                js.type, js.time, js.number, js.value);
#endif
            switch (js.type & ~JS_EVENT_INIT)
            {
            case JS_EVENT_BUTTON:
                cout << "js.value" << js.value << endl;
                cout << "js.number" << js.number << endl;
                cout << "joystickNumber" << joystickNumber << endl;
                if (js.value)
                {
                    buttons[joystickNumber][js.number] = 1;
                    button_pressed = js.number;
                }
                else
                {
                    buttons[joystickNumber][js.number] = 0;
                    button_released = js.number;
                }
                break;
            case JS_EVENT_AXIS:
                axes[joystickNumber][js.number] = js.value / 32767.0f;
                break;
            }
            if ((button_pressed != -1) || (button_released != -1))
                break;
        }
 
       }
    }
    #endif
	for (size_t i = 0; i < number_axes[joystickNumber]; i++)
	{
		m_valuatorValues[i] = axes[joystickNumber][i];
	}

	for (size_t i = 0; i < number_buttons[joystickNumber]; i++)
	{
		m_buttonStates[i] = buttons[joystickNumber][i];
	}
    return true;
}

INPUT_PLUGIN(Joystick)
