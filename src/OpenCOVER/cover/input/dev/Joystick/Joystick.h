/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Joystick.h
 *
 *  Created on: Dec 9, 2015
 *      Author: uwe
 */

#ifndef Joystick_DRIVER_H
#define Joystick_DRIVER_H

#include <OpenThreads/Thread>
#include <string>
#include <cover/input/inputdevice.h>

/**
 * @brief The Joystick class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */


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

#include "TempWindow.h"
#include "cover/coVRPluginSupport.h"

#define NUM_BUTTONS 3

enum
{
	JOYSTICK_BUTTON_EVENTS = 1,
	JOYSTICK_AXES_EVENTS = 2
};

class Joystick : public opencover::InputDevice
{
    virtual bool poll();

public:
	static BOOL CALLBACK EnumObjectsCallback(const DIDEVICEOBJECTINSTANCE *pdidoi, VOID *pContext);
	static BOOL CALLBACK EnumJoysticksCallback(const DIDEVICEINSTANCE *pdidInstance, VOID *pContext);
	BOOL EnumObjects(const DIDEVICEOBJECTINSTANCE *pdidoi);
	BOOL EnumJoysticks(const DIDEVICEINSTANCE *pdidInstance);
	LPDIRECTINPUT8 g_pDI;
	LPDIRECTINPUTDEVICE8 g_pJoystick[MAX_NUMBER_JOYSTICKS];
	TemporaryWindow window;

	int numLocalJoysticks;
	unsigned char number_buttons[MAX_NUMBER_JOYSTICKS];
	int *buttons[MAX_NUMBER_JOYSTICKS];
	unsigned char number_axes[MAX_NUMBER_JOYSTICKS];
	float *axes[MAX_NUMBER_JOYSTICKS];
	unsigned char number_sliders[MAX_NUMBER_JOYSTICKS];
	float *sliders[MAX_NUMBER_JOYSTICKS];
	unsigned char number_POVs[MAX_NUMBER_JOYSTICKS];
	float *POVs[MAX_NUMBER_JOYSTICKS];


	virtual bool needsThread() const { return true; }; //< whether a thread should be spawned - reimplement if not necessary
    Joystick(const std::string &name);
    virtual ~Joystick();
};
#endif
