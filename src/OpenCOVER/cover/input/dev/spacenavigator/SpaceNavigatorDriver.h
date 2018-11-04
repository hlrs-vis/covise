/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** Description: Input Plugin for SpaceNavigator                             */

#ifndef SPACENAVIGATOR_DRIVER_H
#define SPACENAVIGATOR_DRIVER_H

#ifdef WIN32
#include <atlbase.h>
#include <atlcom.h>
#include <atlwin.h>
#include <atltypes.h>
#include <atlctl.h>
#include <atlhost.h>
using namespace ATL;
// you need this to generate the following include file #import "progid:TDxInput.Device.1" no_namespace
#include "TDxInput.tlh"
#include <atlstr.h>
#include <stdio.h>
#else
#include <unistd.h>
#endif

#include <string>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <cover/input/inputdevice.h>

struct hid_device_;
typedef struct hid_device_ hid_device;

typedef struct SMD
{
	float tx;
	float ty;
	float tz;
	float h;
	float p;
	float r;
	unsigned int buttonStatus;
} SpaceMouseData;


class SpaceNavigator;

class SpaceNavigatorDriver : public opencover::InputDevice
{
public:
    virtual bool poll() override;

    SpaceNavigatorDriver(const std::string &config);
    virtual ~SpaceNavigatorDriver();

private:
    bool m_hidapi = false;
    int m_fd = -1;
    std::vector<char> m_buf;
    bool m_spacemouse = false;
    std::vector<int> m_raw;

    void processRaw(int axis, int value);

    bool hidapi_init();
    bool hidapi_poll();
    void hidapi_finish();
    bool hidapi_recalibrate();
    hid_device *m_hidapiHandle = nullptr;
    double m_lastUpdate = -1.;
    bool m_evdevRel = false;
#ifdef WIN32
	void spaceMouseEvent(double transX, double transY, double transZ, double rotX, double rotY, double rotZ, double angle);
	SpaceMouseData smd;
	//HINSTANCE hInstanceGlobal;
	HACCEL hAccel;
	HWND MainhWnd;
	HDC hDC;
	HMENU hMenu;
	HWND hWnd;

	CComPtr<ISensor> g3DSensor;
	CComPtr<IKeyboard> g3DKeyboard;
	__int64 gKeyStates;

#endif
};
#endif
