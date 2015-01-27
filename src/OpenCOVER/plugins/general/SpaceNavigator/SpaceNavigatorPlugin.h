/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SPACENAVIGATOR_PLUGIN_H
#define _SPACENAVIGATOR_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Input Plugin for SpaceNavigator                             **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                             **
 **                                                                          **
 ** History:  			                                             **
 ** June-08  v1	    				                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

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
#else

#include <unistd.h>
#include <string.h>

#endif
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>

typedef struct messageBuf
{
    int ts1;
    int ts2;
    int ts3;
    int ts4;
    unsigned char type;
    unsigned char val[7];
} usbMessage;

class SpaceNavigator;

class DeviceThread : public OpenThreads::Thread
{
public:
    DeviceThread(SpaceNavigator *);
    ~DeviceThread();
    float fvalues[6];
    unsigned int buttonStatus;
    bool isWorking()
    {
        return (fileDesc >= 0);
    }
    virtual void run();

private:
    SpaceNavigator *sn;
    int bufSize;
    std::string deviceFile;
    int fileDesc;
    char *buf;
    bool exit;
};

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

class SpaceNavigator : public coVRPlugin
{
public:
    SpaceNavigator();
    ~SpaceNavigator();

    // this will be called to get the button status
    unsigned int button(int station);
    virtual void preFrame();

private:
    SpaceMouseData smd;
    double Sensitivity;
    DeviceThread *dThread;
    void doNavigation();

    void pollDevice();
    void spacedMouseEvent(double transX, double transY, double transZ, double rotX, double rotY, double rotZ, double angle);
#ifdef WIN32

    //HINSTANCE hInstanceGlobal;
    HACCEL hAccel;
    HWND MainhWnd;
    HDC hDC;
    HMENU hMenu;
    HWND hWnd;

    CComPtr<ISensor> g3DSensor;
    CComPtr<IKeyboard> g3DKeyboard;
    __int64 gKeyStates;

    HWND InitClassWindow(HINSTANCE);
    int InitHiddenWindow(HWND);
    int DeleteHiddenWindow(HWND);
#endif
};
#endif
