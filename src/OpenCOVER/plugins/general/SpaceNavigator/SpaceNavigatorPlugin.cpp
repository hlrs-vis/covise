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
   - run e.g. 'inputattach --magellan /dev/ttyS0&'
   - point COVER.Input.SpaceNavigator to the created event device */

#include "SpaceNavigatorPlugin.h"
#include <cover/coVRNavigationManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRCommunication.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>
#include <cover/coVRConfig.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>
#include <OpenVRUI/osg/mathUtils.h>

//#define VERBOSE

DeviceThread::DeviceThread(SpaceNavigator *s)
{
    sn = s;
    exit = false;
    fileDesc = 0;
    buttonStatus = 0;
    buf = NULL;
    for (int i = 0; i < 6; i++)
        fvalues[i] = 0.0;

    deviceFile = coCoviseConfig::getEntry("device", "COVER.Input.SpaceNavigator", "/dev/input/spacenavigator");

    fileDesc = open(deviceFile.c_str(), O_RDONLY);
    if (fileDesc >= 0)
    {
#ifndef WIN32
        struct stat statbuf;
        statbuf.st_blksize = 4096;
        fstat(fileDesc, &statbuf);
        bufSize = statbuf.st_blksize;
#else
        bufSize = 4096;
#endif
        buf = new char[bufSize];
    }
    else
    {
        fprintf(stderr, "SpaceNavigator Plugin:: failed to opening %s\n", deviceFile.c_str());
    }
}
DeviceThread::~DeviceThread()
{
    exit = true;
    delete[] buf;
}

// see <linux/input.h>
struct InputEvent
{
    struct timeval time;
    uint16_t type;
    uint16_t code;
    int32_t value;
};

void DeviceThread::run()
{

    while (!exit)
    {
        int numRead = read(fileDesc, buf, bufSize);
        if (numRead > 0)
        {
            int i = 0;
            while (i < numRead)
            {
                InputEvent *message = (InputEvent *)(buf + i);
                if (message->type == 1) // EV_KEY - button press
                {
                    int buttonNumber = message->code % 32;

#ifdef VERBOSE
                    fprintf(stderr, "Button %d ", buttonNumber);
#endif
                    if (message->value)
                    {
                        buttonStatus |= 1 << buttonNumber;
#ifdef VERBOSE
                        fprintf(stderr, "Pressed \n");
#endif
                    }
                    else
                    {
                        buttonStatus &= ~(1 << buttonNumber);
#ifdef VERBOSE
                        fprintf(stderr, "Released \n");
#endif
                    }
                }
                else if (message->type == 2) // EV_REL - motion on SpaceNavigator
                {
                    int axis = message->code;
#ifdef VERBOSE
                    fprintf(stderr, "axis=%d\n", axis);
#endif
                    float fvalue = 0.;
                    int value = message->value;
                    if (value > 9)
                    {
                        if (value > 410)
                            value = 410;
                        fvalue = (value - 10) / 400.0;
                    }
                    else if (value < -9)
                    {
                        if (value < -410)
                            value = -410;
                        fvalue = (value + 10) / 400.0;
                    }
                    else
                    {
                        fvalue = 0.0;
                        value = 0;
                    }

                    if (axis % 3 == 1)
                    {
                        ++axis;
                    }
                    else if (axis % 3 == 2)
                    {
                        --axis;
                        fvalue *= -1.;
                    }
#ifdef VERBOSE
                    fprintf(stderr, "axis=%d, val=%f\n", axis, fvalue);
#endif
                    fvalues[axis] = fvalue;
                }
                else if (message->type == 3) // EV_ABS - motion on SpaceMouse
                {
                    int axis = message->code;
                    int value = message->value;
                    float fvalue = 0.;
                    if (abs(value) < 5)
                        value = 0;
                    fvalue = value / 128.;
                    fvalues[axis] = fvalue;
#ifdef VERBOSE
                    fprintf(stderr, "message: type=%d, code=%d, val=%d\n", message->type, message->code, message->value);
                    fprintf(stderr, "axis=%d, val=%f\n", axis, fvalue);
#endif
                }
                i += sizeof(InputEvent);
            }
        }
        else
        {
            perror("read");
            exit = true;
        }
    }
    close(fileDesc);
}

SpaceNavigator::SpaceNavigator()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "SpaceNavigator::SpaceNavigator\n");
    smd.buttonStatus = 0;
    Sensitivity = 1.0;
    dThread = NULL;
#ifdef WIN32
    hWnd = NULL;
    HRESULT hr = ::CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (!SUCCEEDED(hr))
    {
        CString strError;
        strError.FormatMessage(_T("Error 0x%x"), hr);
        ::MessageBox(NULL, strError, _T("CoInitializeEx failed"), MB_ICONERROR | MB_OK);
    }
    else
    {
        HWND hDesktopWnd;

        HRESULT hr = ::CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
        if (!SUCCEEDED(hr))
        {
            CString strError;
            strError.FormatMessage(_T("Error 0x%x"), hr);
            ::MessageBox(NULL, strError, _T("CoInitializeEx failed"), MB_ICONERROR | MB_OK);
        }
        else
        {
            hDesktopWnd = GetDesktopWindow();

            //InitHiddenWindow( MainhWnd );

            HRESULT hr;
            CComPtr<IUnknown> _3DxDevice;

            // Create the device object
            hr = _3DxDevice.CoCreateInstance(__uuidof(Device));
            if (SUCCEEDED(hr))
            {
                CComPtr<ISimpleDevice> _3DxSimpleDevice;

                hr = _3DxDevice.QueryInterface(&_3DxSimpleDevice);
                if (SUCCEEDED(hr))
                {
                    // Get the interfaces to the sensor and the keyboard;
                    g3DSensor = _3DxSimpleDevice->Sensor;
                    g3DKeyboard = _3DxSimpleDevice->Keyboard;

                    // Associate a configuration with this device
                    _3DxSimpleDevice->LoadPreferences(_T("OpenCOVER"));
                    // Connect to the driver
                    _3DxSimpleDevice->Connect();
                }
            }
            hWnd = FindWindow("3DxInput:OpenCOVER", "OpenCOVER.exe");
        }
    }
#else
    dThread = new DeviceThread(this);
    if (dThread->isWorking())
        dThread->start();
#endif
}

// this is called if the plugin is removed at runtime
SpaceNavigator::~SpaceNavigator()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "SpaceNavigator::~SpaceNavigator\n");
#ifdef WIN32
    CComPtr<ISimpleDevice> _3DxDevice;

    // Release the sensor and keyboard interfaces
    if (g3DSensor)
    {
        g3DSensor->get_Device((IDispatch **)&_3DxDevice);
        g3DSensor.Release();
    }
    if (g3DKeyboard)
        g3DKeyboard.Release();
    if (_3DxDevice)
    {
        // Disconnect it from the driver
        _3DxDevice->Disconnect();
        _3DxDevice.Release();
    }
#else
    delete dThread;
#endif
}

void SpaceNavigator::spacedMouseEvent(double transX, double transY, double transZ, double rotX, double rotY, double rotZ, double angle)
{
    smd.tx = transX;
    smd.ty = transY;
    smd.tz = transZ;

    osg::Matrix relMat;
    relMat.makeRotate(angle / 500.0, rotX, rotY, rotZ);
    coCoord coord = relMat;
    smd.h = coord.hpr[0];
    smd.p = coord.hpr[1];
    smd.r = coord.hpr[2];
}

void SpaceNavigator::doNavigation()
{
    osg::Matrix transformMat;
    transformMat = VRSceneGraph::instance()->getTransform()->getMatrix();
    float speed = coVRNavigationManager::instance()->getDriveSpeed();
    if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::XForm)
    {

        osg::BoundingSphere bsphere = VRSceneGraph::instance()->getBoundingSphere();
        if (bsphere.radius() == 0.f)
            bsphere.radius() = 1.f;

        osg::Vec3 originInWorld = bsphere.center() * cover->getBaseMat();

        osg::Matrix relMat;
        relMat.makeTranslate(smd.tx * speed, smd.ty * speed, smd.tz * speed);
        transformMat *= relMat;

        //relMat.makeRotate(angle/500.0,rotX,rotY,rotZ);
        MAKE_EULER_MAT(relMat, smd.h, smd.p, smd.r);
        osg::Matrix originTrans, invOriginTrans;
        originTrans.makeTranslate(originInWorld); // rotate arround the center of the objects in objectsRoot
        invOriginTrans.makeTranslate(-originInWorld);
        relMat = invOriginTrans * relMat * originTrans;
        transformMat *= relMat;
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Fly)
    {

        osg::Matrix relMat;
        relMat.makeTranslate(-smd.tx * speed, -smd.ty * speed, -smd.tz * speed);
        transformMat *= relMat;

        //relMat.makeRotate(-angle/5000.0,rotX,rotY,rotZ);
        MAKE_EULER_MAT(relMat, -smd.h, -smd.p, -smd.r);

        transformMat *= relMat;
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Glide || coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Walk)
    {
        osg::Matrix relMat;
        relMat.makeTranslate(-smd.tx * speed, -smd.ty * speed, -smd.tz * speed);
        transformMat *= relMat;

        relMat.makeRotate(-smd.h / 10.0, 0, 0, 1);
        //MAKE_EULER_MAT(relMat,-smd.h, -smd.p, -smd.r);
        transformMat *= relMat;

        if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Walk)
        {
            coVRNavigationManager::instance()->doWalkMoveToFloor();

            for (int i = 0; i < coVRConfig::instance()->numScreens(); i++)
                coVRConfig::instance()->screens[i].xyz[2] += (smd.p * 10);
        }
    }
	VRSceneGraph::instance()->getTransform()->setMatrix(transformMat);
	coVRCollaboration::instance()->SyncXform();
    //if (!coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM) || coVRCommunication::instance()->isRILockedByMe(coVRCommunication::TRANSFORM))
    //{
    //    VRSceneGraph::instance()->getTransform()->setMatrix(transformMat);
    //    coVRCollaboration::instance()->SyncXform();
    //}
}

// The timer callback is used to poll the 3d input device for change of keystates and
// the cap displacement values.
void SpaceNavigator::preFrame()
{
    pollDevice();
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((char *)&smd, sizeof(smd));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&smd, sizeof(smd));
    }

    coVRNavigationManager::instance()->processHotKeys(smd.buttonStatus);
    doNavigation();
}

unsigned int SpaceNavigator::button(int /*station*/)
{
    return smd.buttonStatus;
}

void SpaceNavigator::pollDevice()
{
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

            auto pRotation = g3DSensor->Rotation;
            auto pTranslation = g3DSensor->Translation;

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
    if (dThread->fvalues[0] < 0)
        smd.tx = -dThread->fvalues[0] * dThread->fvalues[0] * 20;
    else
        smd.tx = dThread->fvalues[0] * dThread->fvalues[0] * 20;

    if (dThread->fvalues[2] < 0)
        smd.ty = dThread->fvalues[2] * dThread->fvalues[2] * 20;
    else
        smd.ty = -dThread->fvalues[2] * dThread->fvalues[2] * 20;

    if (dThread->fvalues[1] < 0)
        smd.tz = -dThread->fvalues[1] * dThread->fvalues[1] * 20;
    else
        smd.tz = dThread->fvalues[1] * dThread->fvalues[1] * 20;

    if (dThread->fvalues[4] < 0)
        smd.h = -dThread->fvalues[4] * dThread->fvalues[4];
    else
        smd.h = dThread->fvalues[4] * dThread->fvalues[4];

    if (dThread->fvalues[3] < 0)
        smd.p = -dThread->fvalues[3] * dThread->fvalues[3];
    else
        smd.p = dThread->fvalues[3] * dThread->fvalues[3];

    if (dThread->fvalues[5] < 0)
        smd.r = dThread->fvalues[5] * dThread->fvalues[5];
    else
        smd.r = -dThread->fvalues[5] * dThread->fvalues[5];

    smd.buttonStatus = dThread->buttonStatus;
#endif
}

COVERPLUGIN(SpaceNavigator)
