/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SpaceNavigatorPlugin_si.h"
#include <cover/coVRNavigationManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRCommunication.h>
#include <cover/VRSceneGraph.h>
#include <config/CoviseConfig.h>
#include <cover/coVRMSController.h>

using namespace covise;
using namespace opencover;

SpaceNavigator::SpaceNavigator()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "SpaceNavigator::SpaceNavigator\n");

    memset(&m_smd, 0, sizeof(m_smd));
    m_trans_sensitivity = 1.0f;
    m_rot_sensitivity = 0.5f;

    SiOpenData open_data;
    // init the SpaceWare input library
    if (SiInitialize() == SPW_DLL_LOAD_ERROR)
    {
        fprintf(stderr, "Error: Could not load siappdll! Did you install the drivers?\n");
        return;
    }

    m_hwnd = FindWindow(NULL, "OpenCOVER");
    if (m_hwnd == NULL)
    {
        fprintf(stderr, "Error: FindWindow() - failed!\n");
        return;
    }

    SiOpenWinInit(&open_data, m_hwnd);
    SiSetUiMode(m_dev_handle, SI_UI_ALL_CONTROLS);

    if ((m_dev_handle = SiOpen("3DxOpenCOVER", SI_ANY_DEVICE, SI_NO_MASK, SI_EVENT, &open_data)) == NULL)
    {
        SiTerminate();
        fprintf(stderr, "Error: Could not open 3Dx-Input-Device!\n");
        return;
    }

    memset(&m_dev_info, 0, sizeof(m_dev_info));
    SpwRetVal err = SiGetDeviceInfo(m_dev_handle, &m_dev_info);
    if (err == SPW_ERROR)
    {
        fprintf(stderr, "Error: SiGetDeviceInfo() - failed!\n");
    }
}

// this is called if the plugin is removed at runtime
SpaceNavigator::~SpaceNavigator()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "SpaceNavigator::~SpaceNavigator\n");

    SiClose(m_dev_handle);
    SiTerminate();
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
        relMat.makeTranslate(m_smd.tx * speed, m_smd.ty * speed, m_smd.tz * speed);
        transformMat *= relMat;

        //relMat.makeRotate(angle/500.0,rotX,rotY,rotZ);
        MAKE_EULER_MAT(relMat, m_smd.h, m_smd.p, m_smd.r);
        osg::Matrix originTrans, invOriginTrans;
        originTrans.makeTranslate(originInWorld); // rotate arround the center of the objects in objectsRoot
        invOriginTrans.makeTranslate(-originInWorld);
        relMat = invOriginTrans * relMat * originTrans;
        transformMat *= relMat;
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Fly)
    {
        osg::Matrix relMat;
        relMat.makeTranslate(-m_smd.tx * speed, -m_smd.ty * speed, -m_smd.tz * speed);
        transformMat *= relMat;

        //relMat.makeRotate(-angle/5000.0,rotX,rotY,rotZ);
        MAKE_EULER_MAT(relMat, -m_smd.h, -m_smd.p, -m_smd.r);

        transformMat *= relMat;
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Glide || coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Walk)
    {
        osg::Matrix relMat;
        relMat.makeTranslate(-m_smd.tx * speed, -m_smd.ty * speed, -m_smd.tz * speed);
        transformMat *= relMat;

        relMat.makeRotate(-m_smd.h / 10.0, 0, 0, 1);
        transformMat *= relMat;

        if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::Walk)
        {
            coVRNavigationManager::instance()->doWalkMoveToFloor();
        }
    }
    if (!coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM) || coVRCommunication::instance()->isRILockedByMe(coVRCommunication::TRANSFORM))
    {
        VRSceneGraph::instance()->getTransform()->setMatrix(transformMat);
        coVRCollaboration::instance()->SyncXform();
    }
}

// The timer callback is used to poll the 3d input device for change of keystates and
// the cap displacement values.
void SpaceNavigator::preFrame()
{
    pollDevice();
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((char *)&m_smd, sizeof(m_smd));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&m_smd, sizeof(m_smd));
    }

    coVRNavigationManager::instance()->processHotKeys(m_smd.buttonStatus);
    doNavigation();
}

unsigned int SpaceNavigator::button(int /*station*/)
{
    return m_smd.buttonStatus;
}

void SpaceNavigator::pollDevice()
{
    unsigned long button_num, mask;
    SiSpwEvent event;
    SiGetEventData event_data;

    MSG msg;

    while (::PeekMessage(&msg, m_hwnd, NULL, NULL, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    m_smd.tx = 0.0f;
    m_smd.ty = 0.0f;
    m_smd.tz = 0.0f;
    m_smd.h = 0.0f;
    m_smd.p = 0.0f;
    m_smd.r = 0.0f;

    // init window platform specific data for a call to SiGetEvent
    SiGetEventWinInit(&event_data, msg.message, msg.wParam, msg.lParam);

    // check whether msg was a Spaceball event and process it
    if (SiGetEvent(m_dev_handle, 0, &event_data, &event) == SI_IS_EVENT)
    {
        if (event.type == SI_MOTION_EVENT)
        {
            // process spaceball motion event
            m_smd.tx = (float)event.u.spwData.mData[SI_TX] * m_trans_sensitivity;
            m_smd.ty = (float)event.u.spwData.mData[SI_TY] * m_trans_sensitivity;
            m_smd.tz = (float)event.u.spwData.mData[SI_TZ] * m_trans_sensitivity;
            osg::Matrix relMat;
            relMat.makeRotate(osg::DegreesToRadians((float)event.u.spwData.mData[SI_RZ] * m_rot_sensitivity), osg::Vec3(0, 1, 0),
                              osg::DegreesToRadians((float)event.u.spwData.mData[SI_RX] * m_rot_sensitivity), osg::Vec3(1, 0, 0),
                              osg::DegreesToRadians((float)event.u.spwData.mData[SI_RY] * m_rot_sensitivity), osg::Vec3(0, 0, 1));
            coCoord coord = relMat;
            m_smd.h = coord.hpr[0];
            m_smd.p = coord.hpr[1];
            m_smd.r = coord.hpr[2];
        }
        if (event.type == SI_ZERO_EVENT)
        {
            // clear previous data, no motion data was recieved
            m_smd.tx = 0.0f;
            m_smd.ty = 0.0f;
            m_smd.tz = 0.0f;
            m_smd.p = 0.0f;
            m_smd.h = 0.0f;
            m_smd.r = 0.0f;
        }
        if (event.type == SI_BUTTON_EVENT)
        {
            if ((button_num = SiButtonPressed(&event)) != SI_NO_BUTTON)
            {
                //SbButtonPressEvent(button_num);
                mask = 1 << (button_num - 1);
                m_smd.buttonStatus |= mask;
            }
            if ((button_num = SiButtonReleased(&event)) != SI_NO_BUTTON)
            {
                //SbButtonReleaseEvent(button_num);
                mask = 1 << (button_num - 1);
                m_smd.buttonStatus &= ~mask;
            }
        }
    }
}

COVERPLUGIN(SpaceNavigator)
