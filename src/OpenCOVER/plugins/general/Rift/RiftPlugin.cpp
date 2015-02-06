/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>

#include <iostream>
#include <fstream>
#include <functional>

#include <string>
#include <vector>

#include "RiftPlugin.h"
#include "cover/coVRConfig.h"
#include "cover/VRViewer.h"
#include "osgViewer/Viewer"
#include <osgViewer/api/Win32/GraphicsHandleWin32>
#include <cover/input/dev/rift/RIFTDriver.h>

RiftPlugin *RiftPlugin::plugin = NULL;

RiftPlugin::RiftPlugin()
    : tab(NULL)
    , res(NULL)
    , residue(NULL)
    , showSticks(NULL)
    , hideSticks(NULL)
{
    VRViewer::instance()->setRenderToTexture(true);

    hmd = new OculusDevice(coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip(), 1, false);

    if (hmd->getHMD())
    {
        fprintf(stderr, "found %s, serial Nr.: %s %d %d\n", hmd->getHMD()->ProductName, hmd->getHMD()->SerialNumber, hmd->getHMD()->WindowsPos.x, hmd->getHMD()->WindowsPos.y);
        coVRConfig::instance()->windows[0].ox = hmd->getHMD()->WindowsPos.x;
        coVRConfig::instance()->windows[0].oy = hmd->getHMD()->WindowsPos.y;
        ovrFovPort fovLeft = hmd->getHMD()->DefaultEyeFov[ovrEye_Left];
        ovrFovPort fovRight = hmd->getHMD()->DefaultEyeFov[ovrEye_Right];
        coVRConfig::instance()->screens[0].tTan = fovLeft.UpTan;
        coVRConfig::instance()->screens[0].bTan = fovLeft.DownTan;
        coVRConfig::instance()->screens[0].lTan = fovLeft.LeftTan;
        coVRConfig::instance()->screens[0].rTan = fovLeft.RightTan;
        coVRConfig::instance()->screens[1].tTan = fovRight.UpTan;
        coVRConfig::instance()->screens[1].bTan = fovRight.DownTan;
        coVRConfig::instance()->screens[1].lTan = fovRight.LeftTan;
        coVRConfig::instance()->screens[1].rTan = fovRight.RightTan;
    }
    else
    {
        fprintf(stderr, "did not find any Oculus Rift\n");
    }
    m_frameIndex = 0;
}

// this will be called in PreFrame
void RiftPlugin::preFrame()
{
}
void RiftPlugin::postFrame()
{
    osg::Matrix m;
    ovrTrackingState ts = ovrHmd_GetTrackingState(hmd->getHMD(), 0);
    osg::Quat q;
    q.set(ts.HeadPose.ThePose.Orientation.x, ts.HeadPose.ThePose.Orientation.y, ts.HeadPose.ThePose.Orientation.z, ts.HeadPose.ThePose.Orientation.w);
    m.makeRotate(q);
    osg::Vec3 t;
    t.set(ts.HeadPose.ThePose.Position.x * 1000, ts.HeadPose.ThePose.Position.y * 1000, ts.HeadPose.ThePose.Position.z * 1000);
    m.postMultTranslate(t);
    hmd->trackingDriver->setMatrix(m);
    
}
void RiftPlugin::preSwapBuffers(int windowNumber)
{

    // End frame timing when swap buffer is done
    ovrHmd_EndFrameTiming(hmd->getHMD());
    // Start a new frame with incremented frame index

    ovrHmd_BeginFrameTiming(hmd->getHMD(), ++m_frameIndex);
}

void RiftPlugin::getMatrix(int station, osg::Matrix &mat)
{
    if (station == 0)
    {
        ovrTrackingState ts = ovrHmd_GetTrackingState(hmd->getHMD(), 0);
        osg::Quat q;
        q.set(ts.HeadPose.ThePose.Orientation.x, ts.HeadPose.ThePose.Orientation.y, ts.HeadPose.ThePose.Orientation.z, ts.HeadPose.ThePose.Orientation.w);
        mat.makeRotate(q);
        osg::Vec3 t;
        t.set(ts.HeadPose.ThePose.Position.x * 1000, ts.HeadPose.ThePose.Position.y * 1000, ts.HeadPose.ThePose.Position.z * 1000);
        mat.postMultTranslate(t);
    }
    else
    {
        mat.makeIdentity();
    }
    //mat.makeTranslate(0,-1000*(station+1),0);
}

// this is called if the plugin is removed at runtime
RiftPlugin::~RiftPlugin()
{

    delete tab;
    delete res;
    delete residue;
    delete showSticks;
    delete hideSticks;
}

bool RiftPlugin::init()
{

    fprintf(stderr, "RiftPlugin::Plugin\n");
    if (plugin == NULL) 
    {
        plugin = this;
    }

#ifdef WIN32 // enable Display on HMD
    osgViewer::ViewerBase::Windows windows;
    VRViewer::instance()->getWindows(windows);
    if (windows.size() > 0)
    {
        osgViewer::GraphicsHandleWin32 *hdl = dynamic_cast<osgViewer::GraphicsHandleWin32 *>(windows.at(0));
        if (hdl)
        {
            ovrHmd_AttachToWindow(hmd->getHMD(), hdl->getHWND(), NULL, NULL);
        }
    }
#endif
    tab = new coTUITab("Rift", coVRTui::instance()->mainFolder->getID());
    tab->setPos(0, 0);

    res = new coTUILabel("Sticks Residue #", tab->getID());
    res->setPos(0, 0);

    residue = new coTUIEditField("residue", tab->getID());
    residue->setEventListener(this);
    residue->setText("0");
    residue->setPos(1, 0);

    showSticks = new coTUIButton("show", tab->getID());
    showSticks->setEventListener(this);
    showSticks->setPos(2, 0);

    hideSticks = new coTUIButton("hide", tab->getID());
    hideSticks->setEventListener(this);
    hideSticks->setPos(3, 0);
    // Create warp ortho camera

    osg::ref_ptr<osg::Camera> cameraWarp = hmd->createWarpOrthoCamera(0.0, 1.0, 0.0, 1.0);
    cameraWarp->setName("WarpOrtho");
    cameraWarp->setViewport(new osg::Viewport(0, 0, hmd->screenResolutionWidth(), hmd->screenResolutionHeight()));

    // Create shader program
    osg::ref_ptr<osg::Program> program = hmd->createShaderProgram();

    // Create distortionMesh for each camera
    //osg::ref_ptr<osg::Geode> leftDistortionMesh = hmd->distortionMesh(OculusDevice::LEFT, program, 0, 0, coVRConfig::instance()->screens[0].viewportXMax-coVRConfig::instance()->screens[0].viewportXMin, coVRConfig::instance()->screens[0].viewportYMax-coVRConfig::instance()->screens[0].viewportYMin);
    osg::ref_ptr<osg::Geode> leftDistortionMesh = hmd->distortionMesh(OculusDevice::LEFT, program, 0, 0, 960, 1080);
    cameraWarp->addChild(leftDistortionMesh);

    //osg::ref_ptr<osg::Geode> rightDistortionMesh = hmd->distortionMesh(OculusDevice::RIGHT, program, 0, 0, coVRConfig::instance()->screens[1].viewportXMax-coVRConfig::instance()->screens[1].viewportXMin, coVRConfig::instance()->screens[1].viewportYMax-coVRConfig::instance()->screens[1].viewportYMin);
    osg::ref_ptr<osg::Geode> rightDistortionMesh = hmd->distortionMesh(OculusDevice::RIGHT, program, 0, 0, 960, 1080);
    cameraWarp->addChild(rightDistortionMesh);

    // Add pre draw camera to handle time warp
    cameraWarp->setPreDrawCallback(new WarpCameraPreDrawCallback(hmd));

    // Attach shaders to each distortion mesh
    osg::StateSet *leftEyeStateSet = leftDistortionMesh->getOrCreateStateSet();
    osg::StateSet *rightEyeStateSet = rightDistortionMesh->getOrCreateStateSet();

    hmd->applyShaderParameters(leftEyeStateSet, program.get(), coVRConfig::instance()->screens[0].renderTargetTexture.get(), OculusDevice::LEFT);
    hmd->applyShaderParameters(rightEyeStateSet, program.get(), coVRConfig::instance()->screens[1].renderTargetTexture.get(), OculusDevice::RIGHT);

    //osg::Group *renderGroup = new osg::Group();
    //renderGroup->addChild(cameraWarp.get());

    cover->getScene()->addChild(cameraWarp.get());
    return true;
}

/*
 * TUI provides the possibility to show/hide sticks of bonds that
 * do not belong to the backbone.
 * User input can be a single number of a residue or a range of numbers 
 */
void RiftPlugin::tabletEvent(coTUIElement *elem)
{
}

void RiftPlugin::tabletReleaseEvent(coTUIElement * /*tUIItem*/)
{
}

COVERPLUGIN(RiftPlugin)
