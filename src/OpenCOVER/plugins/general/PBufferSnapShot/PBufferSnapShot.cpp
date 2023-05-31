/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
/****************************************************************************\
 **                                                            (C)2004 HLRS  **
 **                                                                          **
 ** Description: PBufferSnapShot Plugin                                      **
 **                                                                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

//#define PB_TESTING

#include "PBufferSnapShot.h"

#include <config/CoviseConfig.h>
#define QT_CLEAN_NAMESPACE
#include <QString>
#include <QDebug>
#include <QRegularExpression>
#include "qdir.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/RenderObject.h>
#include <cover/VRViewer.h>
#include <cover/coVRSceneView.h>
#include <cover/coVRMSController.h>
#include <cover/VRWindow.h>
#include <cover/coVRTui.h>
#include <cover/coVRRenderer.h>
#include <util/coFileUtil.h>
#include <osgViewer/Renderer>

#include <PluginUtil/PluginMessageTypes.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>

#include <grmsg/coGRSnapshotMsg.h>
#include <net/tokenbuffer.h>

#include <osgDB/WriteFile>
#include <osg/Camera>

#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>

#define PB_RESOLUTION_MAX_X 13064
#define PB_RESOLUTION_MAX_Y 9800

using namespace std;
using namespace grmsg;
using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::coDirectory;

PBufferSnapShot::Resolution PBufferSnapShot::resolutions[] = {
    { "Custom", 0, 0 },
    { "Native", 0, 0 },
    { "1024x768", 1024, 768 },
    { "2048x1536", 2048, 1536 },
    { "4096x3072", 4096, 3072 },
    { "8192x6144", 8192, 6144 },
    { "Maxpect", PB_RESOLUTION_MAX_X, PB_RESOLUTION_MAX_X / 4 * 3 },
    { "Max", PB_RESOLUTION_MAX_X, PB_RESOLUTION_MAX_Y },
};

class DrawCallback : public osg::Camera::DrawCallback
{
public:
    DrawCallback(PBufferSnapShot *plugin)
        : plugin(plugin)
    {
    }
    virtual void operator()(const osg::Camera &cam) const;

private:
    PBufferSnapShot *plugin;
};

void DrawCallback::operator()(const osg::Camera &cam) const
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::operator()\n");

    (void)cam;

    if (plugin->doSnap)
    {
        if (plugin->stereo)
        {

            osg::ref_ptr<osg::Image> image;
            image = new osg::Image();
            image.get()->allocateImage(2 * plugin->image.get()->s(), plugin->image.get()->t(), 1, plugin->image.get()->getPixelFormat(), GL_UNSIGNED_BYTE);
            image.get()->copySubImage(0, 0, 0, plugin->image.get());
            image.get()->copySubImage(plugin->image.get()->s(), 0, 0, plugin->imageR.get());

            if(osgDB::writeImageFile(*(image.get()), plugin->filename))
            {
                const char *savedFile = coDirectory::canonical(plugin->filename.c_str()); 
                plugin->lastSavedFile = savedFile;
                char *text = new char[strlen(savedFile)+100];
                sprintf(text,"saved: %s",savedFile);
                plugin->tuiSavedFile->setLabel(text); // don't just set filename as label, otherwise the image is displayed.
                plugin->tuiSavedFile->setColor(Qt::black);
            }
            else
            {
                plugin->tuiSavedFile->setLabel("failed to write: " + std::string(coDirectory::canonical(plugin->filename.c_str())));
                plugin->tuiSavedFile->setColor(Qt::red);
            }
        }
        else
        {
            if(osgDB::writeImageFile(*(plugin->image.get()), coDirectory::canonical(plugin->filename.c_str())))
            {
                const char *savedFile = coDirectory::canonical(plugin->filename.c_str()); 
                plugin->lastSavedFile = savedFile;
                char *text = new char[strlen(savedFile)+100];
                sprintf(text," %s",savedFile);
                plugin->tuiSavedFile->setLabel(text);
                plugin->tuiSavedFile->setColor(Qt::black);
            }
            else
            {
                plugin->tuiSavedFile->setLabel("failed to write: " + std::string(coDirectory::canonical(plugin->filename.c_str())));
                plugin->tuiSavedFile->setColor(Qt::red);
            }
        }
        plugin->cameraCallbackExit();
    }
}

PBufferSnapShot::PBufferSnapShot()
: coVRPlugin(COVER_PLUGIN_NAME)
, NumResolutions(sizeof(PBufferSnapShot::resolutions) / sizeof(PBufferSnapShot::resolutions[0]))
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::PBufferSnapShot\n");

    image = new osg::Image();
    imageR = new osg::Image();
    doSnap = false;
    myDoSnap = false;

    filename = "";
    snapID = 0;

    counter = 0;

    removeCamera = false;
    pBufferCamera = NULL;
    pBufferCameraR = NULL;
    drawCallback = new DrawCallback(this);
}
void PBufferSnapShot::prepareSnapshot()
{

    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::prepareSnapshot\n");
    if (coVRMSController::instance()->isMaster() || tuiSnapOnSlaves->getState() == true)
    {

        stereo = tuiStereoCheckbox->getState();
        removeCamera = false;
        // Create the Camera intended for rendering into a Pbuffer
        if (pBufferCameras[osg::Camera::FRAME_BUFFER_OBJECT + tuiRenderingMethod->getSelectedEntry()] == NULL)
        {
            pBufferCameras[osg::Camera::FRAME_BUFFER_OBJECT + tuiRenderingMethod->getSelectedEntry()] = new osg::Camera();
            pBufferCamerasR[osg::Camera::FRAME_BUFFER_OBJECT + tuiRenderingMethod->getSelectedEntry()] = new osg::Camera();
        }
        pBufferCamera = pBufferCameras[osg::Camera::FRAME_BUFFER_OBJECT + tuiRenderingMethod->getSelectedEntry()];
        pBufferCameraR = pBufferCamerasR[osg::Camera::FRAME_BUFFER_OBJECT + tuiRenderingMethod->getSelectedEntry()];

        osg::Camera *cam = dynamic_cast<osg::Camera *>(coVRConfig::instance()->channels[0].camera.get());

        tabletEvent(tuiResolution);
        if (tuiTransparentBackground->getState() == true)
        {
            if (stereo)
                imageR.get()->allocateImage(tuiResolutionX->getValue(), tuiResolutionY->getValue(), 1, GL_RGBA, GL_UNSIGNED_BYTE);
            image.get()->allocateImage(tuiResolutionX->getValue(), tuiResolutionY->getValue(), 1, GL_RGBA, GL_UNSIGNED_BYTE);
        }
        else
        {
            if (stereo)
                imageR.get()->allocateImage(tuiResolutionX->getValue(), tuiResolutionY->getValue(), 1, GL_RGB, GL_UNSIGNED_BYTE);
            image.get()->allocateImage(tuiResolutionX->getValue(), tuiResolutionY->getValue(), 1, GL_RGB, GL_UNSIGNED_BYTE);
        }
        pBufferCamera->setViewport(0, 0, tuiResolutionX->getValue(), tuiResolutionY->getValue());
        pBufferCameraR->setViewport(0, 0, tuiResolutionX->getValue(), tuiResolutionY->getValue());
        pBufferCamera->setRenderOrder(osg::Camera::PRE_RENDER);
        pBufferCameraR->setRenderOrder(osg::Camera::PRE_RENDER);
        pBufferCamera->setRenderTargetImplementation((osg::Camera::RenderTargetImplementation)(osg::Camera::FRAME_BUFFER_OBJECT + tuiRenderingMethod->getSelectedEntry()));
        pBufferCameraR->setRenderTargetImplementation((osg::Camera::RenderTargetImplementation)(osg::Camera::FRAME_BUFFER_OBJECT + tuiRenderingMethod->getSelectedEntry()));

        pBufferCamera->setPostDrawCallback(drawCallback.get());

        if (tuiTransparentBackground->getState() == true)
        {
            pBufferCamera->setClearColor(osg::Vec4(0, 0, 0, 0));
            pBufferCameraR->setClearColor(osg::Vec4(0, 0, 0, 0));
            pBufferCamera->setClearMask(cam->getClearMask());
            pBufferCameraR->setClearMask(cam->getClearMask());
        }
        else
        {
            pBufferCamera->setClearColor(cam->getClearColor());
            pBufferCameraR->setClearColor(cam->getClearColor());
            pBufferCamera->setClearMask(cam->getClearMask());
            pBufferCameraR->setClearMask(cam->getClearMask());
        }
        if (stereo)
        {
            pBufferCamera->setProjectionMatrix(coVRConfig::instance()->channels[0].leftProj);
            pBufferCameraR->setProjectionMatrix(coVRConfig::instance()->channels[0].rightProj);
            pBufferCamera->setViewMatrix(coVRConfig::instance()->channels[0].leftView);
            pBufferCameraR->setViewMatrix(coVRConfig::instance()->channels[0].rightView);
        }
        else
        {
            pBufferCamera->setProjectionMatrix(cam->getProjectionMatrix());
            pBufferCamera->setViewMatrix(cam->getViewMatrix());
        }
        pBufferCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
        pBufferCameraR->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
        pBufferCamera->setView(cam->getView());
        //pBufferCamera->setCullMask(Isect::Visible);
        
        pBufferCamera->setCullMask(~0 & ~(Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible
        pBufferCamera->setCullMaskLeft(~0 & ~(Isect::Right|Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible and not right
        pBufferCamera->setCullMaskRight(~0 & ~(Isect::Left|Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible and not Left
       

        if (stereo)
        {
            osgViewer::Renderer *rendererR = new coVRRenderer(pBufferCameraR.get(), 0);
            pBufferCameraR->setRenderer(rendererR);
            pBufferCameraR->setGraphicsContext(cam->getGraphicsContext());
            VRViewer::instance()->addCamera(pBufferCameraR.get());
            pBufferCameraR->attach(osg::Camera::COLOR_BUFFER, imageR.get());
            pBufferCameraR->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
            rendererR->getSceneView(0)->setSceneData(cover->getScene());
            rendererR->getSceneView(1)->setSceneData(cover->getScene());
        }
        osgViewer::Renderer *renderer = new coVRRenderer(pBufferCamera.get(), 0);
        pBufferCamera->setRenderer(renderer);
        pBufferCamera->setGraphicsContext(cam->getGraphicsContext());
        VRViewer::instance()->addCamera(pBufferCamera.get());
        pBufferCamera->attach(osg::Camera::COLOR_BUFFER, image.get());
        //pBufferCamera->setNearFarRatio(coVRConfig::instance()->nearClip()/coVRConfig::instance()->farClip());
        pBufferCamera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
        pBufferCamera->setPostDrawCallback(drawCallback.get());
        renderer->getSceneView(0)->setSceneData(cover->getScene());
        renderer->getSceneView(1)->setSceneData(cover->getScene());
    }
}

bool PBufferSnapShot::init()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::init\n");

    initUI();

    doInit = true;
    return true;
}

PBufferSnapShot::~PBufferSnapShot()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::~PBufferSnapShot\n");

    if (doSnap)
    {
        if (pBufferCamera.get())
        {
            pBufferCamera->detach(osg::Camera::COLOR_BUFFER);
            pBufferCamera->setGraphicsContext(NULL);
            VRViewer::instance()->removeCamera(pBufferCamera.get());
        }
        if (pBufferCameraR.get())
        {
            pBufferCameraR->detach(osg::Camera::COLOR_BUFFER);
            pBufferCameraR->setGraphicsContext(NULL);
            VRViewer::instance()->removeCamera(pBufferCameraR.get());
        }
    }

    deleteUI();
    cover->getMenu()->remove(snapButton);
}

void PBufferSnapShot::preSwapBuffers(int windowNumber)
{
    if (windowNumber > 0)
        return;

    if (doInit && coVRConfig::instance()->numWindows()>0)
    {
        int maxVP[2] = { 3840, 2160 }, maxRB = 38400;
#ifdef GL_MAX_VIEWPORT_DIMS
        glGetIntegerv(GL_MAX_VIEWPORT_DIMS, maxVP);
#endif
#ifdef GL_MAX_RENDERBUFFER_SIZE
        glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRB);
#endif
        std::cerr << "max viewport: " << maxVP[0] << "x" << maxVP[1] << ", render buffer: " << maxRB << std::endl;
        if (maxVP[0] > maxRB)
            maxVP[0] = maxRB;
        if (maxVP[1] > maxRB)
            maxVP[1] = maxRB;

        resolutions[NumResolutions - 1].x = maxVP[0];
        resolutions[NumResolutions - 1].y = maxVP[1];

        const int sx = coVRConfig::instance()->windows[0].sx;
        const int sy = coVRConfig::instance()->windows[0].sy;
        resolutions[1].x = sx;
        resolutions[1].y = sy;

        const double aspect = (double)sx / sy;
        if (aspect >= 1.)
        {
            resolutions[NumResolutions - 2].x = maxVP[0];
            resolutions[NumResolutions - 2].y = maxVP[0] / aspect;
        }
        else
        {
            resolutions[NumResolutions - 2].x = maxVP[1] * aspect;
            resolutions[NumResolutions - 2].y = maxVP[1];
        }
    }

    if (doInit)
    {
        doInit = false;
    }
}

void PBufferSnapShot::preFrame()
{
    if(lastSavedFile.length()>0) // if we recently saved a file to disk, inform other plugins such as the Office Plugin about it
    {
        TokenBuffer tb;
        tb << lastSavedFile.c_str();
        cover->sendMessage((coVRPlugin *)this, coVRPluginSupport::TO_ALL, PluginMessageTypes::PBufferDoneSnapshot, tb.getData().length(), tb.getData().data());
        lastSavedFile.clear();
    }
    //cerr << "PBufferSnapShot::preFrame info: called" << endl;
    if (removeCamera)
    {
        removeCamera = false;
        pBufferCamera->setGraphicsContext(NULL);
        VRViewer::instance()->removeCamera(pBufferCamera.get());
        pBufferCamera = NULL;
        if (pBufferCameraR.get())
        {
            pBufferCameraR->setGraphicsContext(NULL);
            VRViewer::instance()->removeCamera(pBufferCameraR.get());
            pBufferCameraR = NULL;
        }
    }

    if (myDoSnap)
    {
        doSnap = true;
        myDoSnap = false;
    }

    if (doSnap)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "\n--- PBufferSnapShot::preFrame doSnap\n");
        prepareSnapshot();
        filename = tuiFileName->getText();
        if(counter > 0)
        {
            ostringstream str;
            str.width(7);
            str.fill('0');
            str << counter;
            std::string::size_type suffixPos = filename.rfind('.');
            if (suffixPos != filename.npos)
            {
                filename.insert(suffixPos, str.str());
                //fprintf(stderr,"counter=%d\n", counter);
            }
            else
            {
                filename += str.str();
            }
        }

        if (tuiSnapOnSlaves->getState() && coVRMSController::instance()->getNumSlaves() > 0)
        {
            std::stringstream str;
            str << "s" << coVRMSController::instance()->getID() << "-" << filename;
            filename = str.str();
        }

        if (cerr.bad())
            cerr.clear();

        if (cover->debugLevel(3))
        {
            cerr << "PBufferSnapShot::preFrame info: snapping (0, 0, "
                 << (int)tuiResolutionX->getValue() << ", " << (int)tuiResolutionY->getValue() << ")"
                 << " to " << filename << std::endl;
        }
    }
}

void PBufferSnapShot::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::menuEvent\n");

    if (menuItem == snapButton)
    {
        //prepareSnapshot();
        doSnap = true;
    }
}

void PBufferSnapShot::tabletEvent(coTUIElement *tUIItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::tabletEvent\n");

    if(tUIItem == tuiFileName)
    {   
        counter=0;
    }
    if (tUIItem == tuiResolution)
    {
        int sel = tuiResolution->getSelectedEntry();
        tuiResolutionLabel->setLabel(resolutions[sel].description.c_str());
        if (sel > 1)
        {
            tuiResolutionX->setValue(resolutions[sel].x);
            tuiResolutionY->setValue(resolutions[sel].y);
        }

        if (sel == 1)
        {
            if (coVRConfig::instance()->numWindows() > 0)
            {
                tuiResolutionX->setValue(coVRConfig::instance()->windows[0].sx);
                tuiResolutionY->setValue(coVRConfig::instance()->windows[0].sy);
            }
        }
    }

    if (tUIItem == tuiResolutionX)
    {
        tuiResolution->setSelectedEntry(0);
        tuiResolutionLabel->setLabel(resolutions[0].description.c_str());
    }

    if (tUIItem == tuiResolutionY)
    {
        tuiResolution->setSelectedEntry(0);
        tuiResolutionLabel->setLabel(resolutions[0].description.c_str());
    }

    if (tUIItem == tuiRenderingMethod)
    {
    }
}

void PBufferSnapShot::tabletPressEvent(coTUIElement *tUIItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::tabletPressEvent\n");

    if (tUIItem == tuiSnapButton)
    {
        //prepareSnapshot();
        doSnap = true;
    }
}

void PBufferSnapShot::tabletReleaseEvent(coTUIElement *tUIItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::tabletReleaseEvent\n");
}

//fprintf(stderr,"doSnap=true - writeImageFile\n");
//else
//   fprintf(stderr,"doSnap=false - don't write image\n");

void PBufferSnapShot::cameraCallbackExit() const
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::cameraCallbackExit\n");
    
    ++counter;
    doSnap = false;
    cover->sendMessage((coVRPlugin *)this, coVRPluginSupport::TO_ALL, 0, 15, "stoppingCapture");

    removeCamera = true;

    {
        TokenBuffer tb;

        const char *fullpath = coDirectory::canonical(filename.c_str());

        tb << fullpath;

        /*
      cover->sendMessage( const_cast<PBufferSnapShot*>(this),
            "ACInterface",
            PluginMessageTypes::HLRS_ACInterfaceSnapshotPath,
            tb.get_length(),
            tb.get_data());
      tb.delete_data();
   */

        delete[] fullpath;
    }
}

void PBufferSnapShot::initUI()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::initUI\n");

    snapButton = new coButtonMenuItem("Snapshot");
    snapButton->setMenuListener(this);

    cover->getMenu()->add(snapButton);

    tuiSnapTab = new coTUITab("Snapshot", coVRTui::instance()->mainFolder->getID());
    tuiSnapTab->setPos(0, 0);

    (new coTUILabel("Resolution", tuiSnapTab->getID()))->setPos(0, 0);

    tuiResolution = new coTUIComboBox("Rendering", tuiSnapTab->getID());
    tuiResolution->setEventListener(this);
    tuiResolution->setPos(1, 0);
    for (int i=0; i<NumResolutions; ++i) {
        tuiResolution->addEntry(resolutions[i].description);
        tuiResolution->setSelectedEntry(1); // native
    }

    tuiResolutionX = new coTUIEditIntField("Resolution X", tuiSnapTab->getID());
    tuiResolutionX->setValue(resolutions[1].x);
    tuiResolutionX->setEventListener(this);
    tuiResolutionX->setImmediate(true);
    tuiResolutionX->setMin(0);
    tuiResolutionX->setMax(PB_RESOLUTION_MAX_X);
    tuiResolutionX->setPos(2, 0);
    if (coVRConfig::instance()->numViewports() > 0)
    {
        float vpx = coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin;
        //fprintf(stderr,"vpx=%f\n", vpx);
        tuiResolutionX->setValue(int(coVRConfig::instance()->windows[0].sx * vpx));
    }

    tuiResolutionY = new coTUIEditIntField("Resolution Y", tuiSnapTab->getID());
    tuiResolutionY->setValue(resolutions[1].y);
    tuiResolutionY->setEventListener(this);
    tuiResolutionY->setImmediate(true);
    tuiResolutionY->setMin(0);
    tuiResolutionY->setMax(PB_RESOLUTION_MAX_Y);
    tuiResolutionY->setPos(3, 0);
    if (coVRConfig::instance()->numViewports() > 0)
    {
        float vpy = coVRConfig::instance()->viewports[0].viewportYMax - coVRConfig::instance()->viewports[0].viewportYMin;
        //fprintf(stderr,"vpy=%f\n", vpy);
        tuiResolutionY->setValue(int(coVRConfig::instance()->windows[0].sy * vpy));
    }

    tuiResolutionLabel = new coTUILabel(resolutions[1].description.c_str(), tuiSnapTab->getID());
    tuiResolutionLabel->setPos(1, 1);

    tuiFileNameLabel = new coTUILabel("Filename", tuiSnapTab->getID());
    tuiFileNameLabel->setPos(0, 2);

    tuiFileName = new coTUIEditField("Filename", tuiSnapTab->getID());
    tuiFileName->setEventListener(this);
    tuiFileName->setText("snapshot.png");
    tuiFileName->setPos(1, 2);

    tuiSavedFileLabel = new coTUILabel("saved as:", tuiSnapTab->getID());
    tuiSavedFileLabel->setPos(2, 2);
    
    tuiSavedFile = new coTUILabel("", tuiSnapTab->getID());
    tuiSavedFile->setPos(3, 2);

    tuiSnapButton = new coTUIButton("Do it!", tuiSnapTab->getID());
    tuiSnapButton->setEventListener(this);
    tuiSnapButton->setPos(0, 4);

    tuiRenderingMethod = new coTUIComboBox("Rendering", tuiSnapTab->getID());
    tuiRenderingMethod->setEventListener(this);
    tuiRenderingMethod->setPos(0, 5);
    tuiRenderingMethod->addEntry("FRAME_BUFFER_OBJECT");
    tuiRenderingMethod->addEntry("PIXEL_BUFFER_RTT");
    tuiRenderingMethod->addEntry("PIXEL_BUFFER");
    tuiRenderingMethod->addEntry("FRAME_BUFFER");
    tuiRenderingMethod->addEntry("SEPERATE_WINDOW");
    tuiRenderingMethod->setSelectedEntry(0);

    tuiStereoCheckbox = new coTUIToggleButton("Stereo", tuiSnapTab->getID());
    tuiStereoCheckbox->setEventListener(this);
    tuiStereoCheckbox->setPos(0, 6);
    tuiStereoCheckbox->setState(false);

    tuiSnapOnSlaves = new coTUIToggleButton("SnapOnSlaves", tuiSnapTab->getID());
    tuiSnapOnSlaves->setEventListener(this);
    tuiSnapOnSlaves->setPos(0, 7);
    tuiSnapOnSlaves->setState(false);

    tuiTransparentBackground = new coTUIToggleButton("TransparentBackground", tuiSnapTab->getID());
    tuiTransparentBackground->setEventListener(this);
    tuiTransparentBackground->setPos(0, 8);
    tuiTransparentBackground->setState(true);
}

void PBufferSnapShot::deleteUI()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- PBufferSnapShot::initUI\n");

    delete tuiResolution;
    delete tuiResolutionX;
    delete tuiResolutionY;
    delete tuiResolutionLabel;
    delete tuiFileNameLabel;
    delete tuiFileName;
    delete tuiSnapButton;
    delete tuiRenderingMethod;

    delete tuiSnapTab;
}

void PBufferSnapShot::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin PBufferSnapShot guiToRenderMsg msg=[%s]\n", msg.getString().c_str());

    if (msg.isValid() && msg.getType() == coGRMsg::SNAPSHOT)
    {
        auto &snapshotMsg = msg.as<coGRSnapshotMsg>();
        // fprintf(stderr,"\n--- snapshotMsg PBufferSnapShot coVRGuiToRenderMsg msg=[%s]", msg);

        if (strcmp(snapshotMsg.getIntention(), "snapOnce") == 0)
        {
            snapID++;
            tuiFileName->setText(suggestFileName(snapshotMsg.getFilename()).c_str());
            //prepareSnapshot();
            cover->sendMessage(this, coVRPluginSupport::TO_ALL, 0, 15, "startingCapture");
            doSnap = true;
        }

        if (strcmp(snapshotMsg.getIntention(), "snapAll") == 0)
        {
            doSnap = !doSnap;
            if (!doSnap)
            {
                cover->sendMessage(this, coVRPluginSupport::TO_ALL, 0, 15, "stoppingCapture");
                counter = 0;
            }
            else
            {
                snapID++;
                cover->sendMessage(this, coVRPluginSupport::TO_ALL, 0, 15, "startingCapture");
                tuiFileName->setText(suggestFileName(snapshotMsg.getFilename()).c_str());
            }
        }
        fprintf(stderr, "Snapshot saved to %s intention=%s\n", tuiFileName->getText().c_str(), snapshotMsg.getIntention());
    }
}

string PBufferSnapShot::suggestFileName(string suggestedFilename)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin PBufferSnapShot suggestFileName\n");

    string directory = suggestedFilename;

    if (directory.empty())
    {
        directory = coCoviseConfig::getEntry("COVER.Plugin.PBufferSnapShot.Directory");
    }
    else
    {
        if (endsWith(suggestedFilename, ".png"))
        {
            return suggestedFilename;
        }
    }

    // if directory doesnt exist, set default /var/tmp
    QDir dir(directory.c_str());
    if (!dir.exists())
    {
        if (QDir::temp().exists())
            directory = QDir::temp().absolutePath().toStdString();
        else
            directory = "/var/tmp";
        dir.setCurrent(directory.c_str());
    }

    // look for all file names fitting this pattern:
    // snap[0-9]+_[0-9]{3}.+\.rgb
    // snap_c[0-9]*_[0-9]*_[0-9]{3}\.rgb
    // and then extract the last 3 digits
    const int numOfDigits = 3;
    int next = 0; // next number for a snap file
    QString reg_exp;
    reg_exp = "snap";
    QString idNumber = QString("%1").arg(snapID);
    reg_exp += idNumber; //snapID als string
    reg_exp += "_\\d+"; //beliebige Anzahl von [0-9] (mindestens einmal)
    reg_exp += "\\.png";
    QStringList entries;
    entries = dir.entryList(QStringList("*.png"));
    QStringList::Iterator it = entries.begin();

    while (it != entries.end())
    {
        QRegularExpression regexp(reg_exp);
        auto regexpMatch = regexp.match(*it);

        if (regexpMatch.capturedStart() != 0)
        {
            ++it;
            continue;
        }
        // get the 2nd number and calculate its value
        // get also the number of digits
        int pos = 0;
        QRegularExpression number("\\d+");
        auto numberMatch = number.match(*it);
        pos = numberMatch.capturedStart(1);

        int thisLength = numberMatch.capturedLength(1);
        if (thisLength != numOfDigits)
        {
            ++it;
            continue;
        }
        // pos is the position of the 2nd number
        // now look for the 1st non-zoro digit
        QRegularExpression nonzeronumber("[1-9]");
        auto nonzeronumberMatch = nonzeronumber.match(*it, pos);
        int posNonZero = nonzeronumberMatch.capturedStart();
        if (posNonZero >= 0)
        {
            thisLength -= posNonZero - pos;
            QString EntryName(*it);
            QString thisNumber = EntryName.mid(posNonZero, thisLength);
            int this_number = thisNumber.toInt();
            if (this_number + 1 > next)
            {
                next = this_number + 1;
            }
        }
        else
        {
            next = 1;
        }
        it++;
    }

    // OK now, next ensures we are not reusing a name
    if (directory[directory.length() - 1] != '/')
    {
        directory += '/';
    }

    string fileName("snap");
    QString fileNumbers = QString("%1_%2").arg(snapID).arg(QString("%1").arg(next).rightJustified(numOfDigits, '0'));

    fileName += fileNumbers.toUtf8().constData();
    fileName += ".png";

    directory += fileName;

    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin PBufferSnapShot suggestFileName %s\n", directory.c_str());

    return directory;
}

void PBufferSnapShot::message(int toWhom, int type, int len, const void *buf)
{

    switch (type)
    {
    case PluginMessageTypes::PBufferDoSnap:
    {
        TokenBuffer tb{ covise::DataHandle{(char*)buf, len, false} };
        std::string path;

        tb >> path;

        snapID++;
        tuiFileName->setText(suggestFileName(path).c_str());
        prepareSnapshot();
        myDoSnap = true;
        fprintf(stderr, "Snapshot saved to %s\n", tuiFileName->getText().c_str());
    }
    case PluginMessageTypes::PBufferDoSnapFile:
    {
        TokenBuffer tb{ covise::DataHandle{(char*)buf, len, false} };
        std::string file;
        tb >> file;
        snapID++;
        tuiFileName->setText(file.c_str());
        prepareSnapshot();
        myDoSnap = true;
        fprintf(stderr, "Snapshot saved to %s\n", tuiFileName->getText().c_str());
    }
    default:
        break;
    }
}

bool PBufferSnapShot::endsWith(std::string main, std::string end)
{
    boost::algorithm::to_lower(main);
    boost::algorithm::to_lower(end);
    return boost::algorithm::ends_with(main, end);
}

COVERPLUGIN(PBufferSnapShot)
