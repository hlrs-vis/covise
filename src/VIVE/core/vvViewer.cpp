/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <util/common.h>
#include <util/unixcompat.h>


#include <config/CoviseConfig.h>
#include "vvViewer.h"
#include "vvWindows.h"
#include "vvVIVE.h"
#include "vvMathUtils.h"
#include <OpenVRUI/vsg/mathUtils.h>
#include "vvPluginList.h"
#include "vvPluginSupport.h"
#include "vvConfig.h"
/*#include "coCullVisitor.h"
#include "InitGLOperation.h"*/
#include "vvMarkerTracking.h"
#include "input/input.h"
#include "tridelity.h"

#include "ui/Button.h"
#include "ui/SelectionList.h"
#include "ui/Menu.h"



#include "vvMSController.h"
#include <vsg/lighting/Light.h>
#include <vsg/io/read.h>
#include <vsg/app/RenderGraph.h>
#include <vsg/app/Trackball.h>
#include <vsg/app/CloseHandler.h>
#include <vsg/utils/ComputeBounds.h>

#ifndef _WIN32
#include <termios.h>
#include <unistd.h>
#endif

#define ANIMATIONSPEED 1


#include <cassert>

#ifdef __linux
#include <unistd.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#endif


using namespace covise;

namespace vive {


int vvViewer::unsyncedFrames = 0;

// sync Operation which syncs slaves to master in a cluster configuration
struct RemoteSyncOp : public vsg::Inherit<vsg::Operation, RemoteSyncOp>
{
    RemoteSyncOp()
    {
    }

    virtual void operator()(vsg::Object *object)
    {

        //do sync here before swapBuffers

        // sync multipc instances
        if (vvViewer::unsyncedFrames <= 0)
        {
            vvMSController::instance()->barrierDraw();
        }
        else
            vvViewer::unsyncedFrames--;
    }
};

// call plugins before syncing
struct PreSwapBuffersOp : public vsg::Inherit<vsg::Operation, PreSwapBuffersOp>
{
    PreSwapBuffersOp(int WindowNumber)
    {
        windowNumber = WindowNumber;
    }

    virtual void operator()(vsg::Object *object)
    {

        if (vvVIVE::instance()->initDone())
            vvPluginList::instance()->preSwapBuffers(windowNumber);
    }
    int windowNumber;
};

// call plugins after syncing
struct PostSwapBuffersOp : public vsg::Inherit<vsg::Operation, PostSwapBuffersOp>
{
    PostSwapBuffersOp(int WindowNumber)
    {
        windowNumber = WindowNumber;
    }

    virtual void operator()(vsg::Object *object)
    {
        if (vvVIVE::instance()->initDone())
            vvPluginList::instance()->postSwapBuffers(windowNumber);
    }
    int windowNumber;
};


//OpenCOVER
void vvViewer::vvUpdate()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "vvViewer::update\n");

    if (animateSeparation)
    {
        if (animateSeparation == 1)
        {
            separation += ANIMATIONSPEED;
            if (separation >= requestedSeparation)
            {
                separation = requestedSeparation;
                animateSeparation = 0;
            }
        }
        else if (animateSeparation == 2)
        {
            separation -= ANIMATIONSPEED;
            if (separation <= requestedSeparation)
            {
                separation = requestedSeparation;
                animateSeparation = 0;
            }
        }
    }
    
    if (arTracking)
    {
        if (vpMarker && vpMarker->isVisible())
        {
            viewMat = vpMarker->getCameraTrans();
        }
        //again = true;
    }

    // compute viewer position

    viewPos = getTrans(viewMat);


    if (vv->debugLevel(5))
    {
        fprintf(stderr, "\t viewPos=[%f %f %f]\n", viewPos[0], viewPos[1], viewPos[2]);
        fprintf(stderr, "\n");
    }

    for (int i = 0; i < vvConfig::instance()->numChannels(); i++)
    {
        setFrustumAndView(i);
    }


    //viewer->compile();
    if (advanceToNextFrame())
    {
        // pass any events into EventHandlers assigned to the Viewer
        handleEvents();

        update();

        recordAndSubmit();

        present();
    }
    else
    {
        vvVIVE::instance()->requestQuit();
    }
}

//////////////////////////////////////////////////////////////////////////////
//
// vvViewer::Viewer implemention
//

vsg::ref_ptr<vvViewer> vvViewer::s_singleton;
vvViewer *vvViewer::instance()
{
    if (!s_singleton.get())
        s_singleton = new vvViewer;
    return s_singleton;
}

void vvViewer::destroy()
{
    s_singleton = nullptr;
}

//OpenCOVER
vvViewer::vvViewer()
: ui::Owner("vvViewer", vv->ui)
, animateSeparation(0)
, stereoOn(true)
{
    assert(!s_singleton);

    if (vv->debugLevel(2))
        fprintf(stderr, "\nnew vvViewer\n");
    reEnableCulling = false;

    unsyncedFrames = 0;
    lastFrameTime = 0.0;


    requestedSeparation = 0;

    // automatic angle
    screen_angle = NULL;

    // clear once at startup
    clearWindow = true;

    readConfigFile();

    viewDir.set(0.0f, 0.0f, 0.0f);

    //initial view position
    float xp = coCoviseConfig::getFloat("x", "VIVE.ViewerPosition", 0.0f);
    float yp = coCoviseConfig::getFloat("y", "VIVE.ViewerPosition", -2000.0f);
    float zp = coCoviseConfig::getFloat("z", "VIVE.ViewerPosition", 30.0f);
    viewPos.set(xp, yp, zp);

    initialViewPos = viewPos;
    // set initial view
    viewMat= vsg::translate(viewPos);

    stereoOn = vvConfig::instance()->stereoState();

    //separation = stereoOn ? Input::instance()->eyeDistance() : 0.;
    arTracking = false;
    if (coCoviseConfig::isOn("VIVE.MarkerTracking.TrackViewpoint", false))
    {
        arTracking = true;
       // vpMarker = MarkerTracking::instance()->getMarker("ViewpointMarker");
    }
    overwritePAndV = false;
    
    auto stereosep = new ui::Button("StereoSep", this);
    vv->viewOptionsMenu->add(stereosep);
    stereosep->setText("Stereo separation");
    stereosep->setState(stereoOn);
    stereosep->setCallback([this](bool state){
        stereoOn = state;
        setSeparation(Input::instance()->eyeDistance());
    });

    auto ortho = new ui::Button("Orthographic", this);
    vv->viewOptionsMenu->add(ortho);
    ortho->setText("Orthographic projection");
    ortho->setState(vvConfig::instance()->orthographic());
    ortho->setCallback([this](bool state){
        vvConfig::instance()->setOrthographic(state);
        //requestRedraw();
    });
    /*
    auto threading = new ui::SelectionList("ThreadingModel", this);
    vv->viewOptionsMenu->add(threading);
    threading->setText("Threading model");
    std::vector<std::string> items{"Auto", "Single", "Cull/draw per context", "Draw per context", "Cull per camera, draw per context"};
    threading->setList(items);
    auto model = getThreadingModel();
    switch (model) {
    case AutomaticSelection: threading->select(0); break;
    case SingleThreaded: threading->select(1); break;
    case CullDrawThreadPerContext: threading->select(2); break;
    case DrawThreadPerContext: threading->select(3); break;
    case CullThreadPerCameraDrawThreadPerContext: threading->select(4); break;
    default: threading->select(0); break;
    }
    threading->setCallback([this](int val) {
        switch (val) {
        case 0: setThreadingModel(AutomaticSelection); break;
        case 1: setThreadingModel(SingleThreaded); break;
        case 2: setThreadingModel(CullDrawThreadPerContext); break;
        case 3: setThreadingModel(DrawThreadPerContext); break;
        case 4: setThreadingModel(CullThreadPerCameraDrawThreadPerContext); break;
        default: setThreadingModel(AutomaticSelection); break;
        }
        if (statsDisplay)
           statsDisplay->updateThreadingModelText(getThreadingModel());
    });
    threading->setShortcut("Alt+h");
    threading->addShortcut("Ctrl+h");

    auto showStats = new ui::SelectionList("ShowStats", this);
    vv->viewOptionsMenu->add(showStats);
    showStats->setText("Renderer statistics");
    showStats->setShortcut("Shift+S");
    showStats->append("Off");
    showStats->append("Frames/s");
    showStats->append("Viewer");
    showStats->append("Viewer+camera");
    showStats->append("Viewer+camera+nodes");
    vv->viewOptionsMenu->add(showStats);
    showStats->select(vvConfig::instance()->drawStatistics);
    showStats->setCallback([this](int val){
        vvConfig::instance()->drawStatistics = val;
        statsDisplay->showStats(val, vvViewer::instance());
        //XXX setInstrumentationMode( vvConfig::instance()->drawStatistics );
    });
    
    statsDisplay = new vvStatsDisplay();*/
}


vvViewer::~vvViewer()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\ndelete vvViewer\n");


    unconfig();

#ifndef _WIN32
    if (strlen(monoCommand) > 0)
    {
        if (system(monoCommand) == -1)
        {
            cerr << "exec " << monoCommand << " failed" << endl;
        }
    }
#endif


    //s_singleton = NULL;
}



void vvViewer::setAffinity()
{
#ifdef __linux__
    const int ncpu = sysconf(_SC_NPROCESSORS_ONLN);

    cpu_set_t cpumask;
    CPU_ZERO(&cpumask);

    if(vvConfig::instance()->lockToCPU()>=0)
    {
        CPU_SET(vvConfig::instance()->lockToCPU(), &cpumask);

        std::cerr << "locking to CPU " << vvConfig::instance()->lockToCPU() << std::endl;
    }
    else
    {
        for (int i = 0; i < CPU_SETSIZE && i < ncpu; i++)
        {
            CPU_SET(i, &cpumask);
        }
    }
    int err = pthread_setaffinity_np(pthread_self(), sizeof(cpumask), &cpumask);
    if (err != 0)
    {
        std::cerr << "setting affinity to " << CPU_COUNT(&cpumask) << " CPUs failed: " << strerror(err) <<  std::endl;
    }
//#elif defined(HAVE_THREE_PARAM_SCHED_SETAFFINITY)
//           sched_setaffinity( 0, sizeof(cpumask), &cpumask );
//#elif defined(HAVE_TWO_PARAM_SCHED_SETAFFINITY)
//            sched_setaffinity( 0, &cpumask );
#endif
}

void
vvViewer::unconfig()
{

    auto &conf = *vvConfig::instance();
    for (int i = 0; i < conf.numChannels(); i++)
    {
        destroyChannels(i);
    }
    for (int i =0; i < conf.numPBOs(); ++i) {
        conf.PBOs[i].renderTargetTexture = nullptr;
    }

    myCameras.clear();
}

void
vvViewer::config()
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvViewer::config\n");

    for (int i = 0; i < vvConfig::instance()->numChannels(); i++)
    {
        createChannels(i);
    }

    assignRecordAndSubmitTaskAndPresentation(commandGraphs);

    float r = coCoviseConfig::getFloat("r", "VIVE.Background", 0.0f);
    float g = coCoviseConfig::getFloat("g", "VIVE.Background", 0.0f);
    float b = coCoviseConfig::getFloat("b", "VIVE.Background", 0.0f);
    backgroundColor = vsg::vec4(r, g, b, 1.0f);

    setClearColor(backgroundColor);

    // create the windows and run the threads.
    if (coCoviseConfig::isOn("VIVE.MultiThreaded", false))
    {
        cerr << "vvViewer: using one thread per camera" << endl;
    }
    else
    {
        std::string threadingMode = coCoviseConfig::getEntry("VIVE.MultiThreaded");
        /*if (threadingMode == "CullDrawThreadPerContext")
        {
            setThreadingModel(CullDrawThreadPerContext);
        }
        else if (threadingMode == "ThreadPerContext")
        {
            setThreadingModel(ThreadPerContext);
        }
        else if (threadingMode == "DrawThreadPerContext")
        {
            setThreadingModel(DrawThreadPerContext);
        }
        else if (threadingMode == "CullThreadPerCameraDrawThreadPerContext")
        {
            setThreadingModel(CullThreadPerCameraDrawThreadPerContext);
        }
        else if (threadingMode == "ThreadPerCamera")
        {
            setThreadingModel(ThreadPerCamera);
        }
        else if (threadingMode == "AutomaticSelection")
        {
            setThreadingModel(AutomaticSelection);
        }
        else
            setThreadingModel(SingleThreaded);*/
    }

    //if (mustDraw())
       // realize();

    setAffinity();

    for (int i = 0; i < vvConfig::instance()->numChannels(); i++)
    {
        setFrustumAndView(i);
    }



    bool syncToVBlankConfigured = false;
    bool syncToVBlank = covise::coCoviseConfig::isOn("VIVE.SyncToVBlank", false, &syncToVBlankConfigured);
    for (size_t i=0; i<vvConfig::instance()->numWindows(); ++i)
    {
        if (syncToVBlankConfigured)
        {
            if (auto win = vvConfig::instance()->windows[i].window)
            {
                //win->setSyncToVBlank(syncToVBlank);
            }
        }
    }
}

//OpenCOVER
void vvViewer::updateViewerMat(const vsg::dmat4 &mat)
{

    if (vv->debugLevel(5))
    {
        fprintf(stderr, "vvViewer::updateViewerMat\n");
        //coCoord coord;
        //mat.getOrthoCoord(&coord);
        //fprintf(stderr,"hpr=[%f %f %f]\n", coord.hpr[0], coord.hpr[1], coord.hpr[2]);
    }

    if (!vvConfig::instance()->frozen())
    {
        viewMat = mat;
    }
}

//OpenCOVER
void
vvViewer::setSeparation(float sep)
{
    requestedSeparation = stereoOn ? sep : 0.0f;

    if (requestedSeparation > separation)
    {
        animateSeparation = 1;
    }
    else if (requestedSeparation < separation)
    {
        animateSeparation = 2;
    }
    else
    {
        separation = requestedSeparation;
        animateSeparation = 0;
    }
}

void
vvViewer::flipStereo()
{
    separation = -separation;
}

void vvViewer::setFullscreen(bool state)
{
    if (m_fullscreen != state)
    {
        unconfig();
        m_fullscreen = state;
        config();
    }
}

bool vvViewer::isFullscreen() const
{
    return m_fullscreen;
}

void
vvViewer::setRenderToTexture(bool b)
{
}

//OpenCOVER
void
vvViewer::createChannels(int i)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvViewer::createChannels\n");

#if 0
    vsg::GraphicsContext::WindowingSystemInterface *wsi = vsg::GraphicsContext::getWindowingSystemInterface();
    if (!wsi)
    {
        vsg::notify(vsg::NOTICE) << "vvViewer : Error, no WindowSystemInterface available, cannot create windows." << std::endl;
        return;
    }
#endif


    auto &conf = *vvConfig::instance();
    const int vp = vvConfig::instance()->channels[i].viewportNum;
    if (vp >= vvConfig::instance()->numViewports())
    {
        fprintf(stderr, "vvViewer:: error creating channel %d\n", i);
        fprintf(stderr, "viewportNum %d is out of range (viewports are counted starting from 0)\n", vp);
        return;
    }
    // provide a custom RenderPass
    // window->setRenderPass(createRenderPass(window->getOrCreateDevice()));
    int windowNumber = vvConfig::instance()->viewports[vp].window;
    auto& windowInfo = vvConfig::instance()->windows[windowNumber];

    auto& viewport = vvConfig::instance()->viewports[vvConfig::instance()->channels[i].viewportNum];
    uint32_t width = (viewport.viewportXMax - viewport.viewportXMin)* windowInfo.sx;
    uint32_t height = (viewport.viewportYMax - viewport.viewportYMin) * windowInfo.sy;
    
    double aspectRatio = (double)width / height;
    

    
    // create view
    // compute the bounds of the scene graph to help position camera
    vsg::ComputeBounds computeBounds;
    vv->getScene()->accept(computeBounds);
    vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
    double nearFarRatio = 0.001;



    /*auto lookAt = vsg::LookAt::create(centre + vsg::dvec3(0.0, -radius * 3.5, 0.0),
        centre, vsg::dvec3(0.0, 0.0, 1.0));*/

    vsg::dvec3 eye(0,-1500,0);
    auto lookAt = vsg::LookAt::create(eye, eye + vsg::dvec3(0, 1, 0), vsg::dvec3(0, 0, 1));
    auto perspective = OffAxis::create();
    perspective->proj = vsg::perspective(
        30.0, static_cast<double>(width) / static_cast<double>(height),
        nearFarRatio * radius, radius * 4.5);

    auto viewportstate = vsg::ViewportState::create(viewport.viewportXMin * windowInfo.sx, viewport.viewportYMin * windowInfo.sy, width, height);





    vvConfig::instance()->channels[i].camera = vsg::Camera::create(perspective, lookAt, viewportstate);
    vvConfig::instance()->channels[i].view = vsg::View::create(vvConfig::instance()->channels[i].camera);

    vvConfig::instance()->channels[i].view->addChild(vsg::createHeadlight());

   

    vvConfig::instance()->channels[i].view->addChild(vv->getScene());
    auto renderGraph = vsg::RenderGraph::create(vvConfig::instance()->windows[windowNumber].window, vvConfig::instance()->channels[i].view);
    renderGraph->addChild(vvConfig::instance()->channels[i].view);



    vvConfig::instance()->channels[i].commandGraph = vsg::CommandGraph::create(vvConfig::instance()->windows[windowNumber].window);

    vvConfig::instance()->channels[i].commandGraph->addChild(renderGraph);
    commandGraphs.push_back(vvConfig::instance()->channels[i].commandGraph);

    //addEventHandler(vsg::Trackball::create(vvConfig::instance()->channels[i].camera));



    addEventHandler(EventHandler::create(vvVIVE::instance()));
    addEventHandler(vsg::CloseHandler::create(this)); // needs to be the last event handler





}

void vvViewer::destroyChannels(int i)
{
    auto &conf = *vvConfig::instance();
    auto &chan = conf.channels[i];

    chan.camera = nullptr;
}


void
vvViewer::setClearColor(const vsg::vec4 &color)
{
    for (int i = 0; i < vvConfig::instance()->numChannels(); ++i)
    {
		//if(vvConfig::instance()->channels[i].camera!=NULL)
         //   vvConfig::instance()->channels[i].camera->setClearColor(color);
    }
}

std::pair<vsg::dmat4, vsg::dmat4>
computeViewProjFixedScreen(const vsg::dmat4 &viewerMat, vsg::dvec3 eye, const vsg::dvec3 &xyz, const vsg::dvec3 &hpr, const vsg::vec2 &size, double near_val, double far_val, bool ortho, double worldAngle)
{
    vsg::dmat4 trans;
    // transform the screen to fit the xz-plane
    trans =vsg::translate(-xyz);

    vsg::dmat4 euler= vsg::inverse_4x3(makeEulerMat(hpr));

    vsg::dmat4 mat = euler*trans*(rotate(-worldAngle,vsg::dvec3(1,0,0)));

    // transform the left and right eye with the viewer matrix
    eye =  viewerMat * eye;

    // transform the left and right eye with the screen matrix so that they are in screen relative coordinates
    eye = mat * eye;

    float dx = size[0], dz = size[1];

    auto projFromEye = [ortho, near_val, far_val, dx, dz](const vsg::dvec3 &eye) -> vsg::dmat4
    {
            double n_over_d; // near_val over dist -> Strahlensatz
            double c_dist = -eye[1];
            // relation near_val plane to screen plane
            if (ortho)
                n_over_d = 1.0;
            else
                n_over_d = near_val / c_dist;

            double c_right = n_over_d * (dx / 2.0 - eye[0]);
            double c_left = -n_over_d * (dx / 2.0 + eye[0]);
            double c_top = n_over_d * (dz / 2.0 - eye[2]);
            double c_bottom = -n_over_d * (dz / 2.0 + eye[2]);

            vsg::dmat4 ret;
            if (ortho)
                ret = vsg::orthographic(c_left, c_right, c_bottom, c_top, near_val, far_val);
            else
                ret = vsg::perspective(c_left, c_right, c_bottom, c_top, near_val, far_val);
            return ret;
    };

    vsg::dmat4 proj = projFromEye(eye);

    // take the normal to the plane as orientation this is (0,1,0)
    vsg::dmat4 view;
    view = vsg::lookAt(eye, eye+vsg::dvec3(0,1,0), vsg::dvec3(0,0,1));
    view = view*mat;

    return std::make_pair(view, proj);
}


void
vvViewer::setFrustumAndView(int i)
{
    if (vv->debugLevel(5))
        fprintf(stderr, "vvViewer::setFrustum %d\n", i);

    vvConfig *coco = vvConfig::instance();
    channelStruct *currentChannel = &coco->channels[i];
    screenStruct *currentScreen = &coco->screens[currentChannel->screenNum];

    if (overwritePAndV)
    {
        //currentChannel->camera->setViewMatrix(currentChannel->rightView);
        //currentChannel->camera->setProjectionMatrix(currentChannel->rightProj);
        return;
    }

    auto res = computeFrustumAndView(i);

    currentChannel->leftView = res.left.view;
    currentChannel->leftProj = res.left.proj;
    currentChannel->rightView = res.right.view;
    currentChannel->rightProj = res.right.proj;
    auto viewMatrix = (dynamic_cast<vsg::LookAt*>(vvConfig::instance()->channels[i].camera->viewMatrix.get()));
    auto projMatrix = (dynamic_cast<vive::OffAxis*>(vvConfig::instance()->channels[i].camera->projectionMatrix.get()));
    if(vvConfig::instance()->channels[i].stereoMode == vvConfig::LEFT_EYE)
    {
    viewMatrix->set(inverse(res.left.view));
    projMatrix->proj = res.left.proj;
    }
    else
    {
    viewMatrix->set(inverse(res.right.view));
    projMatrix->proj = res.right.proj;
    }
    // 
    //currentChannel->camera->setViewMatrix(res.middle.view);
    //currentChannel->camera->setProjectionMatrix(res.middle.proj);
}

vsg::dvec3 vvViewer::eyeOffset(vvViewer::Eye eye) const
{
    vsg::dvec3 off(0,0,0);
    if (stereoOn || animateSeparation)
    {
        switch (eye) {
        case EyeMiddle:
            break;
        case EyeLeft:
            off[0] = -separation/2.;
            break;
        case EyeRight:
            off[0] = separation/2.;
            break;
        }
    }

    return off;
}

vvViewer::FrustaAndViews vvViewer::computeFrustumAndView(int i)
{

    vvConfig *coco = vvConfig::instance();
    channelStruct *currentChannel = &coco->channels[i];
    screenStruct *currentScreen = &coco->screens[currentChannel->screenNum];

    vsg::dvec3 xyz; // center position of the screen
    vsg::dvec3 hpr; // orientation of the screen
    vsg::dmat4 mat, trans, euler; // xform screencenter - world origin
    vsg::dmat4 offsetMat;
    vsg::dvec3 leftEye, rightEye, middleEye; // transformed eye position
    float dx, dz; // size of screen

    //othEyesDirOffset; == hpr
    //vsg::dvec3  rightEyePosOffset(0.0,0.0,0.0), leftEyePosOffset(0.0,0.0,0.0);

    dx = currentScreen->hsize;
    dz = currentScreen->vsize;

    hpr = currentScreen->hpr;
    xyz = currentScreen->xyz;

    // first set pos and yaxis
    // next set separation and dir from covise.config
    // which I think workks only if 0 0 0 (dr)

    ////// IWR : get values of moving screen; change only if moved by >1%
    if (screen_angle && screen_angle[0].screen == i)
    {
        float new_angle;

        new_angle = ((*screen_angle[0].value - screen_angle[0].cmin) / (screen_angle[0].cmax - screen_angle[0].cmin)) * (screen_angle[0].maxangle - screen_angle[0].minangle) + screen_angle[0].minangle;

        // change angle only, when change is significant (> 1%)
        float change_delta = fabs(screen_angle[0].maxangle - screen_angle[0].minangle) * 0.01f;

        if (fabs(currentScreen->hpr[screen_angle[0].hpr] - new_angle) > change_delta)
        {
            currentScreen->hpr[screen_angle[0].hpr] = new_angle;
            //		   cerr << "Cereal gives " << *screen_angle[0].value << endl;
            cerr << "Setting Screen angle " << screen_angle[0].hpr << " to " << new_angle << endl;
        }
    }

    // for rendering images for auto-stereoscopic displays (e.g. Tridelity)
    const float off = (stereoOn || animateSeparation) ? currentChannel->stereoOffset : 0.f;
    if (currentChannel->stereo)
    {
        leftEye = eyeOffset(EyeLeft) + vsg::dvec3(off,0,0);
        rightEye = eyeOffset(EyeRight) + vsg::dvec3(off,0,0);
        middleEye = eyeOffset(EyeMiddle) + vsg::dvec3(off,0,0);
    }
    else
    {
        leftEye = rightEye = middleEye = eyeOffset(EyeMiddle) + vsg::dvec3(off,0,0);
    }

    if (!coco->trackedHMD && !coco->HMDMode)
    {
        vsg::dmat4 vm;
        if (currentChannel->fixedViewer)
            vm=translate(initialViewPos);
        else
            vm = viewMat;
        auto l = computeViewProjFixedScreen(vm, leftEye, xyz, hpr, vsg::vec2(dx,dz), coco->nearClip(), coco->farClip(), coco->orthographic(), coco->worldAngle());
        auto r = computeViewProjFixedScreen(vm, rightEye, xyz, hpr, vsg::vec2(dx,dz), coco->nearClip(), coco->farClip(), coco->orthographic(), coco->worldAngle());
        auto m = computeViewProjFixedScreen(vm, middleEye, xyz, hpr, vsg::vec2(dx,dz), coco->nearClip(), coco->farClip(), coco->orthographic(), coco->worldAngle());

        FrustaAndViews res;
        res.left.view = l.first;
        res.left.proj = l.second;
        res.right.view = r.first;
        res.right.proj = r.second;
        res.middle.view = m.first;
        res.middle.proj = m.second;

        return res;
    }

    if (coco->trackedHMD) // moving HMD
    {
        // moving hmd: frustum ist fixed and only a little bit assymetric through stereo
        // transform the left and right eye with the viewer matrix
        rightEye = viewMat * rightEye;
        leftEye = viewMat * leftEye;
        middleEye = viewMat * middleEye;
    }

    else if (coco->HMDMode) // weiss nicht was das fuer ein code ist
        // wenn screen center und dir 000 sind
    {

        // transform the screen to fit the xz-plane

        trans = translate(-xyz);

        euler = inverse_4x3(makeEulerMat(hpr));

        mat = euler * trans * vsg::rotate((double) - coco->worldAngle(), vsg::dvec3(1, 0, 0));


        rightEye += initialViewPos;
        leftEye += initialViewPos;
        middleEye += initialViewPos;

        // transform the left and right eye with this matrix
        rightEye = viewMat * rightEye;
        leftEye = viewMat * leftEye;
        middleEye = viewMat *middleEye;

        // add world angle
        vsg::dmat4 rotAll, newDir;
        rotAll =rotate(static_cast<double>(-coco->worldAngle()), vsg::dvec3(1, 0, 0));
        newDir= rotAll * viewMat;


        viewPos = getTrans(newDir);

    }

    offsetMat = mat;

    FrustaAndViews res;
    // compute right frustum

    // dist of right channel eye to screen (absolute)
    if (coco->trackedHMD)
    {
        if (coco->orthographic())
        {
            res.right.proj = vsg::orthographic(-dx / 2.0, dx / 2.0, -dz / 2.0, dz / 2.0, (double)coco->nearClip(), (double)coco->farClip());
            res.left.proj = vsg::orthographic(-dx / 2.0, dx / 2.0, -dz / 2.0, dz / 2.0, (double)coco->nearClip(), (double)coco->farClip());
            res.middle.proj = vsg::orthographic(-dx / 2.0, dx / 2.0, -dz / 2.0, dz / 2.0, (double)coco->nearClip(), (double)coco->farClip());
        }
        else
        {
            if (currentScreen->lTan != -1)
            {
                double n = coco->nearClip();
                res.right.proj = vsg::perspective(-n * currentScreen->lTan, n * currentScreen->rTan, -n * currentScreen->bTan, n * currentScreen->tTan, (double)coco->nearClip(), (double)coco->farClip());
                res.left.proj = vsg::perspective(-n * currentScreen->lTan, n * currentScreen->rTan, -n * currentScreen->bTan, n * currentScreen->tTan, (double)coco->nearClip(), (double)coco->farClip());
                res.middle.proj = vsg::perspective(-n * currentScreen->lTan, n * currentScreen->rTan, -n * currentScreen->bTan, n * currentScreen->tTan, (double)coco->nearClip(), (double)coco->farClip());
            }
            else
            {
				if (!coco->OpenVR_HMD)
				{
                    res.right.proj = vsg::perspective(coco->HMDViewingAngle, dx / dz, coco->nearClip(), coco->farClip());
                    res.left.proj = vsg::perspective(coco->HMDViewingAngle, dx / dz, coco->nearClip(), coco->farClip());
				}
                res.middle.proj = vsg::perspective(coco->HMDViewingAngle, dx / dz, coco->nearClip(), coco->farClip());
            }
        }
    }
    else
    {
        auto projFromEye = [coco, dx, dz](const vsg::dvec3 &eye) -> vsg::dmat4 {
            double n_over_d; // near over dist -> Strahlensatz
            double c_dist = -eye[1];
            // relation near plane to screen plane
            if (coco->orthographic())
                n_over_d = 1.0;
            else
                n_over_d = coco->nearClip() / c_dist;

            double c_right = n_over_d * (dx / 2.0 - eye[0]);
            double c_left = -n_over_d * (dx / 2.0 + eye[0]);
            double c_top = n_over_d * (dz / 2.0 - eye[2]);
            double c_bottom = -n_over_d * (dz / 2.0 + eye[2]);

            vsg::dmat4 ret;
            if (coco->orthographic())
                ret = vsg::orthographic(c_left, c_right, c_bottom, c_top, (double)coco->nearClip(), (double)coco->farClip());
            else
                ret= vsg::perspective(c_left, c_right, c_bottom, c_top, (double)coco->nearClip(), (double)coco->farClip());
            return ret;
        };

        res.right.proj = projFromEye(rightEye);
        res.left.proj = projFromEye(leftEye);
        res.middle.proj = projFromEye(middleEye);
    }

    // set view
    if (coco->trackedHMD)
    {
        // set the view mat from tracker, translated by separation/2
        vsg::dvec3 viewDir(0, 1, 0), viewUp(0, 0, 1);
        vsg::dmat3 viewMat3(viewMat[0][0], viewMat[0][1], viewMat[0][2],
                            viewMat[1][0], viewMat[1][1], viewMat[1][2],
                            viewMat[2][0], viewMat[2][1], viewMat[2][2]);
        viewDir = viewMat3 * viewDir;
        viewDir = vsg::normalize(viewDir);

        viewUp = viewMat3 * viewUp;
        viewUp = vsg::normalize(viewUp);

        res.right.view = vsg::lookAt(rightEye, rightEye + viewDir, viewUp);
        ///currentScreen->rightView=viewMat;
        res.left.view = vsg::lookAt(leftEye, leftEye + viewDir, viewUp);
        ///currentScreen->leftView=viewMat;

        res.middle.view = vsg::lookAt(middleEye, middleEye + viewDir, viewUp);
        ///currentScreen->camera->setViewMatrix(viewMat);
    }
    else
    {
        // take the normal to the plane as orientation this is (0,1,0)
        res.right.view = vsg::lookAt(vsg::dvec3(rightEye[0], rightEye[1], rightEye[2]), vsg::dvec3(rightEye[0], rightEye[1] + 1, rightEye[2]), vsg::dvec3(0, 0, 1)) * offsetMat;

        res.left.view = vsg::lookAt(vsg::dvec3(leftEye[0], leftEye[1], leftEye[2]), vsg::dvec3(leftEye[0], leftEye[1] + 1, leftEye[2]), vsg::dvec3(0, 0, 1)) * offsetMat;

        res.middle.view = vsg::lookAt(vsg::dvec3(middleEye[0], middleEye[1], middleEye[2]), vsg::dvec3(middleEye[0], middleEye[1] + 1, middleEye[2]), vsg::dvec3(0, 0, 1)) * offsetMat;
    }

    return res;
}

bool vvViewer::compileNode(vsg::ref_ptr<vsg::Node>& node)
{
    if(InitialCompileDone)
    {
        auto result = compileManager->compile(node);
        vsg::updateViewer(*this, result);
        return (bool)result;
    }
    return false;
}

void vvViewer::InitialCompile()
{
    compile();
    InitialCompileDone = true;
}

//OpenCOVER
void
vvViewer::readConfigFile()
{
    if (vv->debugLevel(4))
        fprintf(stderr, "vvViewer::readConfigFile\n");

    /// ================= Moving screen initialisation =================

    // temp initialisation until the real address is determin in
    // the tracker routines
    static float init_cereal_analog_value = 0.0;

    screen_angle = NULL;
    string line = coCoviseConfig::getEntry("analogInput", "VIVE.Input.CerealConfig.ScreenAngle");
    if (!line.empty())
    {
        char hpr;
        cerr << "Reading ScreenAngle";
        screen_angle = new angleStruct;
        screen_angle->value = &init_cereal_analog_value;

        // parameters:  Analog Input, min, max, minangle, maxangle, screen, HPR
        //if(sscanf(line,   "%d %f %f %f %f %d %c",

        const char *configEntry = "VIVE.Input.CerealConfig.ScreenAngle";

        screen_angle->analogInput = coCoviseConfig::getInt("analogInput", configEntry, 0);
        screen_angle->cmin = coCoviseConfig::getFloat("cmin", configEntry, 0.0f);
        screen_angle->cmax = coCoviseConfig::getFloat("cmax", configEntry, 0.0f);
        screen_angle->minangle = coCoviseConfig::getFloat("minangle", configEntry, 0.0f);
        screen_angle->maxangle = coCoviseConfig::getFloat("maxangle", configEntry, 0.0f);
        screen_angle->screen = coCoviseConfig::getInt("screen", configEntry, 0);
        hpr = coCoviseConfig::getEntry("hpr", configEntry)[0];

        *screen_angle->value = screen_angle->cmin;
        cerr << "\n  - Using analog Input #" << screen_angle->analogInput
             << "\n  cMin/cMax " << screen_angle->cmin
             << " ... " << screen_angle->cmax
             << " " << screen_angle->minangle << " " << screen_angle->maxangle << " " << screen_angle->screen << " " << hpr << endl;
        switch (hpr)
        {
        case 'H':
        case 'h':
            screen_angle->hpr = 0;
            break;
        case 'P':
        case 'p':
            screen_angle->hpr = 1;
            break;
        case 'R':
        case 'r':
            screen_angle->hpr = 2;
            break;
        default:
            cerr << "could not identify rotation axis\nIgnoring ...\n";
            screen_angle = NULL;
        }
    }

    /// ================= Moving screen initialisation End =================

    strcpy(stereoCommand, "");
    strcpy(monoCommand, "");
    line = coCoviseConfig::getEntry("command", "VIVE.Stereo");
    if (!line.empty())
    {
        strcpy(stereoCommand, line.c_str());
    }
    line = coCoviseConfig::getEntry("command", "VIVE.Mono");
    if (!line.empty())
    {
        strcpy(monoCommand, line.c_str());
    }
    if (vvConfig::instance()->stereoState())
    {
        if (strlen(stereoCommand) > 0)
            if (system(stereoCommand) == -1)
            {
                cerr << "vvViewer::readConfigFile: stereo command " << stereoCommand << " failed" << endl;
            }
    }
}

// OpenCOVER
void vvViewer::redrawHUD(double interval)
{
    // do a redraw of the scene no more then once per second (default)
    if (vv->currentTime() > lastFrameTime + interval)
    {
        unsyncedFrames++; // do not wait for slaves during temporary updates of the headup display
        //frame(); // draw one frame
        vvWindows::instance()->update();
        vvWindows::instance()->updateContents();
    }
}

void vvViewer::disableSync()
{
    unsyncedFrames = 1000000;
}

bool vvViewer::mustDraw()
{
    return vvConfig::instance()->numChannels() > 0;
}
/*
// OpenCOVER
void vvViewer::frame(double simulationTime)
{

    lastFrameTime = vv->currentTime();

    double startFrame = elapsedTime();
    if (mustDraw())
    {
        vvViewer::Viewer::frame(vv->frameTime());
    }
    else
    {
        if (vvVIVE::instance()->initDone())
            vvPluginList::instance()->clusterSyncDraw();
        if (vvViewer::unsyncedFrames <= 0)
        {
            vvMSController::instance()->barrierDraw();
        }
        else
        {
            vvViewer::unsyncedFrames--;
        }
    }
    if (getViewerStats()->collectStats("frame"))
    {
        double endFrame = elapsedTime();
        vsg::FrameStamp *frameStamp = getViewerFrameStamp();
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "frame begin time", startFrame);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "frame end time", endFrame);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "frame time taken", endFrame - startFrame);
    }

    if (reEnableCulling)
    {
        culling(true);
        reEnableCulling = false;
    }

    vvWindows::instance()->updateContents();
}*/

/*
void vvViewer::renderingTraversals()
{
    Contexts contexts;
    getContexts(contexts);

    // check to see if windows are still valid
    checkWindowStatus();
    if (_done)
        return;

    double beginRenderingTraversals = elapsedTime();

    vsg::FrameStamp *frameStamp = getViewerFrameStamp();

    if (getViewerStats() && getViewerStats()->collectStats("scene"))
    {
        unsigned int frameNumber = frameStamp ? frameStamp->getFrameNumber() : 0;

        Views views;
        getViews(views);
        for (Views::iterator vitr = views.begin();
             vitr != views.end();
             ++vitr)
        {
            vvViewer::View *view = *vitr;
            vsg::Stats *stats = view->getStats();
            vsg::Node *sceneRoot = view->getSceneData();
            if (sceneRoot && stats)
            {
                vvUtil::StatsVisitor statsVisitor;
                sceneRoot->accept(statsVisitor);
                statsVisitor.totalUpStats();

                unsigned int unique_primitives = 0;
                vvUtil::Statistics::PrimitiveCountMap::iterator pcmitr;
                for (pcmitr = statsVisitor._uniqueStats.GetPrimitivesBegin();
                     pcmitr != statsVisitor._uniqueStats.GetPrimitivesEnd();
                     ++pcmitr)
                {
                    unique_primitives += pcmitr->second;
                }

                stats->setAttribute(frameNumber, "Number of unique StateSet", static_cast<double>(statsVisitor._statesetSet.size()));
                stats->setAttribute(frameNumber, "Number of unique Group", static_cast<double>(statsVisitor._groupSet.size()));
                stats->setAttribute(frameNumber, "Number of unique Transform", static_cast<double>(statsVisitor._transformSet.size()));
                stats->setAttribute(frameNumber, "Number of unique LOD", static_cast<double>(statsVisitor._lodSet.size()));
                stats->setAttribute(frameNumber, "Number of unique Switch", static_cast<double>(statsVisitor._switchSet.size()));
                stats->setAttribute(frameNumber, "Number of unique Geode", static_cast<double>(statsVisitor._geodeSet.size()));
                stats->setAttribute(frameNumber, "Number of unique Drawable", static_cast<double>(statsVisitor._drawableSet.size()));
                stats->setAttribute(frameNumber, "Number of unique Geometry", static_cast<double>(statsVisitor._geometrySet.size()));
                stats->setAttribute(frameNumber, "Number of unique Vertices", static_cast<double>(statsVisitor._uniqueStats._vertexCount));
                stats->setAttribute(frameNumber, "Number of unique Primitives", static_cast<double>(unique_primitives));

                unsigned int instanced_primitives = 0;
                for (pcmitr = statsVisitor._instancedStats.GetPrimitivesBegin();
                     pcmitr != statsVisitor._instancedStats.GetPrimitivesEnd();
                     ++pcmitr)
                {
                    instanced_primitives += pcmitr->second;
                }

                stats->setAttribute(frameNumber, "Number of instanced Stateset", static_cast<double>(statsVisitor._numInstancedStateSet));
                stats->setAttribute(frameNumber, "Number of instanced Group", static_cast<double>(statsVisitor._numInstancedGroup));
                stats->setAttribute(frameNumber, "Number of instanced Transform", static_cast<double>(statsVisitor._numInstancedTransform));
                stats->setAttribute(frameNumber, "Number of instanced LOD", static_cast<double>(statsVisitor._numInstancedLOD));
                stats->setAttribute(frameNumber, "Number of instanced Switch", static_cast<double>(statsVisitor._numInstancedSwitch));
                stats->setAttribute(frameNumber, "Number of instanced Geode", static_cast<double>(statsVisitor._numInstancedGeode));
                stats->setAttribute(frameNumber, "Number of instanced Drawable", static_cast<double>(statsVisitor._numInstancedDrawable));
                stats->setAttribute(frameNumber, "Number of instanced Geometry", static_cast<double>(statsVisitor._numInstancedGeometry));
                stats->setAttribute(frameNumber, "Number of instanced Vertices", static_cast<double>(statsVisitor._instancedStats._vertexCount));
                stats->setAttribute(frameNumber, "Number of instanced Primitives", static_cast<double>(instanced_primitives));
            }
        }
    }

    Scenes scenes;
    getScenes(scenes);

    for (Scenes::iterator sitr = scenes.begin();
         sitr != scenes.end();
         ++sitr)
    {
        vvViewer::Scene *scene = *sitr;
        vvDB::DatabasePager *dp = scene ? scene->getDatabasePager() : 0;
        if (dp)
        {
            dp->signalBeginFrame(frameStamp);
        }

        if (scene->getSceneData())
        {
            // fire off a build of the bounding volumes while we
            // are still running single threaded.
            scene->getSceneData()->getBound();
        }
    }

    // vsg::notify(vsg::NOTICE)<<std::endl<<"Start frame"<<std::endl;

    Cameras cameras;
    getCameras(cameras);

    Contexts::iterator itr;

    bool doneMakeCurrentInThisThread = false;

    if (_endDynamicDrawBlock.valid())
    {
        _endDynamicDrawBlock->reset();
    }

    // dispatch the the rendering threads
    if (_startRenderingBarrier.valid())
        _startRenderingBarrier->block();

    // reset any double buffer graphics objects
    for (Cameras::iterator camItr = cameras.begin();
         camItr != cameras.end();
         ++camItr)
    {
        vsg::Camera *camera = *camItr;
        vvViewer::Renderer *renderer = dynamic_cast<vvViewer::Renderer *>(camera->getRenderer());
        if (renderer)
        {
            if (!renderer->getGraphicsThreadDoesCull() && !(camera->getCameraThread()))
            {
                renderer->cull();
            }
        }
    }

    for (itr = contexts.begin();
         itr != contexts.end();
         ++itr)
    {
        if (_done)
            return;
        if (!((*itr)->getGraphicsThread()) && (*itr)->valid())
        {
            doneMakeCurrentInThisThread = true;
            makeCurrent(*itr);
            (*itr)->runOperations();
        }
    }

    // vsg::notify(vsg::NOTICE)<<"Joing _endRenderingDispatchBarrier block "<<_endRenderingDispatchBarrier.get()<<std::endl;
    
    // wait till the rendering dispatch is done.
    if (_endRenderingDispatchBarrier.valid())
        _endRenderingDispatchBarrier->block();

    // begin OpenCOVER
    bool sync = false;
    for (itr = contexts.begin();
         itr != contexts.end();
         ++itr)
    {
        if (_done)
            return;

        if (!((*itr)->getGraphicsThread())) // no graphics thread, so we need to sync manually
        {

            doneMakeCurrentInThisThread = true;
            makeCurrent(*itr);

            glContextOperation(*itr);
	    
            double beginFinish = elapsedTime();
            if (m_requireGlFinish)
            {
                glFinish();
                if (getViewerStats()->collectStats("finish"))
                {
                    double endFinish = elapsedTime();
                    getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "finish begin time", beginFinish);
                    getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "finish end time", endFinish);
                    getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "finish time taken", endFinish - beginFinish);
                }
            }
            sync = true;
        }
    }

    if (vvVIVE::instance()->initDone())
    {
        vvPluginList::instance()->clusterSyncDraw();

        int WindowNum = 0;
        for (itr = contexts.begin();
             itr != contexts.end();
             ++itr)
        {
            if (!((*itr)->getGraphicsThread()) && (*itr)->valid())
            {
                doneMakeCurrentInThisThread = true;
                makeCurrent(*itr);
                vvPluginList::instance()->preSwapBuffers(WindowNum);
                WindowNum++;
            }
        }
    }

    if (sync)
    {
        //do sync here before swapBuffers

        // sync multipc instances
        if (vvViewer::unsyncedFrames <= 0)
        {
            double beginSync = elapsedTime();
            vvMSController::instance()->barrierDraw();
            if (getViewerStats()->collectStats("sync"))
            {
                double endSync = elapsedTime();
                getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "sync begin time", beginSync);
                getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "sync end time", endSync);
                getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "sync time taken", endSync - beginSync);
            }
        }
        else
            vvViewer::unsyncedFrames--;
    }
    double beginSwap = elapsedTime();
    // end OpenCOVER
    for (itr = contexts.begin();
         itr != contexts.end();
         ++itr)
    {
        if (_done)
            return;

        if (!((*itr)->getGraphicsThread()) && (*itr)->valid())
        {
            doneMakeCurrentInThisThread = true;
            makeCurrent(*itr);
            (*itr)->swapBuffers();
        }
    }
    if (getViewerStats()->collectStats("sync"))
    {
        double endSwap = elapsedTime();
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "swap begin time", beginSwap);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "swap end time", endSwap);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "swap time taken", endSwap - beginSwap);
    }

    if (vvVIVE::instance()->initDone())
    {
        int WindowNum = 0;
        for (itr = contexts.begin();
             itr != contexts.end();
             ++itr)
        {
            if (!((*itr)->getGraphicsThread()) && (*itr)->valid())
            {
                doneMakeCurrentInThisThread = true;
                makeCurrent(*itr);
                vvPluginList::instance()->postSwapBuffers(WindowNum);
                WindowNum++;
            }
        }
    }

    for (Scenes::iterator sitr = scenes.begin();
         sitr != scenes.end();
         ++sitr)
    {
        vvViewer::Scene *scene = *sitr;
        vvDB::DatabasePager *dp = scene ? scene->getDatabasePager() : 0;
        if (dp)
        {
            dp->signalEndFrame();
        }
    }

    // wait till the dynamic draw is complete.
    if (_endDynamicDrawBlock.valid())
    {
        // vsg::Timer_t startTick = vsg::Timer::instance()->tick();
        _endDynamicDrawBlock->block();
        // vsg::notify(vsg::NOTICE)<<"Time waiting "<<vsg::Timer::instance()->delta_m(startTick, vsg::Timer::instance()->tick())<<std::endl;;
    }

    if (_releaseContextAtEndOfFrameHint && doneMakeCurrentInThisThread)
    {
        //vv_NOTICE<<"Doing release context"<<std::endl;
        releaseContext();
    }

    if (getViewerStats() && getViewerStats()->collectStats("update"))
    {
        double endRenderingTraversals = elapsedTime();

        // update current frames stats
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "Rendering traversals begin time", beginRenderingTraversals);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "Rendering traversals end time", endRenderingTraversals);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "Rendering traversals time taken", endRenderingTraversals - beginRenderingTraversals);
    }

    _requestRedraw = false;
}*/

}
