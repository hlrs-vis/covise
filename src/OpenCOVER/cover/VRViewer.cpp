/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			VRViewer.C (Performer 2.0)		*
 *									*
 *	Description		stereo viewer class			*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			20.08.97				*
 *				09.01.97 general viewing frustum	*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <GL/glew.h>

#include <util/common.h>
#include <util/unixcompat.h>

#include <osgViewer/View>
#include "coVRRenderer.h"

#include "coVRSceneView.h"
#include <config/CoviseConfig.h>
#include "VRViewer.h"
#include "VRWindow.h"
#include "OpenCOVER.h"
#include <OpenVRUI/osg/mathUtils.h>
#include "coVRPluginList.h"
#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include "coCullVisitor.h"
#include "MarkerTracking.h"
#include "InitGLOperation.h"
#include "input/input.h"
#include "tridelity.h"
#include "ui/Button.h"
#include "ui/SelectionList.h"
#include "ui/Menu.h"
#include "coVRStatsDisplay.h"

#include <osg/LightSource>
#include <osg/ApplicationUsage>
#include <osg/StateSet>
#include <osg/BlendFunc>
#include <osg/Camera>
#include <osg/Material>
#include <osg/CameraView>
#include <osg/DeleteHandler>
#include <osg/TextureRectangle>
#include <osgDB/DatabasePager>
#include <osg/CullStack>
#include <osgText/Text>
#include <osgUtil/Statistics>
#include <osgUtil/UpdateVisitor>
#include <osgUtil/Optimizer>
#include <osgUtil/GLObjectsVisitor>
#include <osgDB/Registry>
#include <osgDB/ReadFile>

#include <osgGA/AnimationPathManipulator>
#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/DriveManipulator>
#include <osgGA/StateSetManipulator>

#include "coVRMSController.h"
#include "MSEventHandler.h"

#ifndef _WIN32
#include <termios.h>
#include <unistd.h>
#endif

#define ANIMATIONSPEED 1

#ifndef OSG_NOTICE
#define OSG_NOTICE std::cerr
#endif

#include <cassert>

#ifdef __linux
#include <unistd.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#endif

#ifndef GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX
#define GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX          0x9047
#define GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX    0x9048
#define GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX  0x9049
#define GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX            0x904A
#define GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX            0x904B
#endif

//#define USE_NVX_INFO

using namespace covise;

namespace opencover {

static void clearGlWindow(bool doubleBuffer)
{
    if (doubleBuffer)
    {
        glDrawBuffer(GL_FRONT);
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (doubleBuffer)
    {
        glDrawBuffer(GL_BACK);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

int VRViewer::unsyncedFrames = 0;
// Draw operation, that does a draw on the scene graph.
struct RemoteSyncOp : public osg::Operation
{
    RemoteSyncOp()
        : osg::Operation("RemoteSyncOp", true)
    {
    }

    virtual void operator()(osg::Object *object)
    {
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context)
            return;

        //do sync here before swapBuffers

        // sync multipc instances
        if (VRViewer::unsyncedFrames <= 0)
        {
            coVRMSController::instance()->syncDraw();
        }
        else
            VRViewer::unsyncedFrames--;
    }
};

// call plugins before syncing
struct PreSwapBuffersOp : public osg::Operation
{
    PreSwapBuffersOp(int WindowNumber)
        : osg::Operation("PreSwapBuffersOp", true)
    {
        windowNumber = WindowNumber;
    }

    virtual void operator()(osg::Object *object)
    {
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context)
            return;

        if (OpenCOVER::instance()->initDone())
            coVRPluginList::instance()->preSwapBuffers(windowNumber);
    }
    int windowNumber;
};

// call plugins after syncing
struct PostSwapBuffersOp : public osg::Operation
{
    PostSwapBuffersOp(int WindowNumber)
        : osg::Operation("PostSwapBuffersOp", true)
    {
        windowNumber = WindowNumber;
    }

    virtual void operator()(osg::Object *object)
    {
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context)
            return;

        if (OpenCOVER::instance()->initDone())
            coVRPluginList::instance()->postSwapBuffers(windowNumber);
    }
    int windowNumber;
};

// Compile operation, that compile OpenGL objects.
// nothing changed
struct ViewerCompileOperation : public osg::Operation
{
    ViewerCompileOperation(osg::Node *scene)
        : osg::Operation("Compile", false)
        , _scene(scene)
    {
    }

    virtual void operator()(osg::Object *object)
    {
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context)
            return;

        // OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
        // osg::notify(osg::NOTICE)<<"Compile "<<context<<" "<<OpenThreads::Thread::CurrentThread()<<std::endl;

        // context->makeCurrent();

        osgUtil::GLObjectsVisitor compileVisitor;
        compileVisitor.setState(context->getState());

        // do the compile traversal
        if (_scene.valid())
            _scene->accept(compileVisitor);

        // osg::notify(osg::NOTICE)<<"Done Compile "<<context<<" "<<OpenThreads::Thread::CurrentThread()<<std::endl;
    }

    osg::ref_ptr<osg::Node> _scene;
};

// Draw operation, that does a draw on the scene graph.
struct ViewerRunOperations : public osg::Operation
{
    ViewerRunOperations()
        : osg::Operation("RunOperation", true)
    {
    }

    virtual void operator()(osg::Object *object)
    {
        static int numClears = 0; // OpenCOVER
        osg::GraphicsContext *context = dynamic_cast<osg::GraphicsContext *>(object);
        if (!context)
            return;
        // OpenCOVER begin
        // clear the whole window, if channels are smaller than window
        VRViewer::instance()->glContextOperation(context);
        // OpenCOVER end

        context->runOperations();
    }
};

//OpenCOVER
bool VRViewer::handleEvents()
{
    return myeh->update();
}

//OpenCOVER
bool VRViewer::update()
{
    bool again = false;
    if (cover->debugLevel(5))
        fprintf(stderr, "VRViewer::update\n");

    if (animateSeparation)
    {
        again = true;
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
            //osg::Matrix vm = vpMarker->getCameraTrans();
            //osg::Vec3 vp;

            //vm.getTrans( viewPos);
            //viewMat.setTrans( viewPos);
            viewMat = vpMarker->getCameraTrans();
        }
        again = true;
    }

    // compute viewer position

    viewPos = viewMat.getTrans();


    if (cover->debugLevel(5))
    {
        fprintf(stderr, "\t viewPos=[%f %f %f]\n", viewPos[0], viewPos[1], viewPos[2]);
        fprintf(stderr, "\n");
    }

    for (int i = 0; i < coVRConfig::instance()->numChannels(); i++)
    {
        setFrustumAndView(i);
    }
    return again;
}

//////////////////////////////////////////////////////////////////////////////
//
// osgViewer::Viewer implemention
//

VRViewer *VRViewer::s_singleton = NULL;
VRViewer *VRViewer::instance()
{
    if (!s_singleton)
        s_singleton = new VRViewer;
    return s_singleton;
}

//OpenCOVER
VRViewer::VRViewer()
: ui::Owner("VRViewer", cover->ui)
, animateSeparation(0)
, stereoOn(true)
{
    assert(!s_singleton);

    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew VRViewer\n");
    reEnableCulling = false;
#if OSG_VERSION_GREATER_OR_EQUAL(3, 6, 0)
    setUseConfigureAffinity(false); // tell OpenSceneGraph not to set affinity (good if you want to run multiple instances on one machine)
#endif

    m_initGlOp = new InitGLOperation();
    setRealizeOperation(m_initGlOp);

    unsyncedFrames = 0;
    lastFrameTime = 0.0;

    myeh = new MSEventHandler;
    _eventHandlers.push_front(myeh);
    setKeyEventSetsDone(false);

    requestedSeparation = 0;

    // automatic angle
    screen_angle = NULL;

    // clear once at startup
    clearWindow = true;

    readConfigFile();

    viewDir.set(0.0f, 0.0f, 0.0f);

    //initial view position
    float xp = coCoviseConfig::getFloat("x", "COVER.ViewerPosition", 0.0f);
    float yp = coCoviseConfig::getFloat("y", "COVER.ViewerPosition", -2000.0f);
    float zp = coCoviseConfig::getFloat("z", "COVER.ViewerPosition", 30.0f);
    viewPos.set(xp, yp, zp);

    initialViewPos = viewPos;
    // set initial view
    viewMat.makeTranslate(viewPos);

    stereoOn = coVRConfig::instance()->stereoState();

    separation = stereoOn ? Input::instance()->eyeDistance() : 0.;
    arTracking = false;
    if (coCoviseConfig::isOn("COVER.MarkerTracking.TrackViewpoint", false))
    {
        arTracking = true;
        vpMarker = MarkerTracking::instance()->getMarker("ViewpointMarker");
    }
    overwritePAndV = false;

    auto stereosep = new ui::Button("StereoSep", this);
    cover->viewOptionsMenu->add(stereosep);
    stereosep->setText("Stereo separation");
    stereosep->setState(stereoOn);
    stereosep->setCallback([this](bool state){
        stereoOn = state;
        setSeparation(Input::instance()->eyeDistance());
    });

    auto ortho = new ui::Button("Orthographic", this);
    cover->viewOptionsMenu->add(ortho);
    ortho->setText("Orthographic projection");
    ortho->setState(coVRConfig::instance()->orthographic());
    ortho->setCallback([this](bool state){
        coVRConfig::instance()->setOrthographic(state);
        update();
        requestRedraw();
    });

    auto threading = new ui::SelectionList("ThreadingModel", this);
    cover->viewOptionsMenu->add(threading);
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
    cover->viewOptionsMenu->add(showStats);
    showStats->setText("Renderer statistics");
    showStats->setShortcut("Shift+S");
    showStats->append("Off");
    showStats->append("Frames/s");
    showStats->append("Viewer");
    showStats->append("Viewer+camera");
    showStats->append("Viewer+camera+nodes");
    cover->viewOptionsMenu->add(showStats);
    showStats->select(coVRConfig::instance()->drawStatistics);
    showStats->setCallback([this](int val){
        coVRConfig::instance()->drawStatistics = val;
        statsDisplay->showStats(val, VRViewer::instance());
        //XXX setInstrumentationMode( coVRConfig::instance()->drawStatistics );
    });

    statsDisplay = new coVRStatsDisplay();
}

//OpenCOVER
VRViewer::~VRViewer()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete VRViewer\n");

    delete statsDisplay;

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

    coVRSceneView::destroyUniforms();

    s_singleton = NULL;
}



void VRViewer::createViewportCameras(int i)
{
    viewportStruct &vp = coVRConfig::instance()->viewports[i];
    if (vp.mode == viewportStruct::Channel)
        return;

    osg::ref_ptr<osg::Camera> cameraWarp = new osg::Camera;
    std::stringstream str;
    str << "WarpOrtho/Viewport " << i;
    cameraWarp->setName(str.str());
    cameraWarp->setClearColor(osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    cameraWarp->setClearMask(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    cameraWarp->setRenderOrder(osg::Camera::POST_RENDER);
    cameraWarp->setAllowEventFocus(false);

    cameraWarp->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    cameraWarp->setProjectionMatrix(osg::Matrix::ortho2D(0, 1, 0, 1));
    cameraWarp->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    int PBOnum = -1;
    switch(vp.mode)
    {
        case viewportStruct::Channel:
            break;
        case viewportStruct::PBO:
            PBOnum = vp.PBOnum;
            break;
        case viewportStruct::TridelityML:
        case viewportStruct::TridelityMV:
                assert(vp.pbos.size() == 5);
                for (size_t i=0; i<vp.pbos.size(); ++i)
                {
                    int PBOnum = vp.pbos[i];
                    if (PBOnum < 0 || PBOnum >= coVRConfig::instance()->numPBOs())
                    {
                        cerr << "invalid PBO index " << PBOnum << " for viewport " << i << endl;
                        return;
                    }
                }
                PBOnum = vp.pbos[0];
                break;
    }
    if (PBOnum < 0 || PBOnum >= coVRConfig::instance()->numPBOs())
    {
        cerr << "invalid PBO index " << PBOnum << " for viewport " << i << endl;
        return;
    }
    windowStruct &win = coVRConfig::instance()->windows[coVRConfig::instance()->PBOs[PBOnum].windowNum];
    if (osg::GraphicsContext *gc = win.context)
    {
        cameraWarp->setGraphicsContext(gc);
    }
    else
    {
        cerr << "no GraphicsContext for viewport " << i << endl;
    }

    const int sx = win.sx;
    const int sy = win.sy;
    cameraWarp->setViewport(new osg::Viewport(vp.viewportXMin * sx, vp.viewportYMin * sy,
                (vp.viewportXMax - vp.viewportXMin) * sx, (vp.viewportYMax - vp.viewportYMin) * sy));

    osg::Geode *geode = new osg::Geode;
    osg::Geometry *geometry = NULL;
    switch(vp.mode)
    {
        case viewportStruct::Channel:
            break;;
        case viewportStruct::PBO:
            {
                geometry = distortionMesh(vp.distortMeshName.c_str());
                osg::ref_ptr<osg::StateSet> state = geode->getOrCreateStateSet();
                state->setTextureAttributeAndModes(0, coVRConfig::instance()->PBOs[PBOnum].renderTargetTexture, osg::StateAttribute::ON);
                if(vp.blendingTextureName.length()>0)
                {
                    osg::Image *blendTexImage = osgDB::readImageFile(vp.blendingTextureName.c_str());
                    osg::Texture2D *blendTex = new osg::Texture2D;
                    blendTex->ref();
                    blendTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP);
                    blendTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP);
                    if (blendTexImage)
                    {
                        blendTex->setImage(blendTexImage);
                    }
                    state->setTextureAttributeAndModes(1, blendTex);
                }
                geode->setStateSet(state.get());
            }
            break;
        case viewportStruct::TridelityML:
        case viewportStruct::TridelityMV:
            {
                osg::ref_ptr<osg::StateSet> state = geode->getOrCreateStateSet();
                osg::Vec3Array *positionArray = new osg::Vec3Array;
                osg::Vec4Array *colorArray = new osg::Vec4Array;
                osg::Vec2Array *textureArray = new osg::Vec2Array;
                osg::UShortArray *indexArray = new osg::UShortArray;
                osg::ref_ptr<osg::DrawElementsUShort> drawElement;
                geometry = new osg::Geometry;
                geometry->setSupportsDisplayList(false);
                geometry->setUseVertexBufferObjects(true);
                positionArray->push_back(osg::Vec3f(0,0,0));
                positionArray->push_back(osg::Vec3f(1,0,0));
                positionArray->push_back(osg::Vec3f(1,1,0));
                positionArray->push_back(osg::Vec3f(0,1,0));
                textureArray->push_back(osg::Vec2f(0,0));
                textureArray->push_back(osg::Vec2f(1,0));
                textureArray->push_back(osg::Vec2f(1,1));
                textureArray->push_back(osg::Vec2f(0,1));

                indexArray->push_back(0);
                indexArray->push_back(1);
                indexArray->push_back(2);
                indexArray->push_back(3);
                colorArray->push_back(osg::Vec4f(1,1,1,1));
                colorArray->push_back(osg::Vec4f(1,1,1,1));
                colorArray->push_back(osg::Vec4f(1,1,1,1));
                colorArray->push_back(osg::Vec4f(1,1,1,1));

                drawElement = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, indexArray->size(), (GLushort *)indexArray->getDataPointer());

                geometry->addPrimitiveSet(drawElement);
                for (int i=0; i<5; ++i)
                {
                    geometry->setTexCoordArray(i,textureArray,osg::Array::BIND_PER_VERTEX);
                }
                geometry->setVertexArray(positionArray);
                geometry->setColorArray(colorArray);
                geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX); // colors per vertex for blending
                geode->addDrawable(geometry);

                for (int i=0; i<5; ++i)
                {
                    int p = vp.pbos[i];
                    state->setTextureAttributeAndModes(i, coVRConfig::instance()->PBOs[p].renderTargetTexture, osg::StateAttribute::ON);
                }

                osg::TextureRectangle *lookupTex = new osg::TextureRectangle;
                osg::Image *lookupTexImage = vp.mode==viewportStruct::TridelityML ? tridelityLookupML() : tridelityLookupMV();
                lookupTex->setImage(lookupTexImage);
                state->setTextureAttributeAndModes(5, lookupTex);

                osg::Program *lookupProgram = new osg::Program;                                                                          
                osg::Shader *lookupFrag = new osg::Shader(osg::Shader::FRAGMENT);
                lookupProgram->addShader(lookupFrag);
                lookupFrag->setShaderSource(
						"uniform sampler2D view0;"
						"uniform sampler2D view1;"
						"uniform sampler2D view2;"
						"uniform sampler2D view3;"
						"uniform sampler2D view4;"
						"uniform sampler2DRect lookup;"
						"uniform vec2 lookupSize;"
                        ""
						"void main()"
						"{"
						"       vec2 coord = vec2(gl_TexCoord[0].x, gl_TexCoord[0].y);"
                        "       float w = lookupSize.x*0.2;"
						"       float xx = mod(gl_FragCoord.x, w);"
						"       float yy = mod(gl_FragCoord.y, lookupSize.y);"
						"       vec3 c0 = texture2D(view0, coord).rgb;"
						"       vec3 c1 = texture2D(view1, coord).rgb;"
						"       vec3 c2 = texture2D(view2, coord).rgb;"
						"       vec3 c3 = texture2D(view3, coord).rgb;"
						"       vec3 c4 = texture2D(view4, coord).rgb;"
						"       vec3 l0 = texture2DRect(lookup, vec2(xx,yy)).rgb;"
						"       vec3 l1 = texture2DRect(lookup, vec2(xx + w, yy)).rgb;"
						"       vec3 l2 = texture2DRect(lookup, vec2(xx + 2.0 * w,yy)).rgb;"
						"       vec3 l3 = texture2DRect(lookup, vec2(xx + 3.0 * w,yy)).rgb;"
						"       vec3 l4 = texture2DRect(lookup, vec2(xx + 4.0 * w,yy)).rgb;"
						"       gl_FragColor.rgb = c4*l4 + c3*l3 + c2*l2 + c1*l1 + c0*l0;"
						"}"
                        );
                for (int i=0; i<5; ++i)
                {
                    std::stringstream str;
                    str << "view" << i;
                    std::string s = str.str();
                    osg::Uniform *tex = new osg::Uniform(s.c_str(), i);
                    state->addUniform(tex);
                }
                osg::Uniform *lookup = new osg::Uniform("lookup", 5);
                state->addUniform(lookup);
                osg::Vec2 size(lookupTexImage->s(), lookupTexImage->t());
                osg::Uniform *lookupSize = new osg::Uniform("lookupSize", size);
                state->addUniform(lookupSize);
                state->setAttributeAndModes(lookupProgram, osg::StateAttribute::ON);

                geode->setStateSet(state.get());
            }
            break;
    }
    if (geometry)
    {
        geode->addDrawable(geometry);
    }
    if (geode)
    {
        osg::ref_ptr<osg::StateSet> state = geode->getOrCreateStateSet();
        state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        state->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
        geode->setStateSet(state.get());
        cameraWarp->addChild(geode);
    }
    cover->getScene()->addChild(cameraWarp.get());

    if (viewportCamera.size() <= i)
        viewportCamera.resize(i+1);
    viewportCamera[i] = cameraWarp;
}

void VRViewer::destroyViewportCameras(int i)
{
    if (viewportCamera.size()>i && viewportCamera[i])
    {
        cover->getScene()->removeChild(viewportCamera[i]);
        viewportCamera[i] = nullptr;
    }
}

void VRViewer::createBlendingCameras(int i)
{
    blendingTextureStruct &bt = coVRConfig::instance()->blendingTextures[i];
        osg::GraphicsContext *gc = coVRConfig::instance()->windows[bt.window].context;

        osg::ref_ptr<osg::Camera> cameraBlend = new osg::Camera;
        std::stringstream str;
        str << "Blending Texture " << i;
        cameraBlend->setName(str.str());
        cameraBlend->setClearColor(osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        cameraBlend->setClearMask(0);
        cameraBlend->setRenderOrder(osg::Camera::POST_RENDER);
        cameraBlend->setAllowEventFocus(false);

        cameraBlend->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
        cameraBlend->setProjectionMatrix(osg::Matrix::ortho2D(0, 1, 0, 1));
        cameraBlend->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        if (gc)
        {
            cameraBlend->setGraphicsContext(gc);
        }


        cameraBlend->setName("BlendingCamera");
        cameraBlend->setViewport(new osg::Viewport(bt.viewportXMin * coVRConfig::instance()->windows[bt.window].sx,
            bt.viewportYMin * coVRConfig::instance()->windows[bt.window].sy,
            (bt.viewportXMax - bt.viewportXMin) * coVRConfig::instance()->windows[bt.window].sx,
            (bt.viewportYMax - bt.viewportYMin) * coVRConfig::instance()->windows[bt.window].sy));


        osg::Vec3Array *coord = new osg::Vec3Array(4);
        (*coord)[0].set(0, 0, 0);
        (*coord)[1].set(1, 0, 0);
        (*coord)[2].set(1, 1, 0);
        (*coord)[3].set(0, 1, 0);

        osg::Vec2Array *texcoord = new osg::Vec2Array(4);

        (*texcoord)[0].set(0.0, 0.0);
        (*texcoord)[1].set(1.0, 0.0);
        (*texcoord)[2].set(1.0, 1.0);
        (*texcoord)[3].set(0.0, 1.0);

        osg::Geode *geode = new osg::Geode();
        osg::Geometry *geometry = new osg::Geometry();

        ushort *vertices = new ushort[4];
        vertices[0] = 0;
        vertices[1] = 1;
        vertices[2] = 2;
        vertices[3] = 3;
        osg::DrawElementsUShort *plane = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, vertices);

        geometry->setVertexArray(coord);
        geometry->addPrimitiveSet(plane);
        geometry->setTexCoordArray(0, texcoord);

        osg::ref_ptr<osg::StateSet> state = geode->getOrCreateStateSet();
        state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        state->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
        state->setMode(GL_BLEND, osg::StateAttribute::ON);
        osg::BlendFunc *blendFunc = new osg::BlendFunc();
        blendFunc->setFunction(osg::BlendFunc::ZERO, osg::BlendFunc::SRC_COLOR);
        state->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
        if(bt.blendingTextureName.length()>0)
        {
            osg::Image *blendTexImage = osgDB::readImageFile(bt.blendingTextureName.c_str());
            osg::Texture2D *blendTex = new osg::Texture2D;
            blendTex->ref();
            blendTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP);
            blendTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP);
            if (blendTexImage)
            {
                blendTex->setImage(blendTexImage);
            }
            state->setTextureAttributeAndModes(0, blendTex);
        }
        
    geode->addDrawable(geometry);
        geode->setStateSet(state.get());

        cameraBlend->addChild(geode);

        cover->getScene()->addChild(cameraBlend.get());

        if (blendingCamera.size() <= i)
            blendingCamera.resize(i+1);
        blendingCamera[i] = cameraBlend;
}

void VRViewer::destroyBlendingCameras(int i)
{
    if (blendingCamera.size()>i && blendingCamera[i])
    {
        cover->getScene()->removeChild(blendingCamera[i]);
    }
}

osg::Geometry *VRViewer::distortionMesh(const char *fileName)
{
    osg::Vec3Array *positionArray = new osg::Vec3Array;
    osg::Vec4Array *colorArray = new osg::Vec4Array;
    osg::Vec2Array *textureArray = new osg::Vec2Array;
    osg::UShortArray *indexArray = new osg::UShortArray;


    osg::ref_ptr<osg::DrawElementsUShort> drawElement;
    // Get triangle indicies

    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
    geometry->setSupportsDisplayList(false);
    geometry->setUseVertexBufferObjects(true);

    FILE *fp = NULL;
    if (fileName && fileName[0])
        fp = fopen(fileName,"r");
    if(fp!=NULL)
    {
        cerr << "reading distortion mesh from " << fileName << std::endl;
        char buf[501];
        int numVert;
        int numFaces;
        int NATIVEYRES;
        int NATIVEXRES;
        float ortholeft=0;
        float orthoright=1;
        float sh = 1.0;
        while(!feof(fp))
        {
            if (!fgets(buf,500,fp))
            {
                std::cerr << "failed 1 to read line from " << fileName << std::endl;
            }
            if(strncmp(buf,"ORTHO_LEFT",10)==0)
            {
                sscanf(buf+10,"%f", &ortholeft);
                sh = 1.0 / (orthoright - ortholeft);
            };
            if(strncmp(buf,"ORTHO_RIGHT",11)==0)
            {
                sscanf(buf+11,"%f", &orthoright);
                sh = 1.0 / (orthoright - ortholeft);
            }
            if(strncmp(buf,"NATIVEXRES",10)==0)
            {
                sscanf(buf+10,"%d", &NATIVEXRES);
            }
            if(strncmp(buf,"NATIVEYRES",10)==0)
            {
                sscanf(buf+10,"%d", &NATIVEYRES);
                float vx,vy,brightness,tx,ty;
                for(int i=0;i<numVert;i++)
                {
                    if (!fgets(buf,500,fp))
                    {
                        std::cerr << "failed 2 to read line from " << fileName << std::endl;
                    }
                    sscanf(buf,"%f %f %f %f %f",&vx,&vy,&brightness,&tx, &ty);
                    positionArray->push_back(osg::Vec3f(vx/NATIVEXRES,1.0-(vy/NATIVEYRES),0));
                    textureArray->push_back(osg::Vec2f((tx-ortholeft)*sh,ty));
                    colorArray->push_back(osg::Vec4f(brightness/255.0,brightness/255.0,brightness/255.0,1));
                }
                int tr[3];
                for(int i=0;i<numFaces;i++)
                {
                    if (!fgets(buf,500,fp))
                    {
                        std::cerr << "failed 3 to read line from " << fileName << std::endl;
                    }
                    sscanf(buf,"[ %d %d %d",&tr[0],&tr[1],&tr[2]);
                    indexArray->push_back(tr[0]);
                    indexArray->push_back(tr[1]);
                    indexArray->push_back(tr[2]);
                }
            }
            else if(strncmp(buf,"VERTICES",8)==0)
            {
                sscanf(buf+8,"%d", &numVert);
            }
            else if(strncmp(buf,"FACES",5)==0)
            {
                sscanf(buf+5,"%d", &numFaces);
            }
        }
        drawElement = new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLES, indexArray->size(), (GLushort *)indexArray->getDataPointer());

    }
    else
    {
        positionArray->push_back(osg::Vec3f(0,0,0));
        positionArray->push_back(osg::Vec3f(1,0,0));
        positionArray->push_back(osg::Vec3f(1,1,0));
        positionArray->push_back(osg::Vec3f(0,1,0));
        textureArray->push_back(osg::Vec2f(0,0));
        textureArray->push_back(osg::Vec2f(1,0));
        textureArray->push_back(osg::Vec2f(1,1));
        textureArray->push_back(osg::Vec2f(0,1));

        indexArray->push_back(0);
        indexArray->push_back(1);
        indexArray->push_back(2);
        indexArray->push_back(3);
        colorArray->push_back(osg::Vec4f(1,1,1,1));
        colorArray->push_back(osg::Vec4f(1,1,1,1));
        colorArray->push_back(osg::Vec4f(1,1,1,1));
        colorArray->push_back(osg::Vec4f(1,1,1,1));

        drawElement = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, indexArray->size(), (GLushort *)indexArray->getDataPointer());
    }

    geometry->addPrimitiveSet(drawElement);
    geometry->setTexCoordArray(0,textureArray,osg::Array::BIND_PER_VERTEX);
    geometry->setTexCoordArray(1,textureArray,osg::Array::BIND_PER_VERTEX);
    geometry->setVertexArray(positionArray);
    geometry->setColorArray(colorArray);
    geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX); // colors per vertex for blending


    return geometry.release();
}

//OpenCOVER
void VRViewer::setAffinity()
{
#ifdef __linux__
    const int ncpu = sysconf(_SC_NPROCESSORS_ONLN);

    cpu_set_t cpumask;
    CPU_ZERO(&cpumask);

    if(coVRConfig::instance()->lockToCPU()>=0)
    {
        CPU_SET(coVRConfig::instance()->lockToCPU(), &cpumask);

        std::cerr << "locking to CPU " << coVRConfig::instance()->lockToCPU() << std::endl;
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
VRViewer::unconfig()
{
    stopThreading();

    auto &conf = *coVRConfig::instance();
    for (int i = 0; i < conf.numBlendingTextures(); i++)
    {
        destroyBlendingCameras(i);
    }
    blendingCamera.clear();
    for (int i = 0; i < conf.numViewports(); i++)
    {
        destroyViewportCameras(i);
    }
    viewportCamera.clear();
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
VRViewer::config()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRViewer::config\n");
    if (!statsHandler)
    {
        statsHandler = new osgViewer::StatsHandler;
        addEventHandler(statsHandler);
    }
    for (int i = 0; i < coVRConfig::instance()->numChannels(); i++)
    {
        createChannels(i);
    }
    for (int i = 0; i < coVRConfig::instance()->numViewports(); i++)
    {
        createViewportCameras(i);
    }
    for (int i = 0; i < coVRConfig::instance()->numBlendingTextures(); i++)
    {
        createBlendingCameras(i);
    }

    float r = coCoviseConfig::getFloat("r", "COVER.Background", 0.0f);
    float g = coCoviseConfig::getFloat("g", "COVER.Background", 0.0f);
    float b = coCoviseConfig::getFloat("b", "COVER.Background", 0.0f);
    backgroundColor = osg::Vec4(r, g, b, 1.0f);

    setClearColor(backgroundColor);

    // create the windows and run the threads.
    if (coCoviseConfig::isOn("COVER.MultiThreaded", false))
    {
        cerr << "VRViewer: using one thread per camera" << endl;
#if OSG_VERSION_LESS_THAN(3, 5, 5)
        osg::Referenced::setThreadSafeReferenceCounting(true);
#endif
        setThreadingModel(CullThreadPerCameraDrawThreadPerContext);
    }
    else
    {
        std::string threadingMode = coCoviseConfig::getEntry("COVER.MultiThreaded");
        if (threadingMode == "CullDrawThreadPerContext")
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
            setThreadingModel(SingleThreaded);
    }
    if (statsDisplay)
        statsDisplay->updateThreadingModelText(getThreadingModel());

    if (mustDraw())
        realize();

    setAffinity();

    for (int i = 0; i < coVRConfig::instance()->numChannels(); i++)
    {
        setFrustumAndView(i);
    }
    assignSceneDataToCameras();
    getUpdateVisitor()->setTraversalMask(Isect::Update);

    if (coVRMSController::instance()->isCluster())
    {
        bool haveSwapBarrier = m_initGlOp->boundSwapBarrier();
        haveSwapBarrier = coVRMSController::instance()->allReduceOr(haveSwapBarrier);
        if (haveSwapBarrier)
        {
#if 0
            // glFinish seems to be required also with enabled swap barriers
            if (cover->debugLevel(1))
                std::cerr << "VRViewer: swap group joined - disabling glFinish" << std::endl;
            m_requireGlFinish = false;
#endif
        }
    }
    else
    {
        if (cover->debugLevel(1))
            std::cerr << "VRViewer: not running in cluster mode - disabling glFinish" << std::endl;
        m_requireGlFinish = false;
    }

    if (statsDisplay)
    {
        statsDisplay->enableFinishStats(m_requireGlFinish);
        statsDisplay->enableSyncStats(coVRMSController::instance()->isCluster());
    }

    bool syncToVBlankConfigured = false;
    bool syncToVBlank = covise::coCoviseConfig::isOn("COVER.SyncToVBlank", false, &syncToVBlankConfigured);
    for (size_t i=0; i<coVRConfig::instance()->numWindows(); ++i)
    {
        if (syncToVBlankConfigured)
        {
            if (auto win = coVRConfig::instance()->windows[i].window)
            {
                win->setSyncToVBlank(syncToVBlank);
            }
        }
    }
}

//OpenCOVER
void VRViewer::updateViewerMat(const osg::Matrix &mat)
{

    if (cover->debugLevel(5))
    {
        fprintf(stderr, "VRViewer::updateViewerMat\n");
        //coCoord coord;
        //mat.getOrthoCoord(&coord);
        //fprintf(stderr,"hpr=[%f %f %f]\n", coord.hpr[0], coord.hpr[1], coord.hpr[2]);
    }

    if (!coVRConfig::instance()->frozen())
    {
        viewMat = mat;
    }
}

//OpenCOVER
void
VRViewer::setSeparation(float sep)
{
    requestedSeparation = stereoOn ? sep : 0.;

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
VRViewer::flipStereo()
{
    separation = -separation;
}

void VRViewer::setFullscreen(bool state)
{
    if (m_fullscreen != state)
    {
        unconfig();
        m_fullscreen = state;
        config();
    }
}

bool VRViewer::isFullscreen() const
{
    return m_fullscreen;
}

void
VRViewer::setRenderToTexture(bool b)
{
}

//OpenCOVER
void
VRViewer::createChannels(int i)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRViewer::createChannels\n");

#if 0
    osg::GraphicsContext::WindowingSystemInterface *wsi = osg::GraphicsContext::getWindowingSystemInterface();
    if (!wsi)
    {
        osg::notify(osg::NOTICE) << "VRViewer : Error, no WindowSystemInterface available, cannot create windows." << std::endl;
        return;
    }
#endif
    auto &conf = *coVRConfig::instance();
    const int vp = coVRConfig::instance()->channels[i].viewportNum;
    if (vp >= coVRConfig::instance()->numViewports())
    {
        fprintf(stderr, "VRViewer:: error creating channel %d\n", i);
        fprintf(stderr, "viewportNum %d is out of range (viewports are counted starting from 0)\n", vp);
        return;
    }
    bool RenderToTexture = false, RenderToOsgWindow = true;
    osg::ref_ptr<osg::GraphicsContext> gc = NULL;
    int pboNum = coVRConfig::instance()->channels[i].PBONum;
    if(vp >= 0)
    {
        //std::cerr << "chan " << i << ": have viewport" << std::endl;
        const int win = coVRConfig::instance()->viewports[vp].window;
        if (win < 0 || win >= coVRConfig::instance()->numWindows())
        {
            fprintf(stderr, "VRViewer:: error creating channel %d\n", i);
            fprintf(stderr, "windowIndex %d is out of range (windows are counted starting from 0)\n", win);
            return;
        }
        gc = coVRConfig::instance()->windows[win].context;
        pboNum = -1;
        if (!coVRConfig::instance()->windows[win].type.empty())
        {
            //std::cerr << "chan " << i << ": no window" << std::endl;
            RenderToOsgWindow = false;
        }
    }
    else
    {
        if (pboNum < 0)
        {
            cerr << "channel " << i << ": neither viewport nor PBO configured" << endl;
            return;
        }
		if (coVRConfig::instance()->PBOs.size() <= pboNum)
		{
			cerr << "PBO " << pboNum << " not available" << endl;
			return;
		}
		if (coVRConfig::instance()->windows.size() <= coVRConfig::instance()->PBOs[pboNum].windowNum)
		{
			cerr << "PBOs window " << coVRConfig::instance()->PBOs[pboNum].windowNum << " not available PBO nr." << pboNum << endl;
			return;
		}
        gc = coVRConfig::instance()->windows[coVRConfig::instance()->PBOs[pboNum].windowNum].context;
        RenderToTexture = true;
    }

    std::stringstream str;
    str << "Channel " << i;
    if (i == 0)
    {
        coVRConfig::instance()->channels[i].camera = _camera;
        _camera->setName(str.str());
    }
    else
    {
        coVRConfig::instance()->channels[i].camera = new osg::Camera;
        coVRConfig::instance()->channels[i].camera->setName(str.str());
        coVRConfig::instance()->channels[i].camera->setView(this);

        coVRConfig::instance()->channels[i].camera->addChild(this->getScene()->getSceneData());
    }
    if (RenderToTexture)
    {
        osg::Camera::RenderTargetImplementation renderImplementation;

        // Voreingestellte Option für Render-Target aus Config auslesen
        std::string buf = coCoviseConfig::getEntry("COVER.Plugin.Vrml97.RTTImplementation");
        if (!buf.empty())
        {
            cerr << "renderImplementation: " << buf << endl;
            if (strcasecmp(buf.c_str(), "fbo") == 0)
                renderImplementation = osg::Camera::FRAME_BUFFER_OBJECT;
            if (strcasecmp(buf.c_str(), "pbuffer") == 0)
                renderImplementation = osg::Camera::PIXEL_BUFFER;
            if (strcasecmp(buf.c_str(), "pbuffer-rtt") == 0)
                renderImplementation = osg::Camera::PIXEL_BUFFER_RTT;
            if (strcasecmp(buf.c_str(), "fb") == 0)
                renderImplementation = osg::Camera::FRAME_BUFFER;
#if OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0)
            if (strcasecmp(buf.c_str(), "window") == 0)
                renderImplementation = osg::Camera::SEPARATE_WINDOW;
#else
            if (strcasecmp(buf.c_str(), "window") == 0)
                renderImplementation = osg::Camera::SEPERATE_WINDOW;
#endif
        }
        else
            renderImplementation = osg::Camera::FRAME_BUFFER_OBJECT;

        const osg::GraphicsContext *cg = coVRConfig::instance()->windows[coVRConfig::instance()->PBOs[pboNum].windowNum].context;
        if (!cg)
        {
            fprintf(stderr, "no graphics context for cover screen %d\n", i);
            return;
        }

        osg::Texture2D *renderTargetTexture = new osg::Texture2D;
        renderTargetTexture->setTextureSize(coVRConfig::instance()->PBOs[pboNum].PBOsx, coVRConfig::instance()->PBOs[pboNum].PBOsy);
        renderTargetTexture->setInternalFormat(GL_RGBA);
        //renderTargetTexture->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
        //	   renderTargetTexture->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
        renderTargetTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        renderTargetTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
        coVRConfig::instance()->PBOs[pboNum].renderTargetTexture = renderTargetTexture;
        coVRConfig::instance()->channels[i].camera->attach(osg::Camera::COLOR_BUFFER, renderTargetTexture);
        coVRConfig::instance()->channels[i].camera->setRenderTargetImplementation(renderImplementation);


        coVRConfig::instance()->channels[i].camera->setViewport(new osg::Viewport(0, 0,coVRConfig::instance()->PBOs[pboNum].PBOsx,
                    coVRConfig::instance()->PBOs[pboNum].PBOsy));
    }
    if (gc.get())
    {
        coVRConfig::instance()->channels[i].camera->setGraphicsContext(gc.get());
        const osg::GraphicsContext::Traits *traits = gc->getTraits();
        osg::ref_ptr<osgViewer::GraphicsWindow> gw = dynamic_cast<osgViewer::GraphicsWindow *>(gc.get());
        if (gw.get())
            gw->getEventQueue()->getCurrentEventState()->setWindowRectangle(traits->x, traits->y, traits->width, traits->height);
        if(!RenderToTexture)
        {
            // set viewport
            int viewportNumber = coVRConfig::instance()->channels[i].viewportNum;
            coVRConfig::instance()->channels[i].camera->setViewport(new osg::Viewport(coVRConfig::instance()->viewports[viewportNumber].viewportXMin * traits->width,
                coVRConfig::instance()->viewports[viewportNumber].viewportYMin * traits->height,
                (coVRConfig::instance()->viewports[viewportNumber].viewportXMax - coVRConfig::instance()->viewports[viewportNumber].viewportXMin) * traits->width,
                (coVRConfig::instance()->viewports[viewportNumber].viewportYMax - coVRConfig::instance()->viewports[viewportNumber].viewportYMin) * traits->height));
            coVRConfig::instance()->channels[i].camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        }
        else
        {
            coVRConfig::instance()->channels[i].camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }


        auto &cam = coVRConfig::instance()->channels[i].camera;
        cam->setClearColor(osg::Vec4(0.0, 0.4, 0.5, 0.0));
        cam->setClearStencil(0);
        if (RenderToOsgWindow)
        {
            GLenum buffer = traits->doubleBuffer ? GL_BACK : GL_FRONT;
            cam->setDrawBuffer(buffer);
            cam->setReadBuffer(buffer);
            cam->setInheritanceMask(osg::CullSettings::NO_VARIABLES);
        }
        else
        {
            cam->setDrawBuffer(GL_NONE);
            //cam->setReadBuffer(GL_NONE);
            cam->setInheritanceMask(cam->getInheritanceMask() | osg::CullSettings::NO_VARIABLES | osg::CullSettings::DRAW_BUFFER);
        }
        cam->setCullMask(~0 & ~(Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible
        cam->setCullMaskLeft(~0 & ~(Isect::Right|Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible and not right
        cam->setCullMaskRight(~0 & ~(Isect::Left|Isect::Collision|Isect::Intersection|Isect::NoMirror|Isect::Pick|Isect::Walk|Isect::Touch)); // cull everything that is visible and not Left
       
        //cam->getGraphicsContext()->getState()->checkGLErrors(osg::State::ONCE_PER_ATTRIBUTE);
    }
    else
    {
        cerr << "window " << coVRConfig::instance()->viewports[coVRConfig::instance()->channels[i].viewportNum].window << " of channel " << i << " not defined" << endl;
    }

    osg::DisplaySettings *ds = NULL;
    ds = _displaySettings.valid() ? _displaySettings.get() : osg::DisplaySettings::instance().get();
    if (i > 0) //we need different displaySettings for other channels
    {
        ds = new osg::DisplaySettings(*(_displaySettings.valid() ? _displaySettings.get() : osg::DisplaySettings::instance().get()));
    }
    if (!coVRConfig::instance()->glVersion.empty())
        ds->setGLContextVersion(coVRConfig::instance()->glVersion);
    if (!coVRConfig::instance()->glProfileMask.empty())
        ds->setGLContextProfileMask(std::stoi(coVRConfig::instance()->glProfileMask));
    if (!coVRConfig::instance()->glContextFlags.empty())
        ds->setGLContextFlags(std::stoi(coVRConfig::instance()->glContextFlags));

    // set up the use of stereo by default.
    ds->setStereo(conf.channels[i].stereo);
    if (coVRConfig::instance()->doMultisample())
        ds->setNumMultiSamples(coVRConfig::instance()->getMultisampleSamples());
    ds->setStereoMode((osg::DisplaySettings::StereoMode)coVRConfig::instance()->channels[i].stereoMode);
    ds->setMinimumNumStencilBits(coVRConfig::instance()->numStencilBits());
    coVRConfig::instance()->channels[i].ds = ds;
    coVRConfig::instance()->channels[i].camera->setDisplaySettings(ds);

    coVRConfig::instance()->channels[i].camera->setComputeNearFarMode(osgUtil::CullVisitor::DO_NOT_COMPUTE_NEAR_FAR);
    //coVRConfig::instance()->channels[i].camera->setCullingMode(osg::CullSettings::VIEW_FRUSTUM_CULLING);
    coVRConfig::instance()->channels[i].camera->setCullingMode(osg::CullSettings::ENABLE_ALL_CULLING);

    osgViewer::Renderer *renderer = new coVRRenderer(coVRConfig::instance()->channels[i].camera.get(), i);
    coVRConfig::instance()->channels[i].camera->setRenderer(renderer);
    if (i != 0)
    {
        addCamera(coVRConfig::instance()->channels[i].camera.get());
    }

    //PHILIP add LOD scale from configuration to deal with LOD updating
    float lodScale = coCoviseConfig::getFloat("COVER.LODScale", 0.0f);
    if (lodScale != 0)
    {
        coVRConfig::instance()->channels[i].camera->setLODScale(lodScale);
    }
}

void VRViewer::destroyChannels(int i)
{
    auto &conf = *coVRConfig::instance();
    auto &chan = conf.channels[i];

    removeCamera(chan.camera);
    chan.camera = nullptr;
}

void VRViewer::forceCompile()
{
    culling(false, osg::CullSettings::ENABLE_ALL_CULLING, true); // disable culling for one frame to load data to all GPUs
    for(int i=0;i<coVRConfig::instance()->channels.size();i++)
    {
        osg::GraphicsOperation *rop = coVRConfig::instance()->channels[i].camera->getRenderer();
        coVRRenderer *r = dynamic_cast<coVRRenderer *>(rop); 
        if(r)
        {
            r->setCompileOnNextDraw(true);
        }
    }
}

void
VRViewer::culling(bool enable, osg::CullSettings::CullingModeValues mode, bool once)
{
    if (once)
    {
        reEnableCulling = true;
    }
    for (int i = 0; i < coVRConfig::instance()->numChannels(); i++)
    {
        if (enable)
            coVRConfig::instance()->channels[i].camera->setCullingMode(mode);
        else
            coVRConfig::instance()->channels[i].camera->setCullingMode(osg::CullSettings::NO_CULLING);
    }
}

void
VRViewer::setClearColor(const osg::Vec4 &color)
{
    for (int i = 0; i < coVRConfig::instance()->numChannels(); ++i)
    {
		if(coVRConfig::instance()->channels[i].camera!=NULL)
            coVRConfig::instance()->channels[i].camera->setClearColor(color);
    }
}

std::pair<osg::Matrix, osg::Matrix>
computeViewProjFixedScreen(const osg::Matrix &viewerMat, osg::Vec3 eye, const osg::Vec3 &xyz, const osg::Vec3 &hpr, const osg::Vec2 &size, double near_val, double far_val, bool ortho, double worldAngle)
{
    osg::Matrix trans;
    // transform the screen to fit the xz-plane
    trans.makeTranslate(-xyz[0], -xyz[1], -xyz[2]);

    osg::Matrix euler;
    MAKE_EULER_MAT_VEC(euler, hpr);
    euler.invert(euler);

    osg::Matrix mat;
    mat.mult(trans, euler);

    euler.makeRotate(-worldAngle, osg::X_AXIS);
    mat.mult(euler, mat);

    // transform the left and right eye with this matrix
    eye = viewerMat.preMult(eye);

    eye = mat.preMult(eye);

    float dx = size[0], dz = size[1];

    auto projFromEye = [ortho, near_val, far_val, dx, dz](const osg::Vec3 &eye) -> osg::Matrix
    {
        float n_over_d; // near_val over dist -> Strahlensatz
        float c_dist = -eye[1];
        // relation near_val plane to screen plane
        if (ortho)
            n_over_d = 1.0;
        else
            n_over_d = near_val / c_dist;

        float c_right = n_over_d * (dx / 2.0 - eye[0]);
        float c_left = -n_over_d * (dx / 2.0 + eye[0]);
        float c_top = n_over_d * (dz / 2.0 - eye[2]);
        float c_bottom = -n_over_d * (dz / 2.0 + eye[2]);

        osg::Matrix ret;
        if (ortho)
            ret.makeOrtho(c_left, c_right, c_bottom, c_top, near_val, far_val);
        else
            ret.makeFrustum(c_left, c_right, c_bottom, c_top, near_val, far_val);
        return ret;
    };

    osg::Matrix proj = projFromEye(eye);

    // take the normal to the plane as orientation this is (0,1,0)
    osg::Matrix view;
    view.makeLookAt(eye, eye+osg::Vec3(0,1,0), osg::Vec3(0,0,1));
    view.preMult(mat);

    return std::make_pair(view, proj);
}


//OpenCOVER
void
VRViewer::setFrustumAndView(int i)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "VRViewer::setFrustum %d\n", i);

    coVRConfig *coco = coVRConfig::instance();
    channelStruct *currentChannel = &coco->channels[i];
    screenStruct *currentScreen = &coco->screens[currentChannel->screenNum];

    if (overwritePAndV)
    {
        currentChannel->camera->setViewMatrix(currentChannel->rightView);
        currentChannel->camera->setProjectionMatrix(currentChannel->rightProj);
        return;
    }

    auto res = computeFrustumAndView(i);

    currentChannel->leftView = res.left.view;
    currentChannel->leftProj = res.left.proj;
    currentChannel->rightView = res.right.view;
    currentChannel->rightProj = res.right.proj;
    currentChannel->camera->setViewMatrix(res.middle.view);
    currentChannel->camera->setProjectionMatrix(res.middle.proj);
}

osg::Vec3 VRViewer::eyeOffset(VRViewer::Eye eye) const
{
    osg::Vec3 off(0,0,0);
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

VRViewer::FrustaAndViews VRViewer::computeFrustumAndView(int i)
{

    coVRConfig *coco = coVRConfig::instance();
    channelStruct *currentChannel = &coco->channels[i];
    screenStruct *currentScreen = &coco->screens[currentChannel->screenNum];

    osg::Vec3 xyz; // center position of the screen
    osg::Vec3 hpr; // orientation of the screen
    osg::Matrix mat, trans, euler; // xform screencenter - world origin
    osg::Matrixf offsetMat;
    osg::Vec3 leftEye, rightEye, middleEye; // transformed eye position
    float dx, dz; // size of screen

    //othEyesDirOffset; == hpr
    //osg::Vec3  rightEyePosOffset(0.0,0.0,0.0), leftEyePosOffset(0.0,0.0,0.0);

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
        float change_delta = fabs(screen_angle[0].maxangle - screen_angle[0].minangle) * 0.01;

        if (fabs(currentScreen->hpr[screen_angle[0].hpr] - new_angle) > change_delta)
        {
            currentScreen->hpr[screen_angle[0].hpr] = new_angle;
            //		   cerr << "Cereal gives " << *screen_angle[0].value << endl;
            cerr << "Setting Screen angle " << screen_angle[0].hpr << " to " << new_angle << endl;
        }
    }

    // for rendering images for auto-stereoscopic displays (e.g. Tridelity)
    const float off = stereoOn ? currentChannel->stereoOffset : 0.f;
    if (stereoOn && currentChannel->stereo) {
        leftEye = eyeOffset(EyeLeft) + osg::Vec3(off,0,0);
        rightEye = eyeOffset(EyeRight) + osg::Vec3(off,0,0);
        middleEye = eyeOffset(EyeMiddle) + osg::Vec3(off,0,0);
    } else {
        leftEye = rightEye = middleEye = eyeOffset(EyeMiddle) + osg::Vec3(off,0,0);
    }

    if (!coco->trackedHMD && !coco->HMDMode)
    {
        osg::Matrix vm;
        if (currentChannel->fixedViewer)
            vm.makeTranslate(initialViewPos);
        else
            vm = viewMat;
        auto l = computeViewProjFixedScreen(vm, leftEye, xyz, hpr, osg::Vec2(dx,dz), coco->nearClip(), coco->farClip(), coco->orthographic(), coco->worldAngle());
        auto r = computeViewProjFixedScreen(vm, rightEye, xyz, hpr, osg::Vec2(dx,dz), coco->nearClip(), coco->farClip(), coco->orthographic(), coco->worldAngle());
        auto m = computeViewProjFixedScreen(vm, middleEye, xyz, hpr, osg::Vec2(dx,dz), coco->nearClip(), coco->farClip(), coco->orthographic(), coco->worldAngle());

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
        rightEye = viewMat.preMult(rightEye);
        leftEye = viewMat.preMult(leftEye);
        middleEye = viewMat.preMult(middleEye);
    }

    else if (coco->HMDMode) // weiss nicht was das fuer ein code ist
        // wenn screen center und dir 000 sind
    {

        // transform the screen to fit the xz-plane

        trans.makeTranslate(xyz[0], xyz[1], xyz[2]);
        trans.invert(trans);

        MAKE_EULER_MAT(euler, hpr[0], hpr[1], hpr[2]);
        euler.invert(euler);

        mat.mult(trans, euler);

        euler.makeRotate(-coco->worldAngle(), osg::X_AXIS);
        //euler.invertN(euler);
        mat.mult(euler, mat);

        rightEye += initialViewPos;
        leftEye += initialViewPos;
        middleEye += initialViewPos;

        // transform the left and right eye with this matrix
        rightEye = viewMat.preMult(rightEye);
        leftEye = viewMat.preMult(leftEye);
        middleEye = viewMat.preMult(middleEye);

        // add world angle
        osg::Matrixf rotAll, newDir;
        rotAll.makeRotate(-coco->worldAngle(), osg::X_AXIS);
        newDir.mult(viewMat, rotAll);

        // first set it if both channels were at the position of a mono channel
        // currentScreen->camera->setOffset(newDir.ptr(),0,0);

        viewPos = newDir.getTrans();

        // for stereo use the pfChanViewOffset to position it correctly
        // and to set the viewing direction (normal of the screen)

        //rightEyePosOffset=rightViewPos - viewPos;
        //leftEyePosOffset=leftViewPos - viewPos;
    }

    offsetMat = mat;

    FrustaAndViews res;
    // compute right frustum

    // dist of right channel eye to screen (absolute)
    if (coco->trackedHMD)
    {
        if (coco->orthographic())
        {
            res.right.proj.makeOrtho(-dx / 2.0, dx / 2.0, -dz / 2.0, dz / 2.0, coco->nearClip(), coco->farClip());
            res.left.proj.makeOrtho(-dx / 2.0, dx / 2.0, -dz / 2.0, dz / 2.0, coco->nearClip(), coco->farClip());
            res.middle.proj.makeOrtho(-dx / 2.0, dx / 2.0, -dz / 2.0, dz / 2.0, coco->nearClip(), coco->farClip());
        }
        else
        {
            if (currentScreen->lTan != -1)
            {
                float n = coco->nearClip();
                res.right.proj.makeFrustum(-n * currentScreen->lTan, n * currentScreen->rTan, -n * currentScreen->bTan, n * currentScreen->tTan, coco->nearClip(), coco->farClip());
                res.left.proj.makeFrustum(-n * currentScreen->lTan, n * currentScreen->rTan, -n * currentScreen->bTan, n * currentScreen->tTan, coco->nearClip(), coco->farClip());
                res.middle.proj.makeFrustum(-n * currentScreen->lTan, n * currentScreen->rTan, -n * currentScreen->bTan, n * currentScreen->tTan, coco->nearClip(), coco->farClip());
            }
            else
            {
				if (!coco->OpenVR_HMD)
				{
                    res.right.proj.makePerspective(coco->HMDViewingAngle, dx / dz, coco->nearClip(), coco->farClip());
                    res.left.proj.makePerspective(coco->HMDViewingAngle, dx / dz, coco->nearClip(), coco->farClip());
				}
                res.middle.proj.makePerspective(coco->HMDViewingAngle, dx / dz, coco->nearClip(), coco->farClip());
            }
        }
    }
    else
    {
        auto projFromEye = [coco, dx, dz](const osg::Vec3 &eye) -> osg::Matrix {
            float n_over_d; // near over dist -> Strahlensatz
            float c_dist = -eye[1];
            // relation near plane to screen plane
            if (coco->orthographic())
                n_over_d = 1.0;
            else
                n_over_d = coco->nearClip() / c_dist;

            float c_right = n_over_d * (dx / 2.0 - eye[0]);
            float c_left = -n_over_d * (dx / 2.0 + eye[0]);
            float c_top = n_over_d * (dz / 2.0 - eye[2]);
            float c_bottom = -n_over_d * (dz / 2.0 + eye[2]);

            osg::Matrix ret;
            if (coco->orthographic())
                ret.makeOrtho(c_left, c_right, c_bottom, c_top, coco->nearClip(), coco->farClip());
            else
                ret.makeFrustum(c_left, c_right, c_bottom, c_top, coco->nearClip(), coco->farClip());
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
        osg::Vec3 viewDir(0, 1, 0), viewUp(0, 0, 1);
        viewDir = viewMat.transform3x3(viewDir, viewMat);
        viewDir.normalize();
        //fprintf(stderr,"viewDir=[%f %f %f]\n", viewDir[0], viewDir[1], viewDir[2]);
        viewUp = viewMat.transform3x3(viewUp, viewMat);
        viewUp.normalize();
        res.right.view.makeLookAt(rightEye, rightEye + viewDir, viewUp);
        ///currentScreen->rightView=viewMat;
        res.left.view.makeLookAt(leftEye, leftEye + viewDir, viewUp);
        ///currentScreen->leftView=viewMat;

        res.middle.view.makeLookAt(middleEye, middleEye + viewDir, viewUp);
        ///currentScreen->camera->setViewMatrix(viewMat);
    }
    else
    {
        // take the normal to the plane as orientation this is (0,1,0)
        res.right.view.makeLookAt(osg::Vec3(rightEye[0], rightEye[1], rightEye[2]), osg::Vec3(rightEye[0], rightEye[1] + 1, rightEye[2]), osg::Vec3(0, 0, 1));
        res.right.view.preMult(offsetMat);

        res.left.view.makeLookAt(osg::Vec3(leftEye[0], leftEye[1], leftEye[2]), osg::Vec3(leftEye[0], leftEye[1] + 1, leftEye[2]), osg::Vec3(0, 0, 1));
        res.left.view.preMult(offsetMat);

        res.middle.view.makeLookAt(osg::Vec3(middleEye[0], middleEye[1], middleEye[2]), osg::Vec3(middleEye[0], middleEye[1] + 1, middleEye[2]), osg::Vec3(0, 0, 1));
        res.middle.view.preMult(offsetMat);
    }

    return res;
}

//OpenCOVER
void
VRViewer::readConfigFile()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRViewer::readConfigFile\n");

    /// ================= Moving screen initialisation =================

    // temp initialisation until the real address is determin in
    // the tracker routines
    static float init_cereal_analog_value = 0.0;

    screen_angle = NULL;
    string line = coCoviseConfig::getEntry("analogInput", "COVER.Input.CerealConfig.ScreenAngle");
    if (!line.empty())
    {
        char hpr;
        cerr << "Reading ScreenAngle";
        screen_angle = new angleStruct;
        screen_angle->value = &init_cereal_analog_value;

        // parameters:  Analog Input, min, max, minangle, maxangle, screen, HPR
        //if(sscanf(line,   "%d %f %f %f %f %d %c",

        const char *configEntry = "COVER.Input.CerealConfig.ScreenAngle";

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
    line = coCoviseConfig::getEntry("command", "COVER.Stereo");
    if (!line.empty())
    {
        strcpy(stereoCommand, line.c_str());
    }
    line = coCoviseConfig::getEntry("command", "COVER.Mono");
    if (!line.empty())
    {
        strcpy(monoCommand, line.c_str());
    }
    if (coVRConfig::instance()->stereoState())
    {
        if (strlen(stereoCommand) > 0)
            if (system(stereoCommand) == -1)
            {
                cerr << "VRViewer::readConfigFile: stereo command " << stereoCommand << " failed" << endl;
            }
    }
}

// OpenCOVER
void VRViewer::redrawHUD(double interval)
{
    // do a redraw of the scene no more then once per second (default)
    if (cover->currentTime() > lastFrameTime + interval)
    {
        unsyncedFrames++; // do not wait for slaves during temporary updates of the headup display
        frame(); // draw one frame
        VRWindow::instance()->update();
        VRWindow::instance()->updateContents();
    }
}

void VRViewer::disableSync()
{
    unsyncedFrames = 1000000;
}

bool VRViewer::mustDraw()
{
    return coVRConfig::instance()->numChannels() > 0;
}

// OpenCOVER
void VRViewer::frame(double simulationTime)
{

    lastFrameTime = cover->currentTime();

    double startFrame = elapsedTime();
    if (mustDraw())
    {
        osgViewer::Viewer::frame(cover->frameTime());
    }
    else
    {
        if (OpenCOVER::instance()->initDone())
            coVRPluginList::instance()->clusterSyncDraw();
        if (VRViewer::unsyncedFrames <= 0)
        {
            coVRMSController::instance()->syncDraw();
        }
        else
        {
            VRViewer::unsyncedFrames--;
        }
    }
    if (getViewerStats()->collectStats("frame"))
    {
        double endFrame = elapsedTime();
        osg::FrameStamp *frameStamp = getViewerFrameStamp();
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "frame begin time", startFrame);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "frame end time", endFrame);
        getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "frame time taken", endFrame - startFrame);
    }

    if (reEnableCulling)
    {
        culling(true);
        reEnableCulling = false;
    }

    VRWindow::instance()->updateContents();
}

void VRViewer::startThreading()
{
    //DebugBreak();
    if (_threadsRunning)
        return;

    osg::notify(osg::INFO) << "VRViewer::VRstartThreading() - starting threading" << std::endl;

    // release any context held by the main thread.
    releaseContext();

    _threadingModel = _threadingModel == AutomaticSelection ? suggestBestThreadingModel() : _threadingModel;

    Contexts contexts;
    getContexts(contexts);

    Cameras cameras;
    getCameras(cameras);

    unsigned int numThreadsOnStartBarrier = 0;
    unsigned int numThreadsOnEndBarrier = 0;
    switch (_threadingModel)
    {
    case (SingleThreaded):
        numThreadsOnStartBarrier = 1;
        numThreadsOnEndBarrier = 1;
        return;
    case (CullDrawThreadPerContext):
        numThreadsOnStartBarrier = contexts.size() + 1;
        numThreadsOnEndBarrier = contexts.size() + 1;
        break;
    case (DrawThreadPerContext):
        numThreadsOnStartBarrier = 1;
        numThreadsOnEndBarrier = 1;
        break;
    case (CullThreadPerCameraDrawThreadPerContext):
        numThreadsOnStartBarrier = cameras.size() + 1;
        numThreadsOnEndBarrier = 1;
        break;
    default:
        osg::notify(osg::NOTICE) << "Error: Threading model not selected" << std::endl;
        return;
    }

    // using multi-threading so make sure that new objects are allocated with thread safe ref/unref
#if OSG_VERSION_LESS_THAN(3, 5, 5)
    osg::Referenced::setThreadSafeReferenceCounting(true);
#endif

    Scenes scenes;
    getScenes(scenes);
    for (Scenes::iterator scitr = scenes.begin();
         scitr != scenes.end();
         ++scitr)
    {
        if ((*scitr)->getSceneData())
        {
            osg::notify(osg::INFO) << "Making scene thread safe" << std::endl;

            // make sure that existing scene graph objects are allocated with thread safe ref/unref
            (*scitr)->getSceneData()->setThreadSafeRefUnref(true);

            // update the scene graph so that it has enough GL object buffer memory for the graphics contexts that will be using it.
            (*scitr)->getSceneData()->resizeGLObjectBuffers(osg::DisplaySettings::instance()->getMaxNumberOfGraphicsContexts());
        }
    }

    int numProcessors = OpenThreads::GetNumberOfProcessors();
    bool affinity = numProcessors > 1;

    Contexts::iterator citr;

    unsigned int numViewerDoubleBufferedRenderingOperation = 0;

    bool graphicsThreadsDoesCull = _threadingModel == CullDrawThreadPerContext || _threadingModel == SingleThreaded;

    for (Cameras::iterator camItr = cameras.begin();
         camItr != cameras.end();
         ++camItr)
    {
        osg::Camera *camera = *camItr;
        osgViewer::Renderer *renderer = dynamic_cast<osgViewer::Renderer *>(camera->getRenderer());
        if (renderer)
        {
            renderer->setGraphicsThreadDoesCull(graphicsThreadsDoesCull);
            renderer->setDone(false);
            ++numViewerDoubleBufferedRenderingOperation;
        }
    }

    if (_threadingModel == CullDrawThreadPerContext)
    {
        _startRenderingBarrier = 0;
        _endRenderingDispatchBarrier = 0;
        _endDynamicDrawBlock = 0;
    }
    else if (_threadingModel == DrawThreadPerContext || _threadingModel == CullThreadPerCameraDrawThreadPerContext)
    {
        _startRenderingBarrier = 0;
        _endRenderingDispatchBarrier = 0;
        _endDynamicDrawBlock = new osg::EndOfDynamicDrawBlock(numViewerDoubleBufferedRenderingOperation);

#ifndef OSGUTIL_RENDERBACKEND_USE_REF_PTR
        if (!osg::Referenced::getDeleteHandler())
            osg::Referenced::setDeleteHandler(new osg::DeleteHandler(2));
        else
            osg::Referenced::getDeleteHandler()->setNumFramesToRetainObjects(2);
#endif
    }

    if (numThreadsOnStartBarrier > 1)
    {
        _startRenderingBarrier = new osg::BarrierOperation(numThreadsOnStartBarrier, osg::BarrierOperation::NO_OPERATION);
        _startRenderingBarrier->setName("_startRenderingBarrier");
    }

    if (numThreadsOnEndBarrier > 1)
    {
        _endRenderingDispatchBarrier = new osg::BarrierOperation(numThreadsOnEndBarrier, osg::BarrierOperation::NO_OPERATION);
        _endRenderingDispatchBarrier->setName("_endRenderingDispatchBarrier");
    }

    osg::ref_ptr<osg::BarrierOperation> swapReadyBarrier = contexts.empty() ? 0 : new osg::BarrierOperation(contexts.size(), osg::BarrierOperation::NO_OPERATION);
    if (swapReadyBarrier != NULL)
        swapReadyBarrier->setName("swapReadyBarrier");

    osg::ref_ptr<osg::SwapBuffersOperation> swapOp = new osg::SwapBuffersOperation();
    if (swapOp != NULL)
        swapOp->setName("swapOp");

    typedef std::map<OpenThreads::Thread *, int> ThreadAffinityMap;
    ThreadAffinityMap threadAffinityMap;

    // begin OpenCOVER
    osg::BarrierOperation *finishOp = new osg::BarrierOperation(contexts.size(), osg::BarrierOperation::GL_FINISH);
    finishOp->setName("osg::BarrierOperation::GL_FINISH");
    // end OpenCOVER

    unsigned int processNum = 1;
    for (citr = contexts.begin();
         citr != contexts.end();
         ++citr, ++processNum)
    {
        osg::GraphicsContext *gc = (*citr);
        if (!gc->isRealized())
        {
            //OSG_INFO<<"ViewerBase::startThreading() : Realizng window "<<gc<<std::endl;
            gc->realize();
            if (gc->valid())
            {
                gc->makeCurrent();
                if (_realizeOperation.valid() && gc->valid())
                {
                    (*_realizeOperation)(gc);
                }
                gc->releaseContext();
            }
        }

        gc->getState()->setDynamicObjectRenderingCompletedCallback(_endDynamicDrawBlock.get());

        // create the a graphics thread for this context
        gc->createGraphicsThread();

        if (affinity)
            gc->getGraphicsThread()->setProcessorAffinity(-1);
        threadAffinityMap[gc->getGraphicsThread()] = -1;

        // add the startRenderingBarrier
        if (_threadingModel == CullDrawThreadPerContext && _startRenderingBarrier.valid())
            gc->getGraphicsThread()->add(_startRenderingBarrier.get());

        // begin OpenCOVER
        // add the rendering operation itself.
        ViewerRunOperations *vr = new ViewerRunOperations();
        vr->setName("ViewerRunOperations");
        gc->getGraphicsThread()->add(vr);
        // end OpenCOVER

        if (_threadingModel == CullDrawThreadPerContext && _endBarrierPosition == BeforeSwapBuffers && _endRenderingDispatchBarrier.valid())
        {
            // add the endRenderingDispatchBarrier
            gc->getGraphicsThread()->add(_endRenderingDispatchBarrier.get());
        }
        gc->getGraphicsThread()->add(finishOp); // make sure all threads have finished drawing

        PreSwapBuffersOp *pso = new PreSwapBuffersOp(processNum - 1);
        gc->getGraphicsThread()->add(pso);

        // wait on a Barrier so that all rendering threads are ready to swap, otherwise we are not sure that after a remote sync, we can swap rightaway
        if (coVRConfig::instance()->numChannels() > 1)
        {

            if (swapReadyBarrier.valid())
                gc->getGraphicsThread()->add(swapReadyBarrier.get());
        }

        if (processNum == 1) // first context, add sync operation
        {
            RemoteSyncOp *rso = new RemoteSyncOp();
            rso->setName("RemoteSyncOp");
            gc->getGraphicsThread()->add(rso);
        }
        // end OpenCOVER

        if (swapReadyBarrier.valid())
            gc->getGraphicsThread()->add(swapReadyBarrier.get());

        // add the swap buffers
        gc->getGraphicsThread()->add(swapOp.get());

        // begin OpenCOVER
        PostSwapBuffersOp *psbo = new PostSwapBuffersOp(processNum - 1);
        psbo->setName("PostSwapBuffersOp");
        gc->getGraphicsThread()->add(psbo);
        // end OpenCOVER

        if (_threadingModel == CullDrawThreadPerContext && _endBarrierPosition == AfterSwapBuffers && _endRenderingDispatchBarrier.valid())
        {
            // add the endRenderingDispatchBarrier
            gc->getGraphicsThread()->add(_endRenderingDispatchBarrier.get());
        }
    }

    if (_threadingModel == CullThreadPerCameraDrawThreadPerContext && numThreadsOnStartBarrier > 1)
    {
        Cameras::iterator camItr;

        for (camItr = cameras.begin();
             camItr != cameras.end();
             ++camItr, ++processNum)
        {
            osg::Camera *camera = *camItr;
            camera->createCameraThread();

            if (affinity)
                camera->getCameraThread()->setProcessorAffinity(-1);
            threadAffinityMap[camera->getCameraThread()] = -1;

            osg::GraphicsContext *gc = camera->getGraphicsContext();

            // add the startRenderingBarrier
            if (_startRenderingBarrier.valid())
                camera->getCameraThread()->add(_startRenderingBarrier.get());

            osgViewer::Renderer *renderer = dynamic_cast<osgViewer::Renderer *>(camera->getRenderer());
            renderer->setName("Renderer");
            renderer->setGraphicsThreadDoesCull(false);
            camera->getCameraThread()->add(renderer);

            if (_endRenderingDispatchBarrier.valid())
            {
                // add the endRenderingDispatchBarrier
                gc->getGraphicsThread()->add(_endRenderingDispatchBarrier.get());
            }
        }

        for (camItr = cameras.begin();
             camItr != cameras.end();
             ++camItr)
        {
            osg::Camera *camera = *camItr;
            if (camera->getCameraThread() && !camera->getCameraThread()->isRunning())
            {
                osg::notify(osg::INFO) << "  camera->getCameraThread()-> " << camera->getCameraThread() << std::endl;
                camera->getCameraThread()->startThread();
            }
        }
    }

#if 0    
    if (affinity) 
    {
        OpenThreads::SetProcessorAffinityOfCurrentThread(-1);
        if (_scene.valid() && _scene->getDatabasePager())
        {
#if 0        
            _scene->getDatabasePager()->setProcessorAffinity(1);
#else
            _scene->getDatabasePager()->setProcessorAffinity(0);
#endif
        }
    }
#endif

#if 0
    if (affinity)
    {
        for(ThreadAffinityMap::iterator titr = threadAffinityMap.begin();
            titr != threadAffinityMap.end();
            ++titr)
        {
            titr->first->setProcessorAffinity(titr->second);
        }
    }
#endif

    for (citr = contexts.begin();
         citr != contexts.end();
         ++citr)
    {
        osg::GraphicsContext *gc = (*citr);
        if (gc->getGraphicsThread() && !gc->getGraphicsThread()->isRunning())
        {
            osg::notify(osg::INFO) << "  gc->getGraphicsThread()->startThread() " << gc->getGraphicsThread() << std::endl;
            fprintf(stderr,"setProcessorAffinity to %d\n",coVRConfig::instance()->lockToCPU());
            gc->getGraphicsThread()->setProcessorAffinity(coVRConfig::instance()->lockToCPU());
            gc->getGraphicsThread()->startThread();
            // OpenThreads::Thread::YieldCurrentThread();
        }
    }

    _threadsRunning = true;

    setAffinity();

    osg::notify(osg::INFO) << "Set up threading" << std::endl;
}

void VRViewer::getCameras(Cameras &cameras, bool onlyActive)
{
    cameras.clear();
    if (_camera.valid() && (!onlyActive || (_camera->getGraphicsContext() && _camera->getGraphicsContext()->valid())))
    {
        cameras.push_back(_camera.get());
    }
    for (std::list<osg::ref_ptr<osg::Camera> >::iterator itr = myCameras.begin();
            itr != myCameras.end();
            ++itr)
    {
        if (!onlyActive || (itr->get()->getGraphicsContext() && itr->get()->getGraphicsContext()->valid()))
            cameras.push_back((*itr).get());
    }
}
//from osgUtil Viewer.cpp
struct LessGraphicsContext
{
    bool operator()(const osg::GraphicsContext *lhs, const osg::GraphicsContext *rhs) const
    {
        int screenLeft = lhs->getTraits() ? lhs->getTraits()->screenNum : 0;
        int screenRight = rhs->getTraits() ? rhs->getTraits()->screenNum : 0;
        if (screenLeft < screenRight)
            return true;
        if (screenLeft > screenRight)
            return false;

        screenLeft = lhs->getTraits() ? lhs->getTraits()->x : 0;
        screenRight = rhs->getTraits() ? rhs->getTraits()->x : 0;
        if (screenLeft < screenRight)
            return true;
        if (screenLeft > screenRight)
            return false;

        screenLeft = lhs->getTraits() ? lhs->getTraits()->y : 0;
        screenRight = rhs->getTraits() ? rhs->getTraits()->y : 0;
        if (screenLeft < screenRight)
            return true;
        if (screenLeft > screenRight)
            return false;

        return lhs < rhs;
    }
};

void VRViewer::getContexts(Contexts &contexts, bool onlyValid)
{
    typedef std::set<osg::GraphicsContext *> ContextSet;
    ContextSet contextSet;
    contexts.clear();
    if (_camera.valid() && _camera->getGraphicsContext() && (_camera->getGraphicsContext()->valid() || !onlyValid))
    {
        if (contextSet.insert(_camera->getGraphicsContext()).second)
            contexts.push_back(_camera->getGraphicsContext());
    }

    //if (!onlyValid)
    {
        for (size_t i=0; i<coVRConfig::instance()->numWindows(); ++i)
        {
            if (contextSet.insert(coVRConfig::instance()->windows[i].context).second)
                contexts.push_back(coVRConfig::instance()->windows[i].context);
        }
    }

    for (std::list<osg::ref_ptr<osg::Camera> >::iterator itr = myCameras.begin();
            itr != myCameras.end();
            ++itr)
    {
        if ((*itr)->getGraphicsContext() && ((*itr)->getGraphicsContext()->valid() || !onlyValid))
        {
            if (contextSet.find((*itr)->getGraphicsContext()) == contextSet.end()) // only add context if not already in list
            {
                if (contextSet.insert((*itr)->getGraphicsContext()).second)
                    contexts.push_back((*itr)->getGraphicsContext());
            }
        }
    }
}

void VRViewer::getScenes(Scenes &scenes, bool /*onlyValid*/)
{
    scenes.clear();
    scenes.push_back(_scene.get());
}

// OpenCOVER
void VRViewer::addCamera(osg::Camera *camera)
{
    bool threadsWereRuinning = _threadsRunning;
    if (threadsWereRuinning)
        stopThreading();

    myCameras.push_back(camera);

    osg::GraphicsContext *gc = camera->getGraphicsContext();
    if (gc && !gc->isRealized())
    {
        //OSG_INFO<<"ViewerBase::startThreading() : Realizng window "<<gc<<std::endl;
        gc->realize();
        if (_realizeOperation.valid() && gc->valid())
        {
            gc->makeCurrent();
            (*_realizeOperation)(gc);
            gc->releaseContext();
        }
    }

    if (threadsWereRuinning)
        startThreading();
}
// OpenCOVER
void VRViewer::removeCamera(osg::Camera *camera)
{

    bool threadsWereRuinning = _threadsRunning;
    if (threadsWereRuinning)
        stopThreading();

    for (std::list<osg::ref_ptr<osg::Camera> >::iterator itr = myCameras.begin();
         itr != myCameras.end();
         ++itr)
    {
        if (itr->get() == camera)
        {
            itr = myCameras.erase(itr);
            break;
        }
    }

    if (threadsWereRuinning)
        startThreading();
}

void VRViewer::renderingTraversals()
{
    bool _outputMasterCameraLocation = false;
    if (_outputMasterCameraLocation)
    {
        Views views;
        getViews(views);

        for (Views::iterator itr = views.begin();
             itr != views.end();
             ++itr)
        {
            osgViewer::View *view = *itr;
            if (view)
            {
                const osg::Matrixd &m = view->getCamera()->getInverseViewMatrix();
                OSG_NOTICE << "View " << view << ", Master Camera position(" << m.getTrans().x() << "," << m.getTrans().y() << "," << m.getTrans().z() << ","
                           << "), rotation(" << m.getRotate().x() << "," << m.getRotate().y() << "," << m.getRotate().z() << "," << m.getRotate().w() << ","
                           << ")" << std::endl;
            }
        }
    }

    Contexts contexts;
    getContexts(contexts);

    // check to see if windows are still valid
    checkWindowStatus();
    if (_done)
        return;

    double beginRenderingTraversals = elapsedTime();

    osg::FrameStamp *frameStamp = getViewerFrameStamp();

    if (getViewerStats() && getViewerStats()->collectStats("scene"))
    {
        unsigned int frameNumber = frameStamp ? frameStamp->getFrameNumber() : 0;

        Views views;
        getViews(views);
        for (Views::iterator vitr = views.begin();
             vitr != views.end();
             ++vitr)
        {
            osgViewer::View *view = *vitr;
            osg::Stats *stats = view->getStats();
            osg::Node *sceneRoot = view->getSceneData();
            if (sceneRoot && stats)
            {
                osgUtil::StatsVisitor statsVisitor;
                sceneRoot->accept(statsVisitor);
                statsVisitor.totalUpStats();

                unsigned int unique_primitives = 0;
                osgUtil::Statistics::PrimitiveCountMap::iterator pcmitr;
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
        osgViewer::Scene *scene = *sitr;
        osgDB::DatabasePager *dp = scene ? scene->getDatabasePager() : 0;
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

    // osg::notify(osg::NOTICE)<<std::endl<<"Start frame"<<std::endl;

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
        osg::Camera *camera = *camItr;
        osgViewer::Renderer *renderer = dynamic_cast<osgViewer::Renderer *>(camera->getRenderer());
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

    // osg::notify(osg::NOTICE)<<"Joing _endRenderingDispatchBarrier block "<<_endRenderingDispatchBarrier.get()<<std::endl;

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

    if (OpenCOVER::instance()->initDone())
    {
        coVRPluginList::instance()->clusterSyncDraw();

        int WindowNum = 0;
        for (itr = contexts.begin();
             itr != contexts.end();
             ++itr)
        {
            if (!((*itr)->getGraphicsThread()) && (*itr)->valid())
            {
                doneMakeCurrentInThisThread = true;
                makeCurrent(*itr);
                coVRPluginList::instance()->preSwapBuffers(WindowNum);
                WindowNum++;
            }
        }
    }

    if (sync)
    {
        //do sync here before swapBuffers

        // sync multipc instances
        if (VRViewer::unsyncedFrames <= 0)
        {
            double beginSync = elapsedTime();
            coVRMSController::instance()->syncDraw();
            if (getViewerStats()->collectStats("sync"))
            {
                double endSync = elapsedTime();
                getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "sync begin time", beginSync);
                getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "sync end time", endSync);
                getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "sync time taken", endSync - beginSync);
            }
        }
        else
            VRViewer::unsyncedFrames--;
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

    if (OpenCOVER::instance()->initDone())
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
                coVRPluginList::instance()->postSwapBuffers(WindowNum);
                WindowNum++;
            }
        }
    }

    for (Scenes::iterator sitr = scenes.begin();
         sitr != scenes.end();
         ++sitr)
    {
        osgViewer::Scene *scene = *sitr;
        osgDB::DatabasePager *dp = scene ? scene->getDatabasePager() : 0;
        if (dp)
        {
            dp->signalEndFrame();
        }
    }

    // wait till the dynamic draw is complete.
    if (_endDynamicDrawBlock.valid())
    {
        // osg::Timer_t startTick = osg::Timer::instance()->tick();
        _endDynamicDrawBlock->block();
        // osg::notify(osg::NOTICE)<<"Time waiting "<<osg::Timer::instance()->delta_m(startTick, osg::Timer::instance()->tick())<<std::endl;;
    }

    if (_releaseContextAtEndOfFrameHint && doneMakeCurrentInThisThread)
    {
        //OSG_NOTICE<<"Doing release context"<<std::endl;
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
}

void VRViewer::glContextOperation(osg::GraphicsContext *ctx)
{
    if (clearWindow || numClears > 0)
    {
        clearWindow = false;
        if (numClears == 0)
        {
            numClears = coVRConfig::instance()->numWindows() * 2;
        }
        --numClears;
        bool emb = std::string(ctx->className()) == "GraphicsWindowEmbedded";
        clearGlWindow(!emb);
        ctx->makeCurrent();
    } // OpenCOVER end

#ifdef USE_NVX_INFO
    if (GLEW_NVX_gpu_memory_info)
    {
#ifdef GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
        if (statsDisplay)
            statsDisplay->enableGpuStats(true);
        if (getViewerStats() && getViewerStats()->collectStats("frame_rate"))
        {
            osg::FrameStamp *frameStamp = getViewerFrameStamp();

            GLint mem=0, obj=0, avail=0;
            glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX, &obj);
            glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX, &mem);
            glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &avail);
            //std::cerr << "AVAIL=" << avail << ", EVICTED " << windowNumber << ": obj=" << obj << ", mem=" << mem << std::endl;

            getViewerStats()->setAttribute(frameStamp->getFrameNumber(), "GPU mem free", avail);
        }
#endif
    }
#endif
}

template <typename Cameras, typename F>
osg::Node::NodeMask getCullMaskImpl(Cameras const& cameras, F func)
{
    if (cameras.size() == 0)
    {
        return 0x0;
    }

    typename Cameras::const_iterator camItr = cameras.begin();
    osg::Node::NodeMask result = ((*camItr)->*func)();
    for (;
         camItr != cameras.end();
         ++camItr)
    {
        assert( (result & ((*camItr)->*func)()) == result );
    }

    return result;
}

osg::Node::NodeMask
VRViewer::getCullMask() /*const*/
{
    Cameras cameras;
    getCameras(cameras);
    return getCullMaskImpl(cameras, &osg::Camera::getCullMask);
}

osg::Node::NodeMask
VRViewer::getCullMaskLeft() /*const*/
{
    Cameras cameras;
    getCameras(cameras);
    return getCullMaskImpl(cameras, &osg::Camera::getCullMaskLeft);
}

osg::Node::NodeMask
VRViewer::getCullMaskRight() /*const*/
{
    Cameras cameras;
    getCameras(cameras);
    return getCullMaskImpl(cameras, &osg::Camera::getCullMaskRight);
}

}
