/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_VIEWER_H
#define VR_VIEWER_H

/*! \file
 \brief  tracked stereo viewer class

 \author Daniela Rainer
 \author Uwe Woessner <woessner@hlrs.de> (osgProducer port)
 \author Martin Aumueller <aumueller@uni-koeln.de> (osgViewer port)
 \author (C) 1996
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   20.08.1997
 \date   2004 (osgProducer)
 \date   2007 (osgViewer)
 */

#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <util/common.h>

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include "ui/Owner.h"

namespace osg
{
class Group;
class DisplaySettings;
}

namespace opencover
{
class MSEventHandler;
class MarkerTrackingMarker;
class angleStruct;
class coVRStatsDisplay;
class InitGLOperation;

class COVEREXPORT VRViewer : public osgViewer::Viewer, public ui::Owner
{
    friend class OpenCOVER;
public:
    /** Whether this cluster instance/rank has any framebuffers to draw into.*/
    static bool mustDraw();

    /** Updated the scene.  Handle any queued up events, do an update traversal and set the CameraGroup's setViewByMatrix if any camera manipulators are active.*/
    virtual bool update();

    /** Dispatch the cull and draw for each of the Camera's for this frame.*/
    void frame(double simulationTime = USE_REFERENCE_TIME) override;

    bool handleEvents();

    void redrawHUD(double interval);

    static int unsyncedFrames;

    /** Start any threads required by the viewer.*/
    void startThreading() override;

    void addCamera(osg::Camera *camera);
    void removeCamera(osg::Camera *camera);

    void setClearColor(const osg::Vec4 &color);
    void setRenderToTexture(bool);
    void flipStereo();

    void setFullscreen(bool state);
    bool isFullscreen() const;

    /** initiate shut down */
    void disableSync();

private:
    static VRViewer *s_singleton;
    MSEventHandler *myeh = nullptr;

    // stereo parameters
    char stereoCommand[500];
    char monoCommand[500];
    double lastFrameTime;

    void renderingTraversals() override;

    // view
    osg::Vec3 viewPos, viewDir; //, viewYDir;
    osg::Vec3 initialViewPos;
    osg::Matrix viewMat;
    osgViewer::StatsHandler *statsHandler = nullptr;

    angleStruct *screen_angle; // Screen angle: IWR movable screen

    void readConfigFile();

    void setStereoMode();

    void createChannels(int i);
    void destroyChannels(int i);

    osg::Vec4 backgroundColor;

    bool arTracking;
    MarkerTrackingMarker *vpMarker;
    bool overwritePAndV;
    bool reEnableCulling;
    
    osg::Geometry *distortionMesh(const char *fileName);
    void createViewportCameras(int i);
    void destroyViewportCameras(int i);
    void createBlendingCameras(int i);
    void destroyBlendingCameras(int i);
    float requestedSeparation, separation;
    int animateSeparation;
    bool stereoOn;

    std::list<osg::ref_ptr<osg::Camera> > myCameras;
    void setAffinity();

public:
    struct FrustumAndView
    {
        osg::Matrix view;
        osg::Matrix proj;
    };
    struct FrustaAndViews
    {
        FrustumAndView left;
        FrustumAndView right;
        FrustumAndView middle;
    };

    FrustaAndViews computeFrustumAndView(int i);

    void setFrustumAndView(int i);

    enum Eye {
        EyeMiddle,
        EyeLeft,
        EyeRight,
    };

    osg::Vec3 eyeOffset(Eye eye) const;

    static VRViewer *instance();
    void setSeparation(float stereoSep);
    VRViewer();

    virtual ~VRViewer();

    void config();
    void unconfig();

    void getCameras(Cameras &cameras, bool onlyActive = true) override;
    void getContexts(Contexts &contexts, bool onlyValid = true) override;
    void getScenes(Scenes &scenes, bool onlyValid = true) override;

    void updateViewerMat(const osg::Matrix &mat);
    void setViewerMat(osg::Matrix &mat)
    {
        viewMat = mat;
    };
    osg::Matrix &getViewerMat()
    {
        return (viewMat);
    };
    osg::Vec3 &getViewerPos()
    {
        return viewPos;
    };
    osg::Vec3 &getInitialViewerPos()
    {
        return initialViewPos;
    };
    void setInitialViewerPos(osg::Vec3 &po)
    {
        initialViewPos = po;
    };

    void culling(bool enable, osg::CullSettings::CullingModeValues mode = osg::CullSettings::ENABLE_ALL_CULLING, bool once = false);
    
    void forceCompile();
    // request angle for movable screen
    angleStruct *getAngleStruct()
    {
        return screen_angle;
    };
    float getSeparation()
    {
        return separation;
    };

    osg::Vec4 getBackgroundColor()
    {
        return backgroundColor;
    };

    bool clearWindow = true; // if set to true, the whole window is cleared once
    int numClears = 0;

    void overwriteViewAndProjectionMatrix(bool state)
    {
        overwritePAndV = state;
    };
    bool isMatrixOverwriteOn()
    {
        return overwritePAndV;
    };

    void glContextOperation(osg::GraphicsContext *ctx);

    // assert: cull masks of all channels are equal!
    osg::Node::NodeMask getCullMask() /*const*/;
    osg::Node::NodeMask getCullMaskLeft() /*const*/;
    osg::Node::NodeMask getCullMaskRight() /*const*/;
    std::vector<osg::ref_ptr<osg::Camera>> viewportCamera;
    std::vector<osg::ref_ptr<osg::Camera>> blendingCamera;

    bool m_fullscreen = false;

    coVRStatsDisplay *statsDisplay = nullptr;
    InitGLOperation *m_initGlOp = nullptr;

    bool m_requireGlFinish = true;
};

std::pair<osg::Matrix, osg::Matrix> COVEREXPORT computeViewProjFixedScreen(const osg::Matrix &viewerMat, osg::Vec3 eye, const osg::Vec3 &xyz, const osg::Vec3 &hpr, const osg::Vec2 &size, double near, double far, bool ortho=false, double worldAngle=0.f);
}
#endif
