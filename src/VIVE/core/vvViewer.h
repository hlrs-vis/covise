/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <vsg/app/Viewer.h>

#include <util/common.h>

#include "ui/Owner.h"

namespace vv
{
class Group;
class DisplaySettings;
}

namespace vive
{
class MSEventHandler;
class MarkerTrackingMarker;
class angleStruct;
class vvStatsDisplay;
class InitGLOperation;


/// Perspective is a ProjectionMatrix that implements the gluPerspective model for setting the projection matrix.
class VVCORE_EXPORT OffAxis : public vsg::Inherit<vsg::ProjectionMatrix, OffAxis>
{
public:
    OffAxis()
    {
    }

    vsg::dmat4 proj;
    explicit OffAxis(const OffAxis& p, const vsg::CopyOp& copyop = {}) :
        Inherit(p, copyop),
        proj(p.proj)
    {
    }


    vsg::ref_ptr<Object> clone(const vsg::CopyOp& copyop = {}) const override { return OffAxis::create(*this, copyop); }

    vsg::dmat4 transform() const override { return proj; }


};

class VVCORE_EXPORT vvViewer : public vsg::Inherit<vsg::Viewer, vvViewer>, public ui::Owner
{
    friend class vvVIVE;
public:
    /** Whether this cluster instance/rank has any framebuffers to draw into.*/
    static bool mustDraw();

    /** Updated the scene.  Handle any queued up events, do an update traversal and set the CameraGroup's setViewByMatrix if any camera manipulators are active.*/
    virtual void vvUpdate();



    void redrawHUD(double interval);

    static int unsyncedFrames;


    void addCamera(vsg::Camera *camera);

    void setClearColor(const vsg::vec4 &color);
    void setRenderToTexture(bool);
    void flipStereo();

    void setFullscreen(bool state);
    bool isFullscreen() const;

    /** initiate shut down */
    void disableSync();

    struct FrustumAndView
    {
        vsg::dmat4 view;
        vsg::dmat4 proj;
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

    vsg::dvec3 eyeOffset(Eye eye) const;

    static vvViewer *instance();
    static void destroy();
    void setSeparation(float stereoSep);
    vvViewer();

    virtual ~vvViewer();

    void config();
    void unconfig();

    void updateViewerMat(const vsg::dmat4 &mat);
    void setViewerMat(vsg::dmat4 &mat)
    {
        viewMat = mat;
    };
    vsg::dmat4 &getViewerMat()
    {
        return (viewMat);
    };
    vsg::dvec3 &getViewerPos()
    {
        return viewPos;
    };
    vsg::dvec3 &getInitialViewerPos()
    {
        return initialViewPos;
    };
    void setInitialViewerPos(vsg::dvec3 &po)
    {
        initialViewPos = po;
    };

    
    // request angle for movable screen
    angleStruct *getAngleStruct()
    {
        return screen_angle;
    };
    float getSeparation()
    {
        return separation;
    };

    vsg::vec4 getBackgroundColor()
    {
        return backgroundColor;
    };

    bool clearWindow = true; // if set to true, the whole window is cleared once
    int numClears = 0;


    bool m_fullscreen = false;
    bool compileNode(vsg::ref_ptr<vsg::Node>& node);

    void InitialCompile();

private:
    static vsg::ref_ptr<vvViewer> s_singleton;
    vsg::CommandGraphs commandGraphs;
    vsg::ref_ptr<vsg::MatrixTransform> loadedScene;

    // stereo parameters
    char stereoCommand[500];
    char monoCommand[500];
    double lastFrameTime;

    // view
    vsg::dvec3 viewPos, viewDir; //, viewYDir;
    vsg::dvec3 initialViewPos;
    vsg::dmat4 viewMat;
    //vvViewer::StatsHandler *statsHandler = nullptr;

    angleStruct* screen_angle; // Screen angle: IWR movable screen

    void readConfigFile();

    void setStereoMode();

    void createChannels(int i);
    void destroyChannels(int i);

    vsg::vec4 backgroundColor;

    bool arTracking;
    MarkerTrackingMarker* vpMarker;
    bool overwritePAndV;
    bool reEnableCulling;

    float requestedSeparation=60.0, separation=60.0;
    int animateSeparation;
    bool stereoOn;

    std::list<vsg::ref_ptr<vsg::Camera> > myCameras;
    void setAffinity();
    bool InitialCompileDone = false;

};

std::pair<vsg::dmat4, vsg::dmat4> VVCORE_EXPORT computeViewProjFixedScreen(const vsg::dmat4 &viewerMat, vsg::dvec3 eye, const vsg::dvec3 &xyz, const vsg::dvec3 &hpr, const vsg::vec2 &size, double near, double far, bool ortho=false, double worldAngle=0.f);
}
EVSG_type_name(vive::vvViewer);
EVSG_type_name(vive::OffAxis);
