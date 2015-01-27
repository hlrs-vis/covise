/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_VIEWER_
#define _INV_VIEWER_

//x**************************************************************************
//
//x * Description    : Inventor viewer base class
//
// * Class(es)      : InvViewer
//
// * inherited from : none
//
// * Author  : Dirk Rantzau
//
// * History : 29.03.94 V 1.0
//
//x**************************************************************************

#include <Inventor/SoType.h>
#include <Inventor/Xt/SoXtRenderArea.h>
#include <Inventor/misc/SoCallbackList.h>
#include <Inventor/SbTime.h>
#include <Inventor/nodes/SoTexture2.h>

// collaborative enhancements
extern void rm_sendDrawstyle(char *message);
extern void rm_sendViewing(char *message);
extern void rm_sendHeadlight(char *message);
extern int rm_isMaster();
extern int rm_isSynced();

// classes
class SoFieldSensor;
class SoNode;
class SoDirectionalLight;
class SoGroup;
class SoRotation;
class SoCamera;
class SoDrawStyle;
class SoLightModel;
class SoTimerSensor;
class SoFieldSensor;
class SoXtClipboard;
class InvViewer;
class SoGetBoundingBoxAction;
class SbPList;
class SoSeparator;
class SoSwitch;
class SoComplexity;
class SoPackedColor;
class SoMaterialBinding;
class SoSFTime;
class SoXtInputFocus;

// callback function prototypes
typedef void InvViewerCB(void *userData, InvViewer *viewer);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: InvViewer
//
//      The Viewer component is the abstract base class for all viewers.
//  It is subclassed from renderArea, adding viewing semantics to Inventor
//  rendering.
//
//////////////////////////////////////////////////////////////////////////////

// C-api: abstract
// C-api: prefix=SoXtVwr
class InvViewer : public SoXtRenderArea
{
public:
    //
    // An EDITOR viewer will create a camera under the user supplied scene
    // graph (specified in setSceneGraph()) if it cannot find one in the
    // scene and will leave the camera behind when supplied with a new scene.
    //
    // A BROWSER viewer will also create a camera if it cannot find one in
    // the scene, but will place it above the scene graph node (camera will
    // not appear in the user supplied scene graph), and will automatically
    // remove it when another scene is supplied to the viewer.
    //
    enum Type
    {
        BROWSER, // default viewer type
        EDITOR
    };

    //
    // list of possible drawing styles
    //
    // Note: Refer to the InvViewer man pages for a complete
    // description of those draw styles.
    //
    enum DrawStyle
    {
        VIEW_AS_IS, // unchanged
        VIEW_HIDDEN_LINE, // render only the front most lines
        VIEW_NO_TEXTURE, // render withought textures
        VIEW_LOW_COMPLEXITY, // render low complexity
        VIEW_LINE, // wireframe draw style
        VIEW_POINT, // point draw style
        VIEW_BBOX, // bounding box draw style
        VIEW_LOW_RES_LINE, // low complexity wireframe + no depth clearing
        VIEW_LOW_RES_POINT, // low complexity point + no depth clearing
        VIEW_SAME_AS_STILL, // forces the INTERACTIVE draw style to match STILL
        VIEW_MESH, // combined shading-wireframe representation
        VIEW_LOW_VOLUME // low resolution volume
    };
    enum DrawType
    {
        STILL, // default to VIEW_NO_TEXTURE (or VIEW_AS_IS)
        INTERACTIVE // default to VIEW_SAME_AS_STILL
    };

    //
    // list of different buffering types
    //
    enum BufferType
    {
        BUFFER_SINGLE,
        BUFFER_DOUBLE,
        BUFFER_INTERACTIVE // changes to double only when interactive
    };

    //
    // Sets/gets the scene graph to render. Whenever a new scene is supplied
    // the first camera encountered will be by default used as the edited
    // camera, else a new camera will be created.
    //
    virtual void setSceneGraph(SoNode *newScene);
    virtual SoNode *getSceneGraph();

    //
    // Set and get the edited camera. setCamera() is only needed if the
    // first camera found when setSceneGraph() is called isn't the one
    // the user really wants to edit.
    //
    // C-api: expose
    // C-api: name=setCam
    virtual void setCamera(SoCamera *cam);
    // C-api: name=getCam
    SoCamera *getCamera()
    {
        return camera;
    }

    //
    // Set and get the camera type that will be created by the viewer if no
    // cameras are found in the scene graph. Possible choices are :
    //	    - SoPerspectiveCamera::getClassTypeId()
    //	    - SoOrthographicCamera::getClassTypeId()
    //
    // NOTE: the set method will only take affect next time a scene graph
    // is specified (and if no camera are found).
    //
    // By default a perspective camera will be created if needed.
    //
    // C-api: expose
    // C-api: name=setCamType
    virtual void setCameraType(SoType type);
    // C-api: name=getCamType
    SoType getCameraType()
    {
        return cameraType;
    }

    //
    // Camera routines.
    //
    // C-api: expose
    virtual void viewAll();
    // C-api: expose
    // C-api: name=saveHomePos
    virtual void saveHomePosition();
    // C-api: expose
    // C-api: name=resetToHomePos
    virtual void resetToHomePosition();

    //
    // Turns the headlight on/off. (default ON)
    //
    // C-api: expose
    virtual void setHeadlight(SbBool onOrOff);
    SbBool isHeadlight()
    {
        return headlightFlag;
    }
    SoDirectionalLight *getHeadlight()
    {
        return headlightNode;
    }

    //
    // Sets/gets the current drawing style in the main view - The user
    // can specify the INTERACTIVE draw style (draw style used when
    // the scene changes) independently from the STILL style.
    //
    // STILL defaults to VIEW_AS_IS.
    // INTERACTIVE defaults to VIEW_NO_TEXTURE on machine that do not support
    // fast texturing, VIEW_SAME_AS_STILL otherwise.
    //
    // Refer to the InvViewer man pages for a complete description
    // of those draw styles.
    //
    // C-api: expose
    // C-api: name=setDStyle
    virtual void setDrawStyle(InvViewer::DrawType type,
                              InvViewer::DrawStyle style);
    // C-api: name=getDStyle
    InvViewer::DrawStyle getDrawStyle(InvViewer::DrawType type);

    // get the top draw-style switch node
    SoSwitch *getDrawStyleSwitch();

    // set the texture list
    void setTextureList(SoNodeList *texList);
    // set the lightmodel state
    void setLightModelState(int state);

    //
    // Sets/gets the current buffering type in the main view.
    // (default BUFFER_DOUBLE)
    //
    // C-api: expose
    // C-api: name=setBufType
    virtual void setBufferingType(InvViewer::BufferType type);
    // C-api: name=getBufType
    InvViewer::BufferType getBufferingType()
    {
        return bufferType;
    }

    // redefine this to call the viewer setBufferingType() method instead.
    virtual void setDoubleBuffer(SbBool onOrOff);

    //
    // Set/get whether the viewer is turned on or off. When turned off
    // events over the renderArea are sent down the sceneGraph
    // (picking can occurs). (default viewing is ON)
    //
    // C-api: expose
    virtual void setViewing(SbBool onOrOff);
    SbBool isViewing() const
    {
        return viewingFlag;
    };

    //
    // Set/get whether the viewer is allowed to change the cursor over
    // the renderArea window. When disabled, the cursor is undefined
    // by the viewer and will not change as the mode of the viewer changes.
    // When re-enabled, the viewer will reset it to the appropriate icon.
    //
    // Disabling the cursor enables the application to set the cursor
    // directly on the viewer window or on any parent widget of the viewer.
    // This can be used when setting a busy cursor on the application shell.
    //
    // Subclasses should redefine this routine to call XUndefineCursor()
    // or XDefineCursor() with the appropariate glyth. The base class
    // routine only sets the flag.
    //
    // C-api: expose
    virtual void setCursorEnabled(SbBool onOrOff);
    SbBool isCursorEnabled() const
    {
        return cursorEnabledFlag;
    }

    //
    // Set and get the auto clipping plane. When auto clipping is ON, the
    // camera near and far planes are dynamically adjusted to be as tight
    // as possible (least amount of stuff is clipped). When OFF, the user
    // is expected to manually set those planes within the preference sheet.
    // (default is ON).
    //
    // C-api: name=setAutoClip
    void setAutoClipping(SbBool onOrOff);
    // C-api: name=isAutoClip
    SbBool isAutoClipping() const
    {
        return autoClipFlag;
    }

    //
    // Turns stereo viewing on/off on the viewer (default off). When in
    // stereo mode, which may not work on all machines, the scene is rendered
    // twice (in the left and right buffers) with an offset between the
    // two views to simulate stereo viewing. Stereo classes have to be used
    // to see the affect and /usr/gfx/setmon needs to be called to set the
    // monitor in stereo mode.
    //
    // The user can also specify what the offset between the two views
    // should be.
    //
    virtual void setStereoViewing(SbBool onOrOff);
    virtual void setStippleStereoViewing(SbBool onOrOff);
    virtual SbBool isStereoViewing();
    virtual SbBool isStrippleStereoViewing()
    {
        return (strippleOn);
    };
    void setStereoOffset(float dist)
    {
        stereoOffset = dist;
    }
    float getStereoOffset()
    {
        return stereoOffset;
    }

    //
    // Seek methods
    //
    // Routine to determine whether or not to orient camera on
    // picked point (detail on) or center of the object's bounding box
    // (detail off). Default is detail on.
    // C-api: name=setDtlSeek
    void setDetailSeek(SbBool onOrOff)
    {
        detailSeekFlag = onOrOff;
    };
    // C-api: name=isDtlSeek
    SbBool isDetailSeek()
    {
        return detailSeekFlag;
    };

    // Set the time a seek takes to change to the new camera location.
    // A value of zero will not animate seek. Default value is 2 seconds.
    void setSeekTime(float seconds)
    {
        seekAnimTime = seconds;
    }
    float getSeekTime()
    {
        return seekAnimTime;
    }

    //
    // add/remove start and finish callback routines on the viewer. Start callbacks will
    // be called whenever the user starts doing interactive viewing (ex: mouse
    // down), and finish callbacks are called when user is done doing
    // interactive work (ex: mouse up).
    //
    // Note: The viewer pointer 'this' is passed as callback data
    //
    // C-api: name=addStartCB
    void addStartCallback(InvViewerCB *f, void *userData = NULL);
    //{ startCBList->addCallback((SoCallbackListCB *)f, userData); };
    // C-api: name=addFinishCB
    void addFinishCallback(InvViewerCB *f, void *userData = NULL)
    {
        finishCBList->addCallback((SoCallbackListCB *)f, userData);
    };
    // C-api: name=removeStartCB
    void removeStartCallback(InvViewerCB *f, void *userData = NULL)
    {
        startCBList->removeCallback((SoCallbackListCB *)f, userData);
    };
    // C-api: name=removeFinishCB
    void removeFinishCallback(InvViewerCB *f, void *userData = NULL)
    {
        finishCBList->removeCallback((SoCallbackListCB *)f, userData);
    };

    // copy/paste the view. eventTime should be the time of the X event
    // which initiated the copy or paste (e.g. if copy/paste is initiated
    // from a keystroke, eventTime should be the time in the X keyboard event.)
    void copyView(Time eventTime);
    void pasteView(Time eventTime);

    // redefine this routine to also correctly set the buffering type
    // on the viewer.
    virtual void setNormalVisual(XVisualInfo *);

    // This can be used to let the viewer know that the scene graph
    // has changed so that the viewer can recompute things like speed which
    // depend on the scene graph size.
    //
    // NOTE: This routine is automatically called whenever setSceneGraph()
    // is called.
    // C-api: expose
    // C-api: name=recompSceneSiz
    virtual void recomputeSceneSize();
    //
    // This routine will toggle the current camera from perspective to
    // orthographic, and from orthographic back to perspective.
    //
    virtual void toggleCameraType();

    int getInteractiveCount()
    {
        return interactiveCount;
    }

protected:
    // Constructor/Destructor
    InvViewer(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        InvViewer::Type type,
        SbBool buildNow);
    ~InvViewer();

    // global vars
    InvViewer::Type type;
    SoCamera *camera; // camera being edited
    SbBool viewingFlag; // FALSE when the viewer is off
    SbBool altSwitchBack; // flag to return to PICK after an Alt release
    SbBool cursorEnabledFlag;
    static SoSFTime *viewerRealTime; // pointer to "realTime" global field
    float sceneSize; // the larger of the X,Y,Z scene BBox
    float viewerSpeed; // default to 1.0 - MyFullViewer add UI to inc/dec

    // local tree variables
    SoSeparator *sceneRoot; // root node given to the RA
    SoNode *sceneGraph; // user supplied scene graph

    // Subclasses can call this routine to handle a common set of events. A Boolean
    // is returned to specify whether the event was handled by the base class.
    // Currently handled events and functions are :
    //	    'Esc' key - toggles viewing on/off
    //	    When viewing OFF - send all events down the scene graph
    //	    When camera == NULL - Discard all viewing events
    //	    'home' Key - calls resetToHomePosition()
    //	    's' Key - toggles seek on/off
    //	    Arrow Keys - moves the camera up/down/right/left in the viewing plane
    SbBool processCommonEvents(XAnyEvent *xe);

    // Invokes the start and finish viewing callbacks. Subclasses NEED to call
    // those routines when they start and finish doing interactive viewing
    // operations so that correct interactive drawing style and buffering
    // types, as well as application callbacks, gets set and called properly.
    //
    // Those routines simply increment and decrement a counter. When the counter
    // changes from 0->1 the start viewing callbacks are called. When the counter
    // changes back from 1->0 the finish viewing callbacks are called.
    // The counter approach enables different parts of a viewer to call those
    // routines withough having to know about each others (which is not
    //
    void interactiveCountInc();
    void interactiveCountDec();

    //
    // This routine is used by subclasses to initiate the seek animation. Given a
    // screen mouse location, this routine will return the picked point
    // and the normal at that point. It will also schedule the sensor to animate
    // if necessary. The routine retuns TRUE if something got picked...
    //
    // Note: if detailSeek is on, the point and normal correspond to the exact
    //	    3D location under the cursor.
    //	    if detailSeek if off, the object bbox center and the camera
    //	    orientation are instead returned.
    SbBool seekToPoint(const SbVec2s &mouseLocation);

    //
    // Subclasses CAN redefine this to interpolate camera position/orientation
    // while the seek animation is going on (called by animation sensor).
    // The parameter t is a [0,1] value corresponding to the animation percentage
    // completion. (i.e. a value of 0.25 means that animation is only 1/4 of the way
    // through).
    //
    virtual void interpolateSeekAnimation(float t);
    virtual void computeSeekFinalOrientation();

    // variables used for interpolating seek animations
    float seekDistance;
    SbBool seekDistAsPercentage; // percentage/absolute flag
    SbBool computeSeekVariables;
    SbVec3f seekPoint, seekNormal;
    SbRotation oldCamOrientation, newCamOrientation;
    SbVec3f oldCamPosition, newCamPosition;

    // Externally set the viewer into/out off seek mode (default OFF). Actual
    // seeking will not happen until the viewer decides to (ex: mouse click).
    //
    // Note: setting the viewer out of seek mode while the camera is being
    // animated will stop the animation to the current location.
    virtual void setSeekMode(SbBool onOrOff);
    SbBool isSeekMode()
    {
        return seekModeFlag;
    }

    // redefine this routine to adjust the camera clipping planes just
    // before doing a redraw. The sensor will be unschedule after the camera
    // is changed in the base class to prevent a second redraw from occuring.
    virtual void actualRedraw();

    // This is called during a paste.
    // Subclasses may wish to redefine this in a way that
    // keeps their viewing paradigm intact.
    virtual void changeCameraValues(SoCamera *newCamera);

    //
    // Convenience routines which subclasses can use when drawing viewer
    // feedback which may be common across multiple viewers. There is for
    // example a convenience routine which sets an orthographics projection
    // and a method to draw common feedback like the roll anchor (used by
    // a couple of viewers).
    //
    // All drawing routines assume that the window and projection is
    // already set by the caller.
    //
    // set an ortho projection of the glx window size - this also turns
    // zbuffering off and lighting off (if necessary).
    static void setFeedbackOrthoProjection(const SbVec2s &glxSize);
    // restores the zbuffer and lighting state that was changed when
    // setFeedbackOrthoProjection() was last called.
    static void restoreGLStateAfterFeedback();
    // draws a simple 2 colored cross at given position
    static void drawViewerCrossFeedback(SbVec2s loc);
    // draws the anchor feedback given center of rotation and mouse location
    static void drawViewerRollFeedback(SbVec2s center, SbVec2s loc);

    // redefine this for a better default draw style (need to wait for context)
    virtual void afterRealizeHook();

    // auto clipping vars and routines
    SbBool autoClipFlag;
    float minimumNearPlane; // minimum near plane as percentage of far
    SoGetBoundingBoxAction *autoClipBboxAction;
    virtual void adjustCameraClippingPlanes();

    //
    // Texture List (access texture maps depending on draw style and light model)
    //
    SoNodeList *textureList;
    int lightModelState;

private:
    SbBool strippleOn;
    // current state vars
    SoType cameraType;
    BufferType bufferType;
    SbBool interactiveFlag; // TRUE while doing interactive work
    float stereoOffset;
    SoXtInputFocus *inputFocus;

    // draw style vars
    DrawStyle stillDrawStyle, interactiveDrawStyle;
    SbBool checkForDrawStyle;
    SoSwitch *drawStyleSwitch; // on/off draw styles
    SoDrawStyle *drawStyleNode; // LINE vs POINT
    SoLightModel *lightModelNode; // BASE_COLOR vs PHONG
    SoPackedColor *colorNode; // hidden line first pass
    SoMaterialBinding *matBindingNode; // hidden line first pass
    SoComplexity *complexityNode; // low complexity & texture off
    void setCurrentDrawStyle(InvViewer::DrawStyle style);
    void doRendering();

    // collaborative enhancements
    void sendDrawstyle(int still, int dynamic);
    void sendViewing(int onoroff);
    void sendHeadlight(int onoroff);

    // copy and paste support
    SoXtClipboard *clipboard;
    static void pasteDoneCB(void *userData, SoPathList *pathList);

    // camera original values, used to restore the camera
    SbBool createdCamera;
    SbVec3f origPosition;
    SbRotation origOrientation;
    float origNearDistance;
    float origFarDistance;
    float origFocalDistance;
    float origHeight;
    float origHeightAngle_;
    SoSFEnum origViewportMapping_;

#ifdef __sgi
    // set to TRUE when we are using the SGI specific stereo extensions
    // which enables us to emulate OpenGL stereo on most machines.
    // We also save the camera original aspect ratio and viewport mapping
    // since we need to temporary strech the camera aspect ratio.
    SbBool useSGIStereoExt;
    float camStereoOrigAspect;
    int camStereoOrigVPMapping;
#endif

    // seek animation vars
    SbBool seekModeFlag; // TRUE when seek turned on externally
    SoFieldSensor *seekAnimationSensor;
    SbBool detailSeekFlag;
    float seekAnimTime;
    SbTime seekStartTime;
    static void seekAnimationSensorCB(void *p, SoSensor *);

    // headlight variables
    SoDirectionalLight *headlightNode;
    SoGroup *headlightGroup;
    SoRotation *headlightRot;
    SbBool headlightFlag; // true when headlight in turned on

    // interactive viewing callbacks
    int interactiveCount;
    SoCallbackList *startCBList;
    SoCallbackList *finishCBList;
    static void drawStyleStartCallback(void *, InvViewer *v);
    static void drawStyleFinishCallback(void *, InvViewer *v);
    static void bufferStartCallback(void *, InvViewer *v);
    static void bufferFinishCallback(void *, InvViewer *v);

    // set the zbuffer on current window to correct state
    void setZbufferState();
    SbBool isZbufferOff();
    void arrowKeyPressed(KeySym key);
};
#endif /* _INV_VIEWER_ */
