/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_COVISE_VIEWER_H
#define _INV_COVISE_VIEWER_H

// **************************************************************************
//
// * Description    : Interctive renderer for COVISE
//
// * Class(es)      : InvCoviseViewer
//
// * inherited from : InvExaminerViewer InvFullViewer InvViewer
//
// * Author  : Dirk Rantzau
//
// * History : 29.03.94 V 1.0
//
// **************************************************************************

//
// ec stuff
//

//
// renderer stuff
//
#include "InvDefs.h"
#include "InvRenderer.h"
#include "InvObjectList.h"
#include "InvError.h"
#include "InvTelePointer.h"

//
// X stuff
//
#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <Xm/RowColumn.h>
#include <Xm/List.h>
#include <Xm/DrawnB.h>

//
// inventor stuff
//
#include <Inventor/nodes/SoNode.h>
#include <Inventor/SbPList.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoCallback.h>
#include <Inventor/actions/SoGLRenderAction.h> // transparency levels
#include <Inventor/actions/SoCallbackAction.h>

// Spaceball support now local
#include "SoXtMagellan.h"

#include "InvExaminerViewer.h"
#include "InvFullViewer.h"
#include "InvViewer.h"
#include "InvColormapManager.h"
#include "InvTextManager.h"

#include "InvTimePart.h"
// utility
#include <util/coIntMultiHash.h>
#include <util/coHashIter.h>
#include "InvTimePartMultiHash.h"

#include "InvPlaneMover.h"
#include "InvAnnotationEditor.h"

#include <string>
#include "ShowL.h"
#include <assert.h>

//
// inventor component classes
//

// interaction
class SoBoxHighlightRenderAction;
class SoSelection;
class SoPickedPoint;

class MyColorEditor;
class InvPartEditor;
class InvClipPlaneEditor;
class SoXtFileBrowser;
class SoXtMaterialEditor;
class SoXtPrintDialog;
class SoXtTransformSliderSet;
class SoXtClipboard;
class SoXtDirectionalLightEditor;

class SoNodeSensor;
class SoDirectionalLight;
class SoLightModel;
class SoEnvironment;
class SoGroup;
class SoLabel;
class SoMaterial;
class InvCoviseViewer;
class SoSwitch;
class SoLight;
class SoTransform;
class SoText2;
class SoFont;
class InvManipList;

//
// fields and structs
//

enum InvEManipMode
{
    SV_NONE, // None
    SV_TRACKBALL, // Trackball manip
    SV_HANDLEBOX, // Handlebox manip
    SV_JACK, // Jack manip
    SV_CENTERBALL, // Centerball manip
    SV_XFBOX, // TransformBox manip
    SV_TABBOX // TabBox manip
};

struct InvCoviseViewerData;
struct InvLightData;
class InvCoviseViewer;

//
// external interface routines for cooperative working between
// different renderers
//
//  + camera
//  + transformation
//  + color
//

extern void rm_sendVRMLCamera(char *message);
extern void rm_sendCamera(char *message);
extern void rm_sendTransformation(char *message);
extern void rm_sendVRMLTelePointer(char *message);
extern void rm_sendTelePointer(char *message);
extern void rm_sendDrawstyle(char *message);
extern void rm_sendLightMode(char *message);
extern void rm_sendSelection(char *message);
extern void rm_sendDeselection(char *message);
extern void rm_sendPart(char *message);
extern void rm_sendReferencePart(char *message);
extern void rm_sendResetScene();
extern void rm_sendTransparency(char *message);
extern void rm_sendSyncMode(char *message);
extern void rm_sendFog(char *message);
extern void rm_sendAntialiasing(char *message);
extern void rm_sendBackcolor(char *message);
extern void rm_sendAxis(char *message);
extern void rm_sendClippingPlane(char *message);
extern void rm_sendViewing(char *message);
extern void rm_sendProjection(char *message);
extern void rm_sendDecoration(char *message);
extern void rm_sendHeadlight(char *message);
extern void rm_sendColormap(char *message);
extern void rm_sendCSFeedback(char *key, char *message);
extern void rm_sendAnnotation(char *key, char *message);
void receiveCamera(char *message);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: InvCoviseViewer
//
//////////////////////////////////////////////////////////////////////////////

#define SYNC_LOOSE 0
#define SYNC_SYNC 1
#define SYNC_TIGHT 2

class InvSequencer;
class InvPlaneMover;

class InvCoviseViewer : public InvRenderer, public SoXtComponent
{

    //
    //---------------------- PUBLIC STUFF ----------------------------------
    //
public:
    // Constructor:

    InvCoviseViewer(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE);
    ~InvCoviseViewer();

    //
    // friends
    //
    friend void receiveCamera(char *message);

    void addSequencer(InvSequencer *);
    void removeSequencer(InvSequencer *);

    void sendCSFeedback(char *key, char *message)
    {
        rm_sendCSFeedback(key, message);
    };
    void sendAnnotationMsg(char *key, char *message)
    {
        rm_sendAnnotation(key, message);
    };

    //
    void updateSlaves();

    InvExaminerViewer *getCurViewer()
    {
        return currentViewer;
    };

    //
    // this stuff has to be defined in all renderer classes
    // derived from InvRenderer
    //

    void show();
    void hide();
    void setMaster();
    void setSlave();
    void setMasterSlave();
    int isMaster();
    int isSynced();
    void setSceneGraph(SoNode *root);
    void addToSceneGraph(SoGroup *child, const char *name, SoGroup *root);
    void removeFromSceneGraph(SoGroup *root, const char *name);
    void replaceSceneGraph(SoNode *root);
    void addToTextureList(SoTexture2 *tex);
    void removeFromTextureList(SoTexture2 *tex);
    void setTransformation(float pos[3], float ori[4], int view,
                           float aspect, float near, float far,
                           float focal, float angle);
    void receiveTransformation(char *message);
    void receiveTelePointer(char *message);
    void receiveDrawstyle(char *message);
    void receiveLightMode(char *message);
    void receiveSelection(char *message);
    void receiveDeselection(char *message);
    void receivePart(char *message);
    void receiveReferencePart(char *message);
    void receiveResetScene();
    void receiveTransparency(char *message);
    void receiveSyncMode(char *message);
    void receiveFog(char *message);
    void receiveAntialiasing(char *message);
    void receiveBackcolor(char *message);
    void receiveAxis(char *message);
    void receiveClippingPlane(char *message);
    void receiveViewing(char *message);
    void receiveProjection(char *message);
    void receiveHeadlight(char *message);
    void receiveDecoration(char *message);
    void receiveColormap(char *message);

    Widget getSeqParent();

    // this was a pure function in SoXtComponent so we have to define this
    // also in any derived class
    Widget buildWidget(Widget parent, const char *name);

    //
    // rendering time routines
    //
    void setRenderTime(float time);

    //
    // Camera operation routines
    //
    void viewAll()
    {
        if (currentViewer != NULL)
            currentViewer->viewAll();
    }
    void viewSelection();
    void saveHomePosition()
    {
        if (currentViewer != NULL)
            currentViewer->saveHomePosition();
    }
    void setCamera(SoCamera *cam)
    {
        if (currentViewer != NULL)
            currentViewer->setCamera(cam);
    }
    // NOTE: because the camera may be changed dynamically (switch between ortho
    // and perspective), the user shouldn't cache the camera.
    SoCamera *getCamera()
    {
        return currentViewer->getCamera();
    }

    void changeCamera(SoCamera *newCamera);

    //
    // Before new data is sent to the viewer, the newData method should
    // be called to disconnect all manipulators and highlights
    //
    void newData();

    // Show/hide the pulldown menu bar (default shown)
    void showMenu(SbBool onOrOff);
    SbBool isMenuShown()
    {
        return showMenuFlag;
    }

    // Show/hide the viewer component trims (default shown)
    void sendDecoration(int onoroff);
    void setDecoration(SbBool onOrOff)
    {
        if (currentViewer != NULL)
            currentViewer->setDecoration(onOrOff);
    }
    SbBool isDecoration()
    {
        return currentViewer != NULL ? currentViewer->isDecoration() : FALSE;
    }

    // Show/hide headlight (default on) and get to the headlight node.
    void sendHeadlight(int onoroff);
    void setHeadlight(SbBool onOrOff)
    {
        if (currentViewer != NULL)
            currentViewer->setHeadlight(onOrOff);
    }
    SbBool isHeadlight()
    {
        return currentViewer != NULL ? currentViewer->isHeadlight() : FALSE;
    }
    SoDirectionalLight *getHeadlight()
    {
        return currentViewer != NULL ? currentViewer->getHeadlight() : NULL;
    }

    //
    // Sets/gets the current drawing style in the main view - The user
    // can specify the INTERACTIVE draw style (draw style used when
    // the scene changes) independently from the STILL style.
    //
    // (default VIEW_AS_IS for both STILL and INTERACTIVE)
    //
    virtual void setDrawStyle(InvViewer::DrawType type,
                              InvViewer::DrawStyle style)
    {
        currentViewer->setDrawStyle(type, style);
    }
    InvViewer::DrawStyle getDrawStyle(InvViewer::DrawType type)
    {
        return currentViewer->getDrawStyle(type);
    }

    //
    // Sets/gets the current buffering type in the main view
    // (default BUFFER_INTERACTIVE on Indigo, BUFFER_DOUBLE otherwise)
    //
    void setBufferingType(InvViewer::BufferType type)
    {
        currentViewer->setBufferingType(type);
    }
    InvViewer::BufferType getBufferingType()
    {
        return currentViewer->getBufferingType();
    }

    // Turn viewing on/off (Default to on) in the viewers.
    void setViewing(SbBool onOrOff)
    {
        currentViewer->setViewing(onOrOff);
    }
    SbBool isViewing()
    {
        return currentViewer->isViewing();
    }

    // Set/get the level of transparency type
    void setTransparencyType(SoGLRenderAction::TransparencyType type)
    {
        currentViewer->setTransparencyType(type);
    }
    SoGLRenderAction::TransparencyType getTransparencyType()
    {
        return currentViewer->getTransparencyType();
    }

    // returns the current render area widget
    Widget getRenderAreaWidget()
    {
        return currentViewer->getRenderAreaWidget();
    }

    // set the EXPLORER user mode callback routine
    void setUserModeEventCallback(SoXtRenderAreaEventCB *fcn);

    // switching parts on/off
    void switchPart(int key, int tag);

    void snap(const char *filename); // snap to filename with current window size
    // snap to any file at any size
    void snap(const char *filename, int sx, int sy);

    // reference point
    void setReferencePoint(int partID);
    void transformScene(int part);
    void resetTransformedScene();

    // setting clipping plane equation
    void setClipPlaneEquation(double point[], double normal[]);

    // (un)manage colormap-form, obj-list etc
    void unmanageObjs();
    void manageObjs();

    // get state of the handle for visual interaction with covise (move CuttingSurfaces)
    int toggleHandleState();

    void createAnnotationEditor(const InvAnnoFlag *af);

    InvTextManager *getTextManager() const
    {
        return text_manager;
    }

    //
    //---------------------- PROTECTED STUFF ----------------------------------
    //
protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call InvCoviseViewer::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER InvCoviseViewer(Widget parent, const char *name);

    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    // Support for menus in the popup planes
    Widget popupWidget;

    // Stuff to do after the component has been realized (shown)
    virtual void afterRealizeHook();

    ShowL cmapSelected_;

    char c_oldname[255];
    static int c_first_time;

    //
    //---------------------- PRIVATE STUFF ----------------------------------
    //

private:
    // sequencer needed to snap all timesteps

    InvSequencer *mySequencer;

    InvPlaneMover *pm_;
    int handleState_; // ==0 : free motion; ==1 snap to Axis
    int vrml_syn_; // ==0 disable VRML msgs ; ==1  enable VRML msgs

    static int selected;

    void constructorCommon(Widget parent, const char *name);

    // COOPERATIVE WORKING STUFF :

    //
    // master/slave stuff
    //
    short master;
    void setMasterSlaveMenu(short type);

    //
    // camera stuff
    //
    SoNodeSensor *cameraSensor;
    static void cameraCallback(void *data, SoSensor *);
    void getTransformation(float pos[3], float ori[4], int *view,
                           float *aspect, float *near, float *far,
                           float *focal, float *angle);

    float c_position[3];
    float c_orientation[4];
    int c_viewport;
    float c_aspectRatio;
    float c_nearDistance;
    float c_farDistance;
    float c_focalDistance;
    float c_heightAngle;

    //
    // Object Transformation stuff
    //
    SoNodeSensor *transformSensor;
    static void transformCallback(void *data, SoSensor *);
    void sendTransformation(const char *name, SoTransform *transform);
    SoTransform *currTransformNode;
    SoPath *currTransformPath;

    SoNodeList *transformNode;
    SoPathList *transformPath;

    Widget topForm_; // contains the upper part of the win
    //
    // Colormap stuff
    //
    InvColormapManager *colormap_manager;
    void sendColormap(char *message);
    /// check text for extended ascii codes
    void checkAscii(char *tgt, const char *src);
    void addColormap(const char *name, const char *colormap);
    void addColormapMenuEntry(const char *name);
    void toggleColormapMenu(Widget theWidget, XtPointer, XtPointer);
    void deleteColormap(const char *name);
    void replaceColormap(const char *name, const char *colormap);
    void createColormapList(char *, XmStringTable, int, Widget);
    static void colormapListCB(Widget, XtPointer, XmListCallbackStruct *);
    void addToColormapList(const char *name);
    void removeFromColormapList(const char *name);
    Widget colormaplist;
    void selectColormapListItem(int onORoff, const char *name);
    void updateColormapListSensitivity(int master, int sync_mode);
    float cmap_x_0, cmap_y_0;
    float cmap_size;
    int cmapPosition;
    enum InvCmapMode
    {
        COLORMAP_BOTTOM_LEFT,
        COLORMAP_TOP_LEFT,
        COLORMAP_TOP_RIGHT,
        COLORMAP_BOTTOM_RIGHT
    };

    InvTextManager *text_manager;

    //
    // Texture List (access texture maps depending on draw style)
    //
    SoNodeList *textureList;

    //
    // Part Switching
    //

    // hash-tables for parts
    coIntMultiHash<std::string> multiHash; // object names
    coHashIter<int, std::string> iter;
    coIntMultiHash<SoSwitch *> switchHash; // corresponding switch nodes
    coHashIter<int, SoSwitch *> switchIter;

    void addPart(const char *name, int partId, SoSwitch *s);
    void replacePart(const char *name, int partId, SoSwitch *s);
    void deletePart(const char *name);

    void sendPart(int partId, int switchTag);

    //
    // Reference part during animation
    //
    TimePartMultiHash<std::string> nameHash; // object names
    coHashIter<TimePart, std::string> nameIter;
    TimePartMultiHash<SoSwitch *> referHash; // corresponding switch nodes
    coHashIter<TimePart, SoSwitch *> referIter;

    SbVec3f refPoint; // bounding box center of reference part

    void addTimePart(const char *name, int timeStep, int partId, SoSwitch *s);
    void replaceTimePart(const char *name, int timeStep, int partId, SoSwitch *s);
    void deleteTimePart(const char *name);

    void sendReferencePart(int refPartId);
    void sendResetScene();

    //
    // Telepointer stuff
    //
    TPHandler *tpHandler;

    void projectTP(InvExaminerViewer *currentViewer, int mousex, int mousey,
                   SbVec3f &intersection, float &aspectRatio);

    void sendTelePointer(InvExaminerViewer *viewer, char *tpname, int state,
                         float px, float py, float pz, float aspectRatio);

    // spacemouse stuff
    //
    SoXtMagellan *spacemouse;

    //
    // LightMode stuff
    //
    void sendLightMode(int type);
    void setLightMode(int type);

    //
    // Selection stuff
    //
    void sendSelection(char *name);
    void setSelection(char *name);
    void sendDeselection(char *name);
    void setDeselection(char *name);

    //
    // Transparency stuff
    //
    void sendTransparency(int level);
    void setTransparency(char *name);

    //
    // Drawstyle stuff
    //

    void sendDrawstyle(int still, int dynamic);
    static void viewerStartEditCB(void *data, InvViewer *viewer);
    static void viewerFinishEditCB(void *data, InvViewer *viewer);

    //
    // Fog stuff
    //
    void sendFog(int onoroff);

    //
    // Antialiasing stuff
    //
    void sendAntialiasing(int onoroff);

    //
    // X stuff
    //

    int viewer_edit_state;

    //
    // lightmodel stuff
    //
    SoLightModel *lightmodel; // 0=PHONG , 1=BASE_COLOR
    int lightmodel_state;

    //
    // sync stuff
    //
    int sync_flag;
    void sendSyncMode();
    void removeEditors();
    void updateSyncLabel();
    void setSyncMode(int flag);
    void updateObjectView();
    void showEditors();

    // axis stuff
    //
    SoNode *makeAxis();
    SoSwitch *axis_switch; // default is whichChild 0 means ON
    int axis_state;
    void sendAxis(int onoroff);
    void setAxis(int onoroff);

    // for getting all X events before the renderer handels it
    //
    static SbBool appEventHandler(void *userData, XAnyEvent *anyevent);

    // Scene graph data
    //
    SoGroup *drawStyle; // place for the switch node of the drawstyle from the internal scene graph
    // of the viewer class InvExaminerViewer
    SoSelection *selection; // the graph under which each incoming node
    // will be placed
    SoSeparator *sceneGraph; // COVISE et al supplied scene graph
    SoSeparator *sceneColor; // colormap scene graph
    SoSeparator *sceneRoot; // the root of the two above

    // Clipping Plane
    //
    SoSwitch *clipSwitch; // default is OFF
    int clipState;
    SoCallback *clipCallback; // callback node for clipping
    SoCallback *finishClipCallback; // callback node to finish clipping
    // clipping callback routine using OpenGL
    static void clippingCB(void *, SoAction *);
    static void finishClippingCB(void *, SoAction *);
    GLdouble eqn[4]; // plane equation (defines the clipping plane)
    void editClippingPlane(); // invokes clipping plane editor

    InvClipPlaneEditor *clippingPlaneEditor;
    void sendClippingPlane(int onoroff, double equation[]);
    void setClippingPlane(double equation[]);
    void setClipping(int onoroff);

    static void depthTestCB(void *userData, SoAction *action);

    // Lights, camera, environment
    //
    SoGroup *lightsCameraEnvironment;
    SoLabel *envLabel;
    SoCamera *camera;
    SoEnvironment *environment;
    SoGroup *lightGroup;

    void createLightsCameraEnvironment();

    // Selection highlight
    SoBoxHighlightRenderAction *highlightRA;

    // spaceball stuff
    SoXtMagellan *spaceball;

    // Widgets and menus
    //
    Widget mgrWidget; // our topmost form widget
    Widget topbarMenuWidget; // topbar menu widgetInvRenderer
    Widget pageinMenuWidget; // Covise specific menu
    Widget leftRow; // empty row, for sequencer
    Widget masterlabel; // master/slave mode
    Widget statelabel; // current state info
    Widget timelabel; // last time for rendering info
    Widget synclabel; // sync mode
    Widget objectlist; // motif selection list for object display
    void updateObjectListSensitivity(int master, int sync_flag);

    // label update routines
    //
    void updateMasterLabel(short master);
    void updateStateLabel(char *label);
    void updateTimeLabel(float time);

    SbBool showMenuFlag;
    InvCoviseViewerData *menuItems; // list of menu items data

    void buildAndLayoutCoviseMenu(InvExaminerViewer *vwr, Widget parent);
    void createList(char *title, XmStringTable items, int nitem, Widget parent);
    Widget createText(Widget parent, char *label, char *text);

    Widget buildWidget(Widget parent);
    void buildAndLayoutTopbarMenu(Widget parent);
    void buildAndLayoutViewer(InvExaminerViewer *vwr);

    static void winCloseCallback(void *userData, SoXtComponent *comp);

    // callback for all menu buttons to perform action
    //
    static void processTopbarEvent(Widget, InvCoviseViewerData *,
                                   XmAnyCallbackStruct *);

    // callback when a menu is about to be displayed
    //
    static void menuDisplay(Widget, InvCoviseViewerData *,
                            XtPointer);

    // this is called after objects are added/deleted or the selection changes
    //
    void updateCommandAvailability();

    // object list stuff
    //
    InvObjectList *list;
    void addToObjectList(const char *name);
    void removeFromObjectList(const char *name);
    void updateObjectList(SoPath *selectionPath, SbBool isSelection);
    void updateObjectListItem(int onORoff, char *name);
    static void objectListCB(Widget, XtPointer, XmListCallbackStruct *);
    void findObjectName(char *objName, const SoPath *selectionPath);
    SoSwitch *findSwitchNode(char *objName);
    SoPath *findShapeByName(char *shapeName);
    SoTransform *findTransformNode(char *Name);
    SoShape *findShapeNode(char *Name);

    // File:

    // file reading methods
    //
    int fileMode;
    char *fileName;
    SbBool useShowcaseBrowser;
    void getFileName();
    void doFileIO(const char *filename);
    SbBool readFile(const char *filename);
    SbBool readEnvFile(const char *filename);
    SbBool writeFile(const char *filename);
    SbBool writeEnvFile(const char *filename);
    SbBool sendVRMLCamera();
    void deleteScene();
    void save();
    void removeCameras(SoGroup *root);

    // motif vars for file reading/writting
    //
    Widget fileDialog;
    static void fileDialogCB(Widget, InvCoviseViewer *,
                             XmFileSelectionBoxCallbackStruct *);

    // showcase file browser vars
    //
    SoXtFileBrowser *browser;
    static void browserCB(void *userData, const char *filename);

    // printing vars
    //
    SoXtPrintDialog *printDialog;
    void print();
    static void beforePrintCallback(void *uData, SoXtPrintDialog *);
    static void afterPrintCallback(void *uData, SoXtPrintDialog *);
    SbBool feedbackShown;

    // Edit:
    //

    // Select parent, if there is one; select everything.
    void pickParent();
    void pickAll();

    // for copy and paste
    //
    SoXtClipboard *clipboard; // copy/paste 3d data
    void destroySelectedObjects();

    // Paste callback - invoked when paste data transfer is complete
    //
    void pasteDone(SoPathList *pathList);
    static void pasteDoneCB(void *userData, SoPathList *pathList);

    // Viewing:
    //
    InvExaminerViewer *currentViewer; // current viewer pt
    void sendViewing(int onoroff);

    // Environment: fog, antialiasing
    //
    SbBool fogFlag; // Fog on/off
    void setFog(SbBool onOrOff); // Turns fog on/off
    SbBool antialiasingFlag; // AA-ing on/off
    void setAntialiasing(SbBool onOrOff); // Turns AA-ing on/off

    // Background color
    MyColorEditor *backgroundColorEditor;
    const SbColor &getBackgroundColor()
    {
        return currentViewer->getBackgroundColor();
    }
    void editBackgroundColor(); // Invokes color editor on bkg
    static void backgroundColorCallback(void *userData,
                                        const SbColor *color);
    void sendBackcolor(float r, float g, float b);

    //
    // Editors
    //

    SbBool ignoreCallback;
    SoMaterial *findMaterialForAttach(const SoPath *target);
    SoPath *findTransformForAttach(const SoPath *target);
    // callback used by Accum state action created by findMaterialForAttach
    static SoCallbackAction::Response findMtlPreTailCB(void *data,
                                                       SoCallbackAction *accum,
                                                       const SoNode *);

    // transform slider set
    SoXtTransformSliderSet *transformSliderSet;
    void createTransformSliderSet();

    // Material editor
    SoXtMaterialEditor *materialEditor;
    void createMaterialEditor();

    // Color editor
    MyColorEditor *colorEditor;
    void createColorEditor();

    // Object editor for parts
    //InvObjectEditor   *objectEditor;
    void createObjectEditor();

    // Part editor
    InvPartEditor *partEditor_;
    void createPartEditor();

    InvAnnotationEditor *annoEditor_;

    //
    // Manips
    //
    InvEManipMode curManip;
    SbBool curManipReplaces;
    InvManipList *maniplist; // list of selection/manip/xfPath triplets.

    // replaces manips with the given type for all selected objects.
    void replaceAllManips(InvEManipMode manipType);

    // attaches a manipulator
    void attachManip(InvEManipMode manipType, SoPath *p);
    void attachManipToAll(InvEManipMode manipType);

    // detaches a manipulator
    void detachManip(SoPath *p);
    void detachManipFromAll();

    // Temporarily remove manips for writing, printing, copying, etc.
    // and restore later
    void removeManips();
    void restoreManips();

    // Callback to adjust size of scale tabs. Used only for SoTabBoxManip
    // Added to viewer as a finishCallback.
    static void adjustScaleTabSizeCB(void *, InvViewer *);

    //
    // Lighting
    //

    SbPList lightDataList;
    SoXtDirectionalLightEditor *headlightEditor;
    InvLightData *headlightData;
    void addLight(SoLight *light);
    InvLightData *addLightEntry(SoLight *light, SoSwitch *lightSwitch);
    void addLightMenuEntry(InvLightData *);
    void turnLightOnOff(InvLightData *data, SbBool flag);
    static void lightToggleCB(Widget, InvLightData *, void *);
    void editLight(InvLightData *data, SbBool flag);
    static void editLightToggleCB(Widget, InvLightData *, void *);
    static void editLightColorCB(Widget, InvLightData *, void *);
    void removeLight(InvLightData *);
    static void removeLightCB(Widget, InvLightData *, void *);
    static void lightSubmenuDisplay(Widget, InvLightData *, void *);
    void transferDirectionalLightLocation(InvLightData *data);

    // vars to make the light manips all the same size
    float lightManipSize;
    SbBool calculatedLightManipSize;

    // temporary remove/add the light manip geometry of the attached manips
    // (used for file writting and printing)
    void removeAttachedLightManipGeometry();
    void addAttachedLightManipGeometry();

    // Ambient lighting color
    MyColorEditor *ambientColorEditor;
    void editAmbientColor(); // Invokes color editor on amb
    static void ambientColorCallback(void *userData,
                                     const SbColor *color);

    //
    // Selection
    //
    // manages changes in the selection.
    int selectionCallbackInactive;
    static void deselectionCallback(void *userData, SoPath *obj);
    static void selectionCallback(void *userData, SoPath *obj);
    static SoPath *pickFilterCB(void *userData, const SoPickedPoint *pick);

    //
    // Convenience routines
    //
    static SbBool isAffectedByTransform(SoNode *node);
    static SbBool isAffectedByMaterial(SoNode *node);

    // user pick function
    //
    SoXtRenderAreaEventCB *userModeCB;
    void *userModedata;
    SbBool userModeFlag;
    bool tpShow_;
};

extern InvCoviseViewer *coviseViewer;
#endif // _INV_COVISE_VIEWER_H
