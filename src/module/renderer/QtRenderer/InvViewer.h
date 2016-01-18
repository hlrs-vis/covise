/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_VIEWER_H
#define _INV_VIEWER_H

#include <util/coTypes.h>
#include <QMetaType>
#include <QListView>
#include <QListWidget>
#include <QEvent>

//
// inventor stuff
//
#include <Inventor/nodes/SoNode.h>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/SbPList.h>
#include <Inventor/SbBasic.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoCallback.h>
#include <Inventor/actions/SoGLRenderAction.h> // transparency levels
#include <Inventor/actions/SoCallbackAction.h>

#include "InvTextManager.h"

// utility
#include <util/coIntMultiHash.h>
#include <util/coHashIter.h>

//#include "ShowL.h"
#include <assert.h>
#include <vector>

//
// inventor component classes
//
class InvObject;
class TPHandler;
class InvObjectList;
//class InvPartEditor;
class InvClipPlaneEditor;
class InvManipList;
class InvPlaneMover;

class SoNodeSensor;
class SoBoxHighlightRenderAction;
class SoEnvironment;
class SoGroup;
class SoLabel;
class SoSwitch;
class SoLight;
class SoTransform;
class SoFont;
class SoClipPlane;
class SoJackDragger;

#ifdef HAVE_EDITORS
class SoQtMaterialEditor;
#endif

class SoQtColorEditor;
class SoMaterial;
class SbColor;
class SoGuiColorEditor;

class Q3ListBoxItem;

class CCoviseWindowCapture;

//////////////////////////////////////////////////////////////////////////////
//
//  Class: InvViewer
//
//////////////////////////////////////////////////////////////////////////////

class InvViewer : public SoQtExaminerViewer
{

    //
    //---------------------- PUBLIC STUFF ----------------------------------
    //
public:
    // Constructor:

    InvViewer(QWidget *parent = 0, const char *name = 0);
    ~InvViewer();

    virtual void render();

    virtual void rightWheelMotion(float value);
    virtual void rightWheelFinish();
    void enableRightWheelSampleControl(bool value);
    float getVolumeSamplingAccuracy();

    int getNumGlobalLutEntries();
    const uchar *getGlobalLut();
    void setGlobalLut(int numEntries, const uchar *entries);
    bool isGlobalLutUpdated();

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

    InvObjectList *list;

    InvPlaneMover *pm_;
    int handleState_;

    int viewer_edit_state;

    void objectListCB(InvObject *item, bool Selected);
    void onSequencerValueChanged(int val);

    void deleteScene();
    void addToSceneGraph(SoGroup *child, const char *name, SoGroup *root);
    void removeFromSceneGraph(SoGroup *root, const char *name);
    void replaceSceneGraph(SoNode *root);
    void addToTextureList(SoTexture2 *tex);
    void removeFromTextureList(SoTexture2 *tex);
    void findObjectName(char *objName, const SoPath *selectionPath);
    void updateObjectView();

    void setTransformation(float pos[3], float ori[4], int view,
                           float aspect, float near, float far,
                           float focal, float angle);

    void getTransformation(float pos[3], float ori[4], int *view,
                           float *aspect, float *near, float *far,
                           float *focal, float *angle);

    int getInteractiveCount() const;

    //
    // Before new data is sent to the viewer, the newData method should
    // be called to disconnect all manipulators and highlights
    //
    void newData();

    //
    // Editors
    //

    SbBool ignoreCallback;
    void removeEditors();
    void showEditors();
    SoMaterial *findMaterialForAttach(const SoPath *target);
    SoPath *findTransformForAttach(const SoPath *target);
    // callback used by Accum state action created by findMaterialForAttach
    static SoCallbackAction::Response findMtlPreTailCB(void *data,
                                                       SoCallbackAction *accum,
                                                       const SoNode *);

    // Background color
    SoQtColorEditor *backgroundColorEditor;
    void editBackgroundColor();
    static void backgroundColorCallback(void *userData,
                                        const SbColor *color);

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
    // Selection stuff
    //
    void setSelection(const char *name);
    void setDeselection(const char *name);

    // axis stuff
    //
    SoNode *makeAxis();
    SoSwitch *axis_switch; // default is whichChild 0 means ON
    int axis_state;
    void setAxis(int);

    // Scene graph data
    //
    SoGroup *drawStyle; // place for the switch node of the drawstyle from the internal scene graph

    // of the viewer class InvExaminerViewer
    SoSelection *selection; // the graph under which each incoming node

    // these should always be children of selection, even after deleting the scene
    SoGroup *permanentSelection;

    // will be placed
    SoSeparator *sceneGraph; // COVISE et al supplied scene graph
    SoSeparator *sceneColor; // colormap scene graph
    SoSeparator *sceneRoot; // the root of the two above

    TPHandler *tpHandler;

    // Selection highlight
    SoBoxHighlightRenderAction *highlightRA;

// Material editor
#ifdef HAVE_EDITORS
    SoQtMaterialEditor *materialEditor;
    SoQtColorEditor *colorEditor;
    void createMaterialEditor();
    void createColorEditor();
#endif

    // get state of the handle for visual interaction with covise (move CuttingSurfaces)
    int toggleHandleState();

    // setting clipping plane equation
    void setClipPlaneEquation(SbVec3f &point, SbVec3f &normal);

    // Clipping Plane
    //
    SoSwitch *clipSwitch;
    SoJackDragger *clipDragger;
    SoClipPlane *clipPlane;
    int clipState; // default is OFF
    void toggleClippingPlaneEditor(); // hides/shows clipping plane editor

    InvClipPlaneEditor *clippingPlaneEditor;
    void sendClippingPlane(int onoroff, SbVec3f &normal, SbVec3f &point);
    void setClippingPlane(SbVec3f &normal, SbVec3f &point);
    void setClipping(int onoroff);

    // set/get methods for billboard rendering
    int getBillboardRenderingMethod();
    void setBillboardRenderingMethod(int iMethod);

    void setBillboardRenderingBlending(bool bBlending);
    bool getBillboardRenderingBlending();

    SoCallback *snapshotCallback;
    static void snapshotCB(void *, SoAction *);

    // render window capturing methods
    static void setRenderWindowCaptureSize(int width, int height);

    static void enableRenderWindowCapture(bool bOn);
    static bool isEnabledRenderWindowCapture();

    static void writeRenderWindowSnapshot();

    static void enableFramePerSecondOutputConsole(bool bOn);
    static bool isEnabledFramePerSecondOutputConsole();

    InvTextManager *getTextManager() const
    {
        return text_manager;
    }


private:
    SoSwitch *m_current_tswitch;

    //
    // Billboarding
    //
    int m_iBillboardRenderingMethod;
    bool m_bBillboardRenderingBlending;

    //
    // Statistics
    //
    static bool m_bFramePerSecondOutputConsole;

    //
    // Window capturing
    //
    static CCoviseWindowCapture *m_pCapture;
    static bool m_bRenderWindowCapture;
    static int m_iCaptureWidth;
    static int m_iCaptureHeight;

    //
    // GUI interactions
    //
    bool rightWheelControlsVolumeSampling;
    float volumeSamplingAccuracy;
    float savedDollyValue;
    uchar *globalLut;
    int numGlobalLutEntries;
    bool globalLutUpdated;

    //
    // Selection
    //
    // manages changes in the selection.
    int selectionCallbackInactive;
    static void deselectionCallback(void *userData, SoPath *obj);
    static void selectionCallback(void *userData, SoPath *obj);
    static SoPath *pickFilterCB(void *userData, const SoPickedPoint *pick);
    SoNode *findShapeNode(const char *Name);
    void findPlaneMover(SoPath *);

    static int isSelected;
    static int c_first_time;

    SoGuiColorEditor *inscene;
    SoMaterial *editMaterial;

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
    void updateObjectList(SoPath *selectionPath, SbBool isSelection);

    //
    // camera stuff
    //
    SoNodeSensor *cameraSensor;
    static void cameraCallback(void *data, SoSensor *);

    float c_position[3];
    float c_orientation[4];
    int c_viewport;
    float c_aspectRatio;
    float c_nearDistance;
    float c_farDistance;
    float c_focalDistance;
    float c_heightAngle;

    //
    // Telepointer stuff
    //

    void projectTP(int mousex, int mousey, SbVec3f &intersection, float &aspectRatio);
    void sendTelePointer(QString &tpname, int state,
                         float px, float py, float pz, float aspectRatio);
    bool tpShow_;
    int mouseX, mouseY, keyState;

    static void viewerStartEditCB(void *data, SoQtViewer *);
    static void viewerFinishEditCB(void *data, SoQtViewer *);


    InvTextManager *text_manager;


    //
    // Convenience routines
    //
    static SbBool isAffectedByTransform(SoNode *node);
    static SbBool isAffectedByMaterial(SoNode *node);

    static void colorEditCB(void *, const SbColor *);

protected:
    // for getting all X events before the renderer handles it
    //
    virtual void processEvent(QEvent *);
    //virtual SbBool processSoEvent( const SoEvent * );
};

extern InvViewer *coviseViewer;

#endif
