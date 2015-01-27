/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// * Description    : Inventor interactive renderer for the PAGEIN project
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

//#ifndef __hpux
#include <covise/covise.h>
#include <config/CoviseConfig.h>
#define MENUS_IN_POPUP
//#endif

#include <unistd.h> // for access()
#include <sys/param.h> // for MAXPATHLEN
//
// X stuff
//
#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>

//
// Motif stuff
//
#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/CascadeB.h>
#include <Xm/CascadeBG.h>
#include <Xm/FileSB.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/Label.h>
#include <Xm/FileSB.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/Text.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>

//
// Inventor stuff
//
#include <Inventor/SoDB.h>
#include <Inventor/SoNodeKitPath.h>
#include <Inventor/SoPickedPoint.h>
#include <Inventor/SoOffscreenRenderer.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtClipboard.h>
#include <Inventor/Xt/SoXtDirectionalLightEditor.h>
#include <Inventor/Xt/SoXtMaterialEditor.h>
#include <Inventor/Xt/SoXtPrintDialog.h>
#include <Inventor/Xt/SoXtResource.h>
#include <Inventor/Xt/SoXtTransformSliderSet.h>
#include <Inventor/Xt/viewers/SoXtWalkViewer.h>
#include <Inventor/actions/SoBoxHighlightRenderAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/details/SoNodeKitDetail.h>
#include <Inventor/draggers/SoDirectionalLightDragger.h>
#include <Inventor/draggers/SoTabBoxDragger.h>
#include <Inventor/manips/SoCenterballManip.h>
#include <Inventor/manips/SoDirectionalLightManip.h>
#include <Inventor/manips/SoHandleBoxManip.h>
#include <Inventor/manips/SoJackManip.h>
#include <Inventor/manips/SoPointLightManip.h>
#include <Inventor/manips/SoSpotLightManip.h>
#include <Inventor/manips/SoTabBoxManip.h>
#include <Inventor/manips/SoTrackballManip.h>
#include <Inventor/manips/SoTransformManip.h>
#include <Inventor/manips/SoTransformBoxManip.h>
#include <Inventor/nodekits/SoBaseKit.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoEnvironment.h>
#include <Inventor/nodes/SoLabel.h>
#include <Inventor/SoLists.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/nodes/SoLight.h>
#ifdef CO_hp1020
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoCone.h>
#endif
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPointLight.h>
#include <Inventor/nodes/SoSelection.h>
#include <Inventor/nodes/SoShape.h>
#include <Inventor/nodes/SoSpotLight.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/sensors/SoNodeSensor.h>

#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoGLRenderAction.h>

#include "SoXtLinuxMagellan.h"

#include <string>

#include "InvExaminerViewer.h"
#include "InvFullViewer.h"
#include "InvViewer.h"
#include "InvSequencer.h"

#include "InvCoviseViewer.h"
#include "InvManipList.h"
#include "InvColorEditor.h"
#include "InvPartEditor.h"
#include "InvClipPlaneEditor.h"

#include "InvPlaneMover.h"
#include "InvAnnotationManager.h"
#ifdef _AIRBUS
#include "InvViewpointManager.h"
#endif

#include "InvOffscreenRenderer.h"

#include "InvRenderManager.h"

#ifdef TIMING
#include "InvTimer.h"
#endif

#include <GL/gl.h>

InvCoviseViewer *coviseViewer = NULL;

#include <util/coStringTable.h>
extern coStringTable partNames;

//#include <invent.h>

//
// external host and proc_id
//
extern int proc_id;
extern char *host;
extern Widget MasterRequest;
extern char username[100];
#ifndef WIN32

// X11 snapshot function for Linux
#define uint32 uint32_tiff
#include "tiffio.h"
#undef uint32
void x11SnapTIFF(Widget wid, const char *filename);
#endif

//
// renderer menu stuff
//

enum MenuEntries
{

    SV_FILE = 0, // start at 0 since we use an array
    SV_FILE_SAVE,
    SV_FILE_SAVE_AS,
    SV_FILE_COPY,
    SV_FILE_PRINT,
    SV_FILE_SAVE_ENV,
    SV_FILE_READ_ENV,
    SV_FILE_SNAP,
    SV_FILE_SNAP_ALL,
    SV_FILE_RESIZE_PAL,

    SV_EDIT,
    SV_EDIT_PICK_PARENT,
    SV_EDIT_PICK_ALL,
    SV_EDIT_COPY,

    SV_VIEW,
    SV_VIEW_PICK,
    SV_VIEW_USER,
    SV_VIEW_SELECTION,
    SV_VIEW_FOG,
    SV_VIEW_ANTIALIASING,
    SV_VIEW_SCREEN_TRANSPARENCY,
    SV_VIEW_BLEND_TRANSPARENCY,
    SV_VIEW_DELAY_BLEND_TRANSPARENCY,
    SV_VIEW_SORT_BLEND_TRANSPARENCY,
    SV_VIEW_BKG_COLOR,
    SV_VIEW_AXIS,
    SV_VIEW_CLIPPING,

    SV_EDITOR,
    SV_EDITOR_MATERIAL,
    SV_EDITOR_COLOR,
    SV_EDITOR_TRANSFORM,
    SV_EDITOR_PARTS,
    SV_EDITOR_SNAPH,
    SV_EDITOR_FREEH,

    SV_MANIP,
    SV_MANIP_TRACKBALL,
    SV_MANIP_HANDLEBOX,
    SV_MANIP_JACK,
    SV_MANIP_CENTERBALL,
    SV_MANIP_XFBOX,
    SV_MANIP_TABBOX,
    SV_MANIP_NONE,
    SV_MANIP_REPLACE_ALL,

    SV_LIGHT,
    SV_LIGHT_MODEL,
    SV_LIGHT_ADD_DIRECT,
    SV_LIGHT_ADD_POINT,
    SV_LIGHT_ADD_SPOT,
    SV_LIGHT_AMBIENT_EDIT,
    SV_LIGHT_TURN_ON,
    SV_LIGHT_TURN_OFF,
    SV_LIGHT_SHOW_ALL,
    SV_LIGHT_HIDE_ALL,

    SV_COLORMAP,
    SV_COLORMAP_BOTTOM_LEFT,
    SV_COLORMAP_TOP_LEFT,
    SV_COLORMAP_TOP_RIGHT,
    SV_COLORMAP_BOTTOM_RIGHT,

    SV_SYNC,
    SV_SYNC_LOOSE,
    SV_SYNC_MA_SL,
    SV_SYNC_TIGHT,

    SV_HELP,
    SV_HELP_1,
    SV_HELP_2,

    SV_MENU_NUM // this must be the last entry
};

// different types of menu item buttons
enum MenuItems
{
    SV_SEPARATOR,
    SV_PUSH_BUTTON,
    SV_TOGGLE_BUTTON,
    SV_RADIO_BUTTON
};

struct InvCoviseViewerButton
{
    const char *name;
    int id;
    int buttonType; // PUSH, TOGGLE, RADIO
    const char *accelerator; // e.g. "Alt <Key> p" or "Ctrl <Key> u"
    const char *accelText; // text that appears in the menu item
};
struct InvCoviseViewerMenu
{
    const char *name;
    int id;
    struct InvCoviseViewerButton *subMenu;
    int subItemCount;
};

static InvCoviseViewerButton fileData[] = {
    { "Save", SV_FILE_SAVE, SV_PUSH_BUTTON, "Alt <Key> s", "Alt+s" },
    { "Save As...", SV_FILE_SAVE_AS, SV_PUSH_BUTTON, "Alt Shift <Key> s", "Alt+S" },
    { "Snap", SV_FILE_SNAP, SV_PUSH_BUTTON, "Ctrl <Key> s", "Ctrl+s" },
    { "Snap All", SV_FILE_SNAP_ALL, SV_PUSH_BUTTON, "Ctrl Shift <Key> s", "Ctrl+S" },
    { "Resize to PAL ratio", SV_FILE_RESIZE_PAL, SV_PUSH_BUTTON, 0, 0 },
    { "Copy View", SV_FILE_COPY, SV_PUSH_BUTTON, "Alt <Key> c", "Alt+S" },
    { "Print...", SV_FILE_PRINT, SV_PUSH_BUTTON, "Alt <Key> p", "Alt+p" },
    { 0, 0, SV_SEPARATOR, 0, 0 },
    { "Read Camera Env...", SV_FILE_READ_ENV, SV_PUSH_BUTTON, 0, 0 },
    { "Save Camera Env...", SV_FILE_SAVE_ENV, SV_PUSH_BUTTON, 0, 0 },
};

static InvCoviseViewerButton viewData[] = {
    { "Pick/Edit", SV_VIEW_PICK, SV_TOGGLE_BUTTON, 0, 0 },
    { "Light Edit Mode", SV_VIEW_USER, SV_TOGGLE_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "View Selection", SV_VIEW_SELECTION, SV_PUSH_BUTTON, 0, 0 },
    { "Fog", SV_VIEW_FOG, SV_TOGGLE_BUTTON, 0, 0 },
    { "Antialiasing", SV_VIEW_ANTIALIASING, SV_TOGGLE_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "Screen Door Transparency", SV_VIEW_SCREEN_TRANSPARENCY, SV_RADIO_BUTTON, 0, 0 },
    { "Blended Transparency", SV_VIEW_BLEND_TRANSPARENCY, SV_RADIO_BUTTON, 0, 0 },
    { "Delayed Blended Transparency", SV_VIEW_DELAY_BLEND_TRANSPARENCY, SV_RADIO_BUTTON, 0, 0 },
    { "Sorted Blended Transparency", SV_VIEW_SORT_BLEND_TRANSPARENCY, SV_RADIO_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "Edit Background Color...", SV_VIEW_BKG_COLOR, SV_PUSH_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "Hide Coordinate Axes", SV_VIEW_AXIS, SV_TOGGLE_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "Clipping Plane", SV_VIEW_CLIPPING, SV_TOGGLE_BUTTON, 0, 0 }
};

static InvCoviseViewerButton editorData[] = {
    { "Material Editor...", SV_EDITOR_MATERIAL, SV_PUSH_BUTTON, 0, 0 },
    { "Color Editor...", SV_EDITOR_COLOR, SV_PUSH_BUTTON, 0, 0 },
    { "Object Transform...", SV_EDITOR_TRANSFORM, SV_PUSH_BUTTON, 0, 0 },
    { "Parts ...", SV_EDITOR_PARTS, SV_PUSH_BUTTON, 0, 0 },
    { "Snap handle to axis", SV_EDITOR_SNAPH, SV_PUSH_BUTTON, 0, 0 },
    { "Free handle motion", SV_EDITOR_FREEH, SV_PUSH_BUTTON, 0, 0 }
};

static InvCoviseViewerButton manipData[] = {
    { "Trackball", SV_MANIP_TRACKBALL, SV_RADIO_BUTTON, 0, 0 },
    { "HandleBox", SV_MANIP_HANDLEBOX, SV_RADIO_BUTTON, 0, 0 },
    { "Jack", SV_MANIP_JACK, SV_RADIO_BUTTON, 0, 0 },
    { "Centerball", SV_MANIP_CENTERBALL, SV_RADIO_BUTTON, 0, 0 },
    { "TransformBox", SV_MANIP_XFBOX, SV_RADIO_BUTTON, 0, 0 },
    { "TabBox", SV_MANIP_TABBOX, SV_RADIO_BUTTON, 0, 0 },
    { "None", SV_MANIP_NONE, SV_RADIO_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "Replace", SV_MANIP_REPLACE_ALL, SV_TOGGLE_BUTTON, 0, 0 }
};

static InvCoviseViewerButton lightData[] = {
    { "Lightmodel BASE_COLOR", SV_LIGHT_MODEL, SV_RADIO_BUTTON, 0, 0 },
    { "Create Dir Light", SV_LIGHT_ADD_DIRECT, SV_PUSH_BUTTON, 0, 0 },
    { "Create Point Light", SV_LIGHT_ADD_POINT, SV_PUSH_BUTTON, 0, 0 },
    { "Create Spot Light", SV_LIGHT_ADD_SPOT, SV_PUSH_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "Ambient Lighting...", SV_LIGHT_AMBIENT_EDIT, SV_PUSH_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
    { "Turn all ON", SV_LIGHT_TURN_ON, SV_PUSH_BUTTON, 0, 0 },
    { "Turn all OFF", SV_LIGHT_TURN_OFF, SV_PUSH_BUTTON, 0, 0 },
    { "Show all Icons", SV_LIGHT_SHOW_ALL, SV_PUSH_BUTTON, 0, 0 },
    { "Hide all Icons", SV_LIGHT_HIDE_ALL, SV_PUSH_BUTTON, 0, 0 },
    { "", SV_SEPARATOR, 0, 0, 0 },
};

/*
static InvCoviseViewerButton colormapData[] = {
    {"Bottom Left",   SV_COLORMAP_BOTTOM_LEFT,   SV_RADIO_BUTTON, 0, 0},
    {"Top Left",      SV_COLORMAP_TOP_LEFT,      SV_RADIO_BUTTON, 0, 0},
    {"Top Right",     SV_COLORMAP_TOP_RIGHT,     SV_RADIO_BUTTON, 0, 0},
    {"Bottom Right",  SV_COLORMAP_BOTTOM_RIGHT,  SV_RADIO_BUTTON, 0, 0},
    {"",	      SV_SEPARATOR }
};
*/
static InvCoviseViewerButton syncData[] = {
    { "Loose Coupling", SV_SYNC_LOOSE, SV_TOGGLE_BUTTON, 0, 0 },
    { "Master/Slave", SV_SYNC_MA_SL, SV_TOGGLE_BUTTON, 0, 0 },
    { "Tight Coupling", SV_SYNC_TIGHT, SV_TOGGLE_BUTTON, 0, 0 }
};

static InvCoviseViewerButton helpData[] = {
    { "Online Help", SV_HELP_1, SV_PUSH_BUTTON, 0, 0 },
    { "Online User's Guide", SV_HELP_2, SV_PUSH_BUTTON, 0, 0 }
};

static InvCoviseViewerMenu pulldownData[] = {
    //  {name, 	id,	    	subMenu,    subItemCount}
    { "File", SV_FILE, fileData, XtNumber(fileData) },
    { "Viewing", SV_VIEW, viewData, XtNumber(viewData) },
    { "Editors", SV_EDITOR, editorData, XtNumber(editorData) },
    { "Manips", SV_MANIP, manipData, XtNumber(manipData) },
    { "Lights", SV_LIGHT, lightData, XtNumber(lightData) },
    //    {"Colormap",SV_COLORMAP, 	colormapData,XtNumber(colormapData)  },
    { "Sync", SV_SYNC, syncData, XtNumber(syncData) },
    { "Help", SV_HELP, helpData, XtNumber(helpData) }
};

//
//  Macros and constants
//

// toggle button macros
#define TOGGLE_ON(BUTTON) \
    XmToggleButtonSetState((Widget)BUTTON, TRUE, FALSE)
#define TOGGLE_OFF(BUTTON) \
    XmToggleButtonSetState((Widget)BUTTON, FALSE, FALSE)

#define FOG_FUDGE 1.6
#define SV_NUM_LIGHTS 6

#define SWITCH_LIGHT_OFF(SWITCH) (SWITCH)->whichChild.setValue(SO_SWITCH_NONE)
#define SWITCH_LIGHT_ON(SWITCH) (SWITCH)->whichChild.setValue(SO_SWITCH_ALL)
#define IS_LIGHT_ON(SWITCH) ((SWITCH)->whichChild.getValue() == SO_SWITCH_ALL)

#ifndef _AIRBUS
#define SV_ENV_LABEL "COVISE Renderer Environment v1.0"
#else
#define SV_ENV_LABEL "NS3D Renderer Environment v1.0"
#endif

// helper for writing rgb
void putbyte(FILE *outf, char val)
{
    unsigned char buf[1];

    buf[0] = val;
    if (fwrite(buf, 1, 1, outf) != 1)
    {
        cerr << "fwrite error in putbyte" << endl;
    }
}

void putshort(FILE *outf, unsigned short val)
{
    unsigned char buf[2];

    buf[0] = (val >> 8);
    buf[1] = (val >> 0);
    if (fwrite(buf, 2, 1, outf) != 1)
    {
        cerr << "fwrite error in putshort" << endl;
    }
}

/*static int putlong(FILE *outf, unsigned long val)
{
   unsigned char buf[4];

   buf[0] = (val>>24);
   buf[1] = (val>>16);
   buf[2] = (val>>8);
   buf[3] = (val>>0);
   return fwrite(buf,4,1,outf);
}*/

//
// Viewer Structs
//
struct InvCoviseViewerData
{
    int id;
    InvCoviseViewer *classPt;
    Widget widget;
};

class InvLightData
{
public:
    // Constructor inits everything to NULL
    InvLightData();

    InvCoviseViewer *classPt;
    SoSwitch *lightSwitch;
    SoTranslation *translation; // for placing a directional light manip
    SoScale *scale;
    SoLight *light;
    SoScale *scaleInverse;
    SoTranslation *translationInverse;
    SoType type;
    char *name;
    MyColorEditor *colorEditor;
    SbBool isManip;
    SbBool shouldBeManip; // Used to remember what it was when
    // they all get turned off for writing,
    // printing, etc.
    Widget cascadeWidget;
    Widget submenuWidget;
    Widget onOffWidget;
    Widget iconWidget;
    Widget editColorWidget;
    Widget removeWidget;
};

InvLightData::InvLightData()
{
    classPt = NULL;
    lightSwitch = NULL;
    translation = NULL;
    scale = NULL;
    light = NULL;
    scaleInverse = NULL;
    translationInverse = NULL;
    name = NULL;
    colorEditor = NULL;
    cascadeWidget = NULL;
    submenuWidget = NULL;
    onOffWidget = NULL;
    iconWidget = NULL;
    editColorWidget = NULL;
    removeWidget = NULL;
}

//
//
// axis data

#ifdef CO_hp1020

static char *axis = "\
#Inventor V2.0 ascii\n\
Separator { \
    PickStyle { style UNPICKABLE } \
    LightModel { model BASE_COLOR } \
    MaterialBinding { value PER_PART } \
    DrawStyle { lineWidth 2 } \
    Coordinate3 { point [0 0 0, 1 0 0, 0 1 0, 0 0 1] } \
    BaseColor { rgb [1 0 0, 0 1 0, 0 0 1] } \
    IndexedLineSet { coordIndex [1, 0, 2, -1, 0, 3] } \
     \
    LightModel { model PHONG } \
    MaterialBinding { value OVERALL } \
    Complexity { value .1 } \
    Separator { \
    	Material { \
	    diffuseColor    [ 0.5 0 0 ] \
	    emissiveColor   [ 0.5 0 0 ] \
	} \
	Translation { translation 1 0 0 } \
    	RotationXYZ { axis Z angle -1.570796327 } \
    	Cone { bottomRadius .2 height .3 } \
    } \
    Separator { \
    	Material { \
	    diffuseColor    [ 0 0.5 0 ] \
	    emissiveColor   [ 0 0.5 0 ] \
	} \
	Translation { translation 0 1 0 } \
    	Cone { bottomRadius .2 height .3 } \
    } \
    Material { \
	diffuseColor    [ 0 0 0.5 ] \
	emissiveColor   [ 0 0 0.5 ] \
    } \
    Translation { translation 0 0 1 } \
    RotationXYZ { axis X angle 1.570796327 } \
    Cone { bottomRadius .2 height .3 } \
} ";

/*
#elif __linux__
static char *axis =
"Separator {\n"
"    LightModel { model BASE_COLOR }\n"
"    MaterialBinding { value PER_FACE }\n"
"    DrawStyle { lineWidth 2 }\n"
"    Coordinate3 { point [0 0 0, 1 0 0, 0 1 0, 0 0 1] }\n"
"    BaseColor { rgb [1 0 0, 0 1 0, 0 0 1 ] }\n"
"    IndexedLineSet {\n"
"          coordIndex [0, 1, -1, 0, 2, -1, 0, 3] }\n"
"    Separator {\n"
"        BaseColor{ rgb 1 0 0  }\n"
"        Translation { translation 1 0 0 }\n"
"        RotationXYZ { axis Z angle -1.570796327 }\n"
"        Cone { bottomRadius .05 height .15 }\n"
"    }\n"
"    Separator {\n"
"        BaseColor{ rgb 0 1 0 }\n"
"        Translation { translation 0 1 0 }\n"
"        Cone { bottomRadius .05 height .15 }\n"
"    }\n"
"    Separator {\n"
"        BaseColor{ rgb 0 0 1}\n"
"        Translation { translation 0 0 1 }\n"
"        RotationXYZ { axis X angle 1.570796327 }\n"
"        Cone { bottomRadius .05 height .15 }\n"
"    }\n"
"}\n";

*/

#else
static const char *axis = "Separator {\n"
                          "    LightModel { model BASE_COLOR }\n"
                          "    MaterialBinding { value PER_FACE }\n"
                          "    DrawStyle { lineWidth 2 }\n"
                          "    Coordinate3 { point [0 0 0, 1 0 0, 0 1 0, 0 0 1] }\n"
                          "    BaseColor { rgb [1 0 0, 0 1 0, 0 0 1 ] }\n"
                          "    IndexedLineSet {\n"
                          "          coordIndex [0, 1, -1, 0, 2, -1, 0, 3] }\n"
                          "    Separator {\n"
                          "        BaseColor{ rgb 1 0 0  }\n"
                          "        Translation { translation 1 0 0 }\n"
                          "        RotationXYZ { axis Z angle -1.570796327 }\n"
                          "        Cone { bottomRadius .05 height .15 }\n"
                          "        Translation { translation 0 .15 0 }\n"
                          "        Text2 { string "
                          "X"
                          " }\n"
                          "    }\n"
                          "    Separator {\n"
                          "        BaseColor{ rgb 0 1 0 }\n"
                          "        Translation { translation 0 1 0 }\n"
                          "        Cone { bottomRadius .05 height .15 }\n"
                          "        Translation { translation 0 .15 0 }\n"
                          "        Text2 { string "
                          "Y"
                          " }\n"
                          "    }\n"
                          "    Separator {\n"
                          "        BaseColor{ rgb 0 0 1}\n"
                          "        Translation { translation 0 0 1 }\n"
                          "        RotationXYZ { axis X angle 1.570796327 }\n"
                          "        Cone { bottomRadius .05 height .15 }\n"
                          "        Translation { translation 0 .15 0 }\n"
                          "        Text2 { string "
                          "Z"
                          " }\n"
                          "    }\n"
                          "}\n";
#endif

int InvCoviseViewer::selected = 0;
int InvCoviseViewer::c_first_time = 0;

//======================================================================
//
// Description:
//	Constructor for the Renderer.
//      Creates the Topbar menu
//
// Use: public
//======================================================================
InvCoviseViewer::InvCoviseViewer(Widget parent,
                                 const char *name,
                                 SbBool /* Uwe Woessner (wegen Warnung) buildInsideParent */)
    : handleState_(0)
    , master(TRUE)
    , sync_flag(SYNC_SYNC)

{
    Annotations->initialize(this);
#ifdef _AIRBUS
    Viewpoints->initialize(this);
#endif
    constructorCommon(parent, name);
    coviseViewer = this;
    mySequencer = NULL;

    vrml_syn_ = coCoviseConfig::getInt("VrmlSrv.VrmlSyn", 0);
}

//======================================================================
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//======================================================================
InvCoviseViewer::InvCoviseViewer(Widget parent,
                                 const char *name)
    : master(TRUE)
    , sync_flag(SYNC_SYNC)
{
    Annotations->initialize(this);
#ifdef _AIRBUS
    Viewpoints->initialize(this);
#endif
    constructorCommon(parent, name);
    coviseViewer = this;
    mySequencer = NULL;
    Annotations->initialize(this);
#ifdef _AIRBUS
    Viewpoints->initialize(this);
#endif

    vrml_syn_ = coCoviseConfig::getInt("VrmlSrv.VrmlSyn", 0);
}

void InvCoviseViewer::addSequencer(InvSequencer *seq)
{
    Arg args[2];
    mySequencer = seq;

    XtSetArg(args[0], XmNsensitive, True);
    XtSetValues(menuItems[SV_FILE_SNAP_ALL].widget, args, 1);
}

void InvCoviseViewer::removeSequencer(InvSequencer *seq)
{
    Arg args[2];
    (void)seq;
    mySequencer = NULL;

    XtSetArg(args[0], XmNsensitive, false);
    XtSetValues(menuItems[SV_FILE_SNAP_ALL].widget, args, 1);
}

//======================================================================
//
// Called by the constructors
//
// private
//======================================================================
void
InvCoviseViewer::constructorCommon(Widget, const char *)
{
    /* ! old !
                                    sceneRoot
                                        |
                                -----------------
                                |               |
                            colormap        sceneGraph
                                                |
                  ----------------------------------------
                                      |             |           |            |
                               lightsAndCamera  lightModel   axisSwitch   selection
                                                                |            |
   axisStuff    allCoviseDataComesHere

   */

    /*  ! old !

                                            sceneGraph
                                                 |
                   --------------------------------------------------------------------
                                       |             |           |            |                           |
                                lightsAndCamera  lightModel   axisSwitch   selection                   colormap
                                                                 |            |                          |
                                                              axisStuff    allCoviseDataComesHere   ------------
                                                                                                    |          |
                                                                                                cmapcamera     |
   cmapStuff

   */
    /* present user supplied scene graph:

                          sceneGraph
                              |
                 -------------------------------------------------------------------------------------
                        |             |           |            |         |          |           |           |
                 lightsAndCamera  lightModel  axisSwitch   colormap  drawStyle  clipSwitch   selection  finishClipping
                                                  |            |                    |           |
                                              axisStuff   ------------          clipCallback allCoviseDataComesHere
                                                          |          |
                                                      cmapcamera     |
   cmapStuff

   */

    int i;

    setClassName("InvCoviseViewer");
    SoSelection *inputGraph = new SoSelection;
    // char *envFile = NULL;

    // selection is the users scene graph.
    selection = inputGraph;
    currentViewer = NULL;
    createLightsCameraEnvironment();

    // the scene viewer supplies its own camera and lights.
    // in fact, we remove any cameras that might be in the users graph.
    sceneGraph = new SoSeparator();
    // sceneColor = new SoSeparator();
    // sceneRoot  = new SoSeparator();
    // sceneRoot->ref();

    sceneGraph->ref();

    sceneGraph->addChild(lightsCameraEnvironment);

    //
    // Widget and menu variables
    //
    mgrWidget = NULL;
    showMenuFlag = TRUE;

    menuItems = new InvCoviseViewerData[SV_MENU_NUM];
    for (i = 0; i < SV_MENU_NUM; i++)
    {
        menuItems[i].id = i;
        menuItems[i].classPt = this;
        menuItems[i].widget = NULL;
    }
    popupWidget = NULL;

    // lightmodel
    //
    //
    lightmodel = new SoLightModel;
    lightmodel_state = 1;
    lightmodel->model = SoLightModel::PHONG;
    sceneGraph->addChild(lightmodel);

    pm_ = new InvPlaneMover(this);

    selection->addChild(Annotations->getGroup());
    sceneGraph->addChild(pm_->getSeparator());

    //
    // add axis
    int axStatus;
    if (coCoviseConfig::isOn("Renderer.Axis", true))
    {
        axStatus = CO_ON;
    }
    else
    {
        axStatus = CO_OFF;
    }

    axis_switch = new SoSwitch;
    axis_switch->whichChild.setValue(0);
    axis_switch->addChild(makeAxis());
    sceneGraph->addChild(axis_switch);
    setAxis(axStatus);
    sendAxis(axStatus);
    axis_state = axStatus;

    text_manager = new InvTextManager();
    sceneGraph->addChild(text_manager->getRootNode());

    //
    // Initialize colormap support
    //
    colormap_manager = new InvColormapManager();
    sceneGraph->addChild(colormap_manager->getRootNode());
    cmap_x_0 = -1.0;
    cmap_y_0 = -0.9;
    cmap_size = 0.7;
    cmapPosition = COLORMAP_BOTTOM_LEFT;

    drawStyle = new SoGroup;
    sceneGraph->addChild(drawStyle);

    //
    // Texture list
    //
    textureList = new SoNodeList();

    // clipping
    clipSwitch = new SoSwitch;
    clipSwitch->whichChild.setValue(SO_SWITCH_NONE);
    clipState = CO_OFF;
    clipCallback = new SoCallback;
    clipCallback->setCallback(InvCoviseViewer::clippingCB, this);
    clipSwitch->addChild(clipCallback);
    sceneGraph->addChild(clipSwitch);
    // default plane equation ( = no clipping)
    eqn[0] = 0.0;
    eqn[1] = 0.0;
    eqn[2] = 0.0;
    eqn[3] = 0.0;

    // reference part
    // initialize refPoint
    refPoint[0] = 0.0;
    refPoint[1] = 0.0;
    refPoint[2] = 0.0;

    //
    // add the selection node under which everything else will
    // be placed
    //
    sceneGraph->addChild(selection);

    // finish clipping
    finishClipCallback = new SoCallback;
    finishClipCallback->setCallback(InvCoviseViewer::finishClippingCB, this);
    sceneGraph->addChild(finishClipCallback);

#ifdef __linux__
    tpHandler = new TPHandler();
    sceneGraph->addChild(tpHandler->getRoot());
#endif

    //
    // put everything together
    //
    // sceneRoot->addChild(sceneGraph);
    // sceneRoot->addChild(sceneColor);

    //
    // Widget and menu variables
    //
    mgrWidget = NULL;
    //   showMenuFlag = TRUE;
    topbarMenuWidget = NULL;
    pageinMenuWidget = NULL;
    popupWidget = NULL;

    //
    // File
    //
    fileName = NULL;
    fileDialog = NULL;
    browser = NULL;
    printDialog = NULL;

    antialiasingFlag = FALSE;
    backgroundColorEditor = NULL;

    //
    // Selection
    //     These callbacks are used to update the SceneViewer state after
    //     the current selection changes (e.g. attach/detach editors and manips).
    //
    selection->addSelectionCallback(InvCoviseViewer::selectionCallback, this);
    //    selection->addSelectionCallback(InvPlaneMover::selectionCB, pm_);
    selection->addDeselectionCallback(InvCoviseViewer::deselectionCallback, this);
    //   selection->addDeselectionCallback(InvPlaneMover::deSelectionCB, pm_);
    selection->setPickFilterCallback(InvCoviseViewer::pickFilterCB, this);
    //selection->setPickFilterCallback(InvPlaneMover::pickFilterCB, (void *) pm_);
    selectionCallbackInactive = FALSE; // flag is needed since the needed temporal removal of selection
    // callbacks seems to be broken
    highlightRA = new SoBoxHighlightRenderAction;

    // current transform node to watch for data change
    currTransformNode = NULL;
    currTransformPath = NULL;

    // List of transform nodes related to the current transform node
    transformNode = NULL;
    transformNode = NULL;

    //
    // add the transform sensor for the transformation change reporting
    //
    transformSensor = new SoNodeSensor(InvCoviseViewer::transformCallback, this);

    //
    // Editors
    //
    ignoreCallback = FALSE;
    materialEditor = NULL;
    colorEditor = NULL;
    transformSliderSet = NULL;
    partEditor_ = NULL;
    annoEditor_ = NULL;
    clippingPlaneEditor = NULL;

    //
    // Manips
    //
    curManip = SV_NONE;
    highlightRA->setVisible(TRUE); // highlight visible when no manip
    curManipReplaces = TRUE;
    maniplist = new InvManipList;

    //
    // User callback
    //
    userModeCB = NULL;
    userModedata = NULL;
    userModeFlag = FALSE;

    // Lights
    ambientColorEditor = NULL;
    headlightData = new InvLightData;
    headlightData->classPt = this;
    headlightData->name = strdup("Headlight");
    headlightData->type = SoDirectionalLight::getClassTypeId();
    headlightData->colorEditor = NULL;
    headlightData->isManip = FALSE;
    headlightData->shouldBeManip = FALSE;
    headlightEditor = NULL;
    calculatedLightManipSize = FALSE;

    // build a object list
    list = new InvObjectList();

    // set slave setup as a default !
    master = FALSE;

    // viewer edit state = CO_OFF at default
    viewer_edit_state = CO_OFF;

    // default sync mode is master/slave
    sync_flag = SYNC_SYNC;

    // currently there is no spacemouse
    spacemouse = NULL;
}

//======================================================================
//
// Description:
//    Destructor.
//
// Use: public
//======================================================================
InvCoviseViewer::~InvCoviseViewer()
{
    delete text_manager;

    // detach and delete the manips
    detachManipFromAll();
    delete maniplist;

    // detach and delete the viewers
    currentViewer->setSceneGraph(NULL);

    // delete menu items data
    delete[/*SV_MENU_NUM*/] menuItems;
    delete headlightData;
    delete headlightEditor;

    delete printDialog;

    // Editor components
    delete materialEditor;
    delete colorEditor;
    delete transformSliderSet;
    delete ambientColorEditor;
    delete backgroundColorEditor;
    delete partEditor_;
    delete annoEditor_;
    delete clippingPlaneEditor;

    sceneGraph->unref();
}

//======================================================================
//
// Description:
//	projection of telepointer
//
//
// Use: private
//======================================================================
void
InvCoviseViewer::projectTP(InvExaminerViewer *Viewer, int mousex, int mousey,
                           SbVec3f &intersection, float &aspectRatio)
{
    // take the x,y position of the mouse, and normalize to [0,1].
    // X windows have 0,0 at the upper left,
    // Inventor expects 0,0 to be the lower left.

    const int xOffset = 61;
    const int yOffset = 33;

    float mx_norm;
    float my_norm;

    SbVec2s size = Viewer->getSize();
    // !! Attention: SoXtRenderArea with offset
    size[0] = size[0] - xOffset;
    size[1] = size[1] - yOffset;

    mx_norm = float(mousex) / size[0];
    my_norm = float(size[1] - mousey) / size[1];

    /*
     cerr << endl << "-------------" << endl;
     cerr << "x: " << mx_norm << endl;
     cerr << "y: " << my_norm << endl;
   */

    // viewport mapping
    SoOrthographicCamera *cam = (SoOrthographicCamera *)tpHandler->getCamera();
    // set aspect ratio explicitely
    aspectRatio = size[0] / (float)size[1];
    cam->aspectRatio.setValue(aspectRatio);
    // default setting
    cam->height.setValue(2.0);
    // scale height
    cam->scaleHeight(1 / aspectRatio);

    /*
     float h = cam->height.getValue();
     fprintf(stderr, "Height Camera: %.6f\n", h);
     fprintf(stderr, "Aspect Ratio Viewport: %.6f\n", aspectRatio);
   */

    // get view volume -> rectangular box
    SbViewVolume viewVolume = tpHandler->getCamera()->getViewVolume();

    // project the mouse point to a line
    SbVec3f p0, p1;
    viewVolume.projectPointToLine(SbVec2f(mx_norm, my_norm), p0, p1);

    // take the midpoint of the line as telepointer position
    intersection = (p0 + p1) / 2.0f;

    /*
     float fx,fy,fz;
     intersection.getValue(fx,fy,fz);
     cerr << "Telepointer vwr: (" << mousex << ", " << mousey << ") ("
     << fx << ", " << fy << ", " << fz << ")\n";
   */
}

//====================================================================
// X event handling routine called before events are passed to inventor
// for telepointer handling
// Several changes by Uwe Woessner
//====================================================================
SbBool
InvCoviseViewer::appEventHandler(void *userData, XAnyEvent *anyevent)
{

    InvCoviseViewer *viewer = (InvCoviseViewer *)userData;
    XMotionEvent *me;
    int mouseX, mouseY;
    // float xn, yn; // normalized screen coordinates
    float aspRat; // aspect ratio
    SbBool handled = FALSE;
    SbVec3f intersection;
    SbMatrix mx;
    SbRotation rot;
    SbVec3f vectmp;
    SbVec3f vec;
    float px, py, pz;

    char message[300];
    float pos[3];
    float ori[4];
    int view;
    float aspect;
    float near;
    float far;
    float focal;
    float angleORheightangle; // depends on camera type !

    char objName[255];
    static int rotate = CO_ON;
    static int translate = CO_ON;
    float radians;

    //
    // spacemouse stuff
    //
    if (viewer->spacemouse)
    {

        const SoEvent *event = viewer->spacemouse->translateEvent(anyevent);
        InvExaminerViewer *v = viewer->currentViewer;
        SoCamera *camera = v->getCamera();

        // Look for the last selected object

        SoPath *p = NULL;
        SoPath *xfPath;
        SoTransform *xfTrans;
        int i;
        for (i = 0; i < viewer->selection->getNumSelected(); i++)
        {
            p = (*(viewer->selection))[i];
        }

        if (i > 0)
        {
            xfPath = viewer->findTransformForAttach(p);
            xfPath->ref();
            SoFullPath *fp = (SoFullPath *)xfPath;
            xfTrans = (SoTransform *)fp->getTail();
            xfTrans->ref();
            // got the transform if i>0 !

            if (event != NULL)
            {
                if (event->isOfType(SoMotion3Event::getClassTypeId()))
                {
                    const SoMotion3Event *motion = (const SoMotion3Event *)event;

                    mx = camera->orientation.getValue();
                    if (translate == CO_ON)
                    {
                        vectmp = motion->getTranslation().getValue();
                        mx.multVecMatrix(vectmp, vec);
                        xfTrans->translation.setValue(xfTrans->translation.getValue() + vec);
                    }

                    if (rotate == CO_ON)
                    {
                        rot = motion->getRotation();
                        rot.getValue(vectmp, radians);
                        mx.multVecMatrix(vectmp, vec);
                        rot.setValue(vec, radians);
                        xfTrans->rotation.setValue(xfTrans->rotation.getValue() * rot);
                    }

                    // update slaves!
                    if (viewer->master == TRUE && viewer->sync_flag == SYNC_TIGHT)
                    {
                        strcpy(objName, "NONE");
                        viewer->findObjectName(&objName[0], p);
                        if (strcmp(objName, "NONE") != 0)
                            viewer->sendTransformation(objName, xfTrans);
                        // cerr << "Moving " << objName << "!" << endl;
                    }

                    xfTrans->unref();
                    xfPath->unref();
                }

                else if (event->isOfType(SoSpaceballButtonEvent::getClassTypeId()))
                {
                    if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON1))
                    {
                        // enable/disable rotation
                        if (rotate == CO_OFF)
                        {
                            rotate = CO_ON;
                        }
                        else
                        {
                            rotate = CO_OFF;
                        }
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON2))
                    {
                        // enable/disable translation
                        if (translate == CO_OFF)
                        {
                            translate = CO_ON;
                        }
                        else
                        {
                            translate = CO_OFF;
                        }
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON3))
                    {
                        // home translation/rotation
                        xfTrans->rotation.setValue(0.0, 0.0, 0.0, 1.0);
                        xfTrans->translation.setValue(0.0, 0.0, 0.0);
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, PICK))
                    {
                        // reset tranlation/rotation and camera !
                        v->resetToHomePosition();
                        xfTrans->rotation.setValue(0.0, 0.0, 0.0, 1.0);
                        xfTrans->translation.setValue(0.0, 0.0, 0.0);
                    }
                }
            }
        }
        else
        {
            //  move camera if no object selected

            if (event != NULL)
            {
                if (event->isOfType(SoMotion3Event::getClassTypeId()))
                {
                    const SoMotion3Event *motion = (const SoMotion3Event *)event;

                    mx = camera->orientation.getValue();
                    if (translate == CO_ON)
                    {
                        vectmp = motion->getTranslation().getValue();
                        mx.multVecMatrix(vectmp, vec);
                        camera->position.setValue(camera->position.getValue() - vec);
                    }

                    if (rotate == CO_ON)
                    {
                        float radians;
                        rot = motion->getRotation();
                        rot.getValue(vectmp, radians);
                        mx.multVecMatrix(vectmp, vec);
                        rot.setValue(vec, -radians);
                        camera->orientation.setValue(camera->orientation.getValue() * rot);
                        vectmp = camera->position.getValue();
                        mx = rot;
                        mx.multVecMatrix(vectmp, vec);
                        camera->position.setValue(vec);
                    }
                }

                else if (event->isOfType(SoSpaceballButtonEvent::getClassTypeId()))
                {
                    if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON1))
                    {
                        // enable/disable rotation
                        if (rotate == CO_OFF)
                        {
                            rotate = CO_ON;
                        }
                        else
                        {
                            rotate = CO_OFF;
                        }
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON2))
                    {
                        // enable/disable translation
                        if (translate == CO_OFF)
                        {
                            translate = CO_ON;
                        }
                        else
                        {
                            translate = CO_OFF;
                        }
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON3))
                    {
                        // home translation/rotation
                        v->viewAll();
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON5))
                    {
                        viewer->spacemouse->transScale /= 2;
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, BUTTON6))
                    {
                        viewer->spacemouse->transScale *= 2;
                    }
                    else if (SO_SPACEBALL_PRESS_EVENT(event, PICK))
                    {
                        // reset tranlation/rotation and camera !
                        v->resetToHomePosition();
                    }
                }
            }
        }
    } // if (viewer->spavemouse)

    //
    // other X stuff (telepointer, mouse movement etc.)
    //

    char mess[150];

    switch (anyevent->type)
    {

    case MotionNotify:

        //	cerr << "InvCoviseViewer::appEventHandler(..) MotionNotify " << endl;

        // update slaves!
        me = (XMotionEvent *)anyevent;
        if (me->state & (Button1Mask | Button2Mask | Button3Mask | Button4Mask | Button5Mask))
        {

            if (viewer->master == TRUE && viewer->sync_flag == SYNC_TIGHT)
            {
                viewer->getTransformation(pos, ori, &view, &aspect, &near, &far, &focal, &angleORheightangle);
                sprintf(message, "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %5d %7.3f %7.3f %7.3f %7.3f %7.3f",
                        pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3], view, aspect, near, far, focal, angleORheightangle);
                //cerr << endl << " ||||||||||||||||  AppEventHandler ! " << endl;
                rm_sendCamera(message);
                // viewer->colormap_manager->updateColormaps((void *)viewer->currentViewer);
                if (viewer->vrml_syn_)
                    viewer->sendVRMLCamera();
            }
        }
        if (me->state & KeyPressMask)
        {
            //print_comment(__LINE__,__FILE__,"mouse motion and key pressed");
            mouseX = me->x;
            mouseY = me->y;
            viewer->projectTP(viewer->currentViewer, mouseX, mouseY, intersection, aspRat);
            intersection.getValue(px, py, pz);

            // test locally
            if (viewer->tpShow_)
            {
                sprintf(mess, "%s %d %f %f %f %f", username, CO_ON, px, py, pz, aspRat);
                viewer->sendTelePointer(viewer->currentViewer, username, CO_ON, px, py, pz, aspRat);
            }
            else
            {
                sprintf(mess, "%s %d %f %f %f %f", username, CO_OFF, px, py, pz, aspRat);
                viewer->sendTelePointer(viewer->currentViewer, username, CO_OFF, px, py, pz, aspRat);
            }

            viewer->tpHandler->handle(mess, viewer->currentViewer);

            if (viewer->vrml_syn_)
                rm_sendVRMLTelePointer(mess);
#ifdef _AIRBUS
            handled = FALSE;
#else
            handled = TRUE;
#endif
        }
        break;

    case KeyPress:
    {

        //	cerr << "InvCoviseViewer::appEventHandler(..) KeyPress " << endl;

        XKeyEvent *keyEvent = (XKeyEvent *)anyevent;

        const int bufLen(32);
        char buf[bufLen];
        KeySym keysym_return;

        XLookupString(keyEvent, buf, bufLen, &keysym_return, NULL);

        if ((keysym_return == XK_Shift_L) || (keysym_return == XK_Shift_R))
        {
            me = (XMotionEvent *)anyevent;
            mouseX = me->x;
            mouseY = me->y;
            viewer->projectTP(viewer->currentViewer, mouseX, mouseY, intersection, aspRat);
            intersection.getValue(px, py, pz);
            viewer->sendTelePointer(viewer->currentViewer, username, CO_ON, px, py, pz, aspRat);
            // test locally
            sprintf(mess, "%s %d %f %f %f %f", username, CO_ON, px, py, pz, aspRat);

            viewer->tpHandler->handle(mess, viewer->currentViewer);
            //print_comment(__LINE__,__FILE__,"Key pressed");

            viewer->tpShow_ = true;
            if (viewer->vrml_syn_)
                rm_sendVRMLTelePointer(mess);

#ifndef _AIRBUS
            handled = TRUE;
#else
            handled = FALSE;
#endif
        }
    }
    break;

    //print_comment(__LINE__,__FILE__,"Key pressed");

    case KeyRelease:

        me = (XMotionEvent *)anyevent;
        mouseX = me->x;
        mouseY = me->y;
        viewer->projectTP(viewer->currentViewer, mouseX, mouseY, intersection, aspRat);
        intersection.getValue(px, py, pz);
        viewer->sendTelePointer(viewer->currentViewer, username, CO_RMV, px, py, pz, aspRat);
        // test locally
        sprintf(mess, "%s %d %f %f %f %f", username, CO_RMV, px, py, pz, aspRat);
        viewer->tpHandler->handle(mess, viewer->currentViewer);

        if (viewer->vrml_syn_)
            rm_sendVRMLTelePointer(mess);

        viewer->tpShow_ = false;
        handled = TRUE;
        break;

    default:
        handled = FALSE;
        break;
    }

#ifdef _COLLAB_VIEWER

    if (strcmp(username, "me"))
    {
        me = (XMotionEvent *)anyevent;
        mouseX = me->x;
        mouseY = me->y;
        viewer->projectTP(viewer->currentViewer, mouseX, mouseY, intersection, aspRat);
        intersection.getValue(px, py, pz);
        viewer->sendTelePointer(viewer->currentViewer, username, CO_ON, px, py, pz, aspRat);
        // test locally
        sprintf(mess, "%s %d %f %f %f %f", username, CO_ON, px, py, pz, aspRat);
        if (viewer->isHeadlight())
        {
            sprintf(mess, "%s %d %f %f %f %f", username, CO_ON, px, py, pz, aspRat);
            viewer->sendTelePointer(viewer->currentViewer, username, CO_ON, px, py, pz, aspRat);
        }
        else
        {
            sprintf(mess, "%s %d %f %f %f %f", username, CO_OFF, px, py, pz, aspRat);
            viewer->sendTelePointer(viewer->currentViewer, username, CO_OFF, px, py, pz, aspRat);
        }
        viewer->tpHandler->handle(mess, viewer->currentViewer);
        //print_comment(__LINE__,__FILE__,"Key pressed");

        viewer->tpShow_ = true;
        if (viewer->vrml_syn_)
            rm_sendVRMLTelePointer(mess);
    }
#endif

    return handled;
}

//======================================================================
//
// Description:
//	receive new telepointer
//
//
// Use: public
//======================================================================
void InvCoviseViewer::receiveTelePointer(char *message)
{
    tpHandler->handle(message, currentViewer);
}

//======================================================================
//
// Description:
//	receive new telepointer
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendTelePointer(InvExaminerViewer *, char *tpname,
                                      int state, float px, float py, float pz,
                                      float aspectRatio)
{
    char message[MAXPATHLEN + 150];

    sprintf(message, "%s %d %f %f %f %f", tpname, state, px, py, pz, aspectRatio);

    // set the transformation in the correct node
    rm_sendTelePointer(message);
}

//======================================================================
//
// Description:
//	send drawstyle message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendDrawstyle(int still, int dynamic)
{
    char message[90];

    sprintf(message, "%d %d", still, dynamic);

    if (master == TRUE && sync_flag > SYNC_LOOSE)
        rm_sendDrawstyle(message);
}

//======================================================================
//
// Description:
//	send selection message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendSelection(char *name)
{
    char message[100];

    if (master == TRUE && sync_flag > SYNC_LOOSE && name != NULL)
    {
        sprintf(message, "%s", name);
        rm_sendSelection(message);
        // cerr << "InvCoviseViewer::sendSelection: sending out message" << endl;
    }
}

//======================================================================
//
// Description:
//	send selection message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendDeselection(char *name)
{
    char message[100];

    if (master == TRUE && sync_flag > SYNC_LOOSE && name != NULL)
    {
        sprintf(message, "%s", name);
        rm_sendDeselection(message);
        // cerr << "InvCoviseViewer::sendDeselection: sending out message" << endl;
    }
}

//======================================================================
//
// Description:
//	send part switching message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendPart(int partId, int switchTag)
{
    char message[90];

    sprintf(message, "%d %d", partId, switchTag);

    if (master == TRUE && sync_flag > SYNC_LOOSE)
        rm_sendPart(message);
}

//======================================================================
//
// Description:
//	send "reference part" message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendReferencePart(int refPartId)
{
    char message[90];

    sprintf(message, "%d", refPartId);

    if (master == TRUE && sync_flag > SYNC_LOOSE)
        rm_sendReferencePart(message);
}

//======================================================================
//
// Description:
//	send "reset scene" message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendResetScene()
{
    if (master == TRUE && sync_flag > SYNC_LOOSE)
        rm_sendResetScene();
}

//======================================================================
//
// Description:
//	send transparency message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendTransparency(int level)
{
    char message[90];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d", level);
        rm_sendTransparency(message);
    }
}

//======================================================================
//
// Description:
//	send fog message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendFog(int level)
{
    char message[90];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d", level);
        rm_sendFog(message);
    }
}

//======================================================================
//
// Description:
//	send antialiasing message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendAntialiasing(int level)
{
    char message[90];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d", level);
        rm_sendAntialiasing(message);
    }
}

//======================================================================
//
// Description:
//	send viewing message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendViewing(int level)
{
    char message[90];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d", level);
        rm_sendViewing(message);
    }
}

//======================================================================
//
// Description:
//	send axis message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendAxis(int level)
{
    char message[90];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d", level);
        rm_sendAxis(message);
    }
}

//======================================================================
//
// Description:
//	send clipping plane message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendClippingPlane(int onoroff, double equation[])
{
    char message[100];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d %f %f %f %f", onoroff, equation[0], equation[1], equation[2], equation[3]);
        rm_sendClippingPlane(message);
    }
}

//======================================================================
//
// Description:
//	send backcolor message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendBackcolor(float r, float g, float b)
{
    char message[95];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%f %f %f", r, g, b);
        rm_sendBackcolor(message);
    }
}

//======================================================================
//
// Description:
//	send decoration message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendDecoration(int level)
{
    char message[90];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d", level);
        rm_sendDecoration(message);
    }
}

//======================================================================
//
// Description:
//	send headlight message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendHeadlight(int level)
{
    char message[90];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%d", level);
        rm_sendHeadlight(message);
    }
}

//======================================================================
//
// Description:
//	send colormap message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendColormap(char *string)
{
    char message[100];

    if (master == TRUE && sync_flag > SYNC_LOOSE)
    {
        sprintf(message, "%s", string);
        rm_sendColormap(message);
    }
}

//======================================================================
//
// Description:
//	send sync mode message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendSyncMode()
{

    if (master == TRUE)
    {
        char message[90];

        if (sync_flag == SYNC_LOOSE)
            strcpy(message, "LOOSE");
        else if (sync_flag == SYNC_SYNC)
            strcpy(message, "SYNC");
        else
            strcpy(message, "TIGHT");

        rm_sendSyncMode(message);

        // now update all object transformations according to master positions
    }
}

//======================================================================
//
// Description:
//	send light mode message after it has changed here
//      and we are master
//
//
// Use: private
//======================================================================
void InvCoviseViewer::sendLightMode(int type)
{
    char message[90];

    sprintf(message, "%d", type);

    if (master == TRUE && sync_flag > SYNC_LOOSE)
        rm_sendLightMode(message);
}

//======================================================================
//
// Description:
//    Hide editors in slave
//
// Use: private
//======================================================================
void InvCoviseViewer::removeEditors()
{
    if (master == FALSE)
    {

        // Editor components
        if (materialEditor != NULL)
        {
            if (materialEditor->isAttached())
                materialEditor->detach();
            materialEditor->hide();
        }

        if (colorEditor != NULL)
        {
            if (colorEditor->isAttached())
                colorEditor->detach();
            colorEditor->hide();
        }

        if (transformSliderSet != NULL)
        { // hope that helps
            transformSliderSet->hide();
            ;
        }

        if (ambientColorEditor != NULL)
        {
            if (ambientColorEditor->isAttached())
                ambientColorEditor->detach();
            ambientColorEditor->hide();
        }

        if (backgroundColorEditor != NULL)
        {
            if (backgroundColorEditor->isAttached())
                backgroundColorEditor->detach();
            backgroundColorEditor->hide();
        }
    }
}

//======================================================================
//
// Description:
//    Bring opened editors back on the screen
//
// Use: private
//======================================================================
void InvCoviseViewer::showEditors()
{

    if (master == FALSE)
    {

        // Editor components
        if (materialEditor != NULL)
        {
            if (!materialEditor->isVisible())
                materialEditor->show();
        }

        if (colorEditor != NULL)
        {
            if (!colorEditor->isVisible())
                colorEditor->show();
        }

        if (transformSliderSet != NULL)
        {
            if (!transformSliderSet->isVisible())
                transformSliderSet->show();
            ;
        }

        if (ambientColorEditor != NULL)
        {
            if (!ambientColorEditor->isVisible())
                ambientColorEditor->show();
        }

        if (backgroundColorEditor != NULL)
        {
            if (!backgroundColorEditor->isVisible())
                backgroundColorEditor->show();
        }
    }
}

//======================================================================
//
// Description:
//	receive drawstyle message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveDrawstyle(char *message)
{
    int still, dynamic;

    if (sscanf(message, "%d %d", &still, &dynamic) != 2)
    {
        fprintf(stderr, "InvCoviseViewer::receiveDrawstyle: sscanf failed\n");
    }

    // set drawstyle in viewer ...
    setDrawStyle((InvViewer::DrawType)still, (InvViewer::DrawStyle)dynamic);
}

//======================================================================
//
// Description:
//	receive fog message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveFog(char *message)
{
    int onoroff;

    if (sscanf(message, "%d", &onoroff) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveFog: sscanf failed\n");
    }

    // set fog in viewer ...
    setFog(onoroff);
}

//======================================================================
//
// Description:
//	receive antialiasing message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveAntialiasing(char *message)
{
    int onoroff;

    if (sscanf(message, "%d", &onoroff) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveAntialiasing: sscanf failed\n");
    }

    // set antialiasing in viewer ...
    setAntialiasing(onoroff);
}

//======================================================================
//
// Description:
//	receive axis message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveAxis(char *message)
{
    int onoroff;

    if (sscanf(message, "%d", &onoroff) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveAxis: sscanf failed\n");
    }

    // set antialiasing in viewer ...
    setAxis(onoroff);
}

//======================================================================
//
// Description:
//	receive clipping plane message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveClippingPlane(char *message)
{
    int onoroff;
    double equation[4];

    if (sscanf(message, "%d %lf %lf %lf %lf", &onoroff, &equation[0], &equation[1], &equation[2], &equation[3]) != 5)
    {
        fprintf(stderr, "InvCoviseViewer::receiveClippingPlane: sscanf failed\n");
    }

    // set clipping plane in viewer ...
    setClipping(onoroff);
    if (onoroff == CO_ON)
        setClippingPlane(equation);
}

//======================================================================
//
// Description:
//	receive viewing message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveViewing(char *message)
{
    int onoroff;

    if (sscanf(message, "%d", &onoroff) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveViewing: sscanf failed\n");
    }

    // set viewing mode in viewer ...
    setViewing(onoroff);
}

//======================================================================
//
// Description:
//	receive backcolor message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveBackcolor(char *message)
{
    float r, g, b;

    if (sscanf(message, "%f %f %f", &r, &g, &b) != 3)
    {
        fprintf(stderr, "InvCoviseViewer::receiveBackcolor: sscanf failed\n");
    }

    // set backcolor in viewer ...
    currentViewer->setBackgroundColor(SbColor(r, g, b));
    // keep fog color up to date with bkg color
    environment->fogColor.setValue(SbColor(r, g, b));
}

//======================================================================
//
// Description:
//	receive viewing message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveHeadlight(char *message)
{
    int onoroff;

    if (sscanf(message, "%d", &onoroff) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveHeadlight: sscanf failed\n");
    }

    // set headlight mode in viewer ...
    setHeadlight(onoroff);
}

//======================================================================
//
// Description:
//	receive colormap message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveColormap(char *message)
{
    char name[100];

    //cerr << "InvCoviseViewer::receiveColormap(..) called" << endl;
    if (sscanf(message, "%s", &name[0]) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveColormap: sscanf failed\n");
    }
    colormap_manager->hideAllColormaps();
    //cerr << "InvCoviseViewer::receiveColormap(..) DRAW cmap " << endl;
    colormap_manager->showColormap(name, (SoXtExaminerViewer *)currentViewer);
    selectColormapListItem(TRUE, name);

    // updating data used for storing the current colormap selection
    cmapSelected_.removeAll();
    if (strcmp(name, "NONE") != 0)
    {
        // internally we keep track only of the reduced name
        char *red_name = new char[strlen(name) + 1];
        char *p_cut;

        red_name = new char[strlen(name) + 1];
        strcpy(red_name, name);
        p_cut = strrchr(red_name, '_');
        if (p_cut)
        {
            p_cut[0] = '\0';
        }

        if (c_first_time == 0)
        {
            strcpy(c_oldname, "DEFAULT");
            c_first_time = 1;
        }

        cmapSelected_.add(red_name, 1);

        strcpy(c_oldname, red_name);
        delete[] red_name;
    }
}

//======================================================================
//
// Description:
//	check ascii string for extended ascii characters
//
// Use: private
//======================================================================

void InvCoviseViewer::checkAscii(char *tgt, const char *src)
{
    unsigned int pos, end;

    for (pos = 0, end = 0; pos < strlen(src) && end < 255; pos++)
    {
        if (isascii(src[pos]))
        {
            tgt[end++] = src[pos];
        }
        else
        {
            switch ((unsigned char)src[pos])
            {
            case 0xFC:
            {
                tgt[end++] = 'u';
                tgt[end++] = 'e';
                break;
            }
            case 0xE4:
            {
                tgt[end++] = 'a';
                tgt[end++] = 'e';
                break;
            }
            case 0xC4:
            {
                tgt[end++] = 'A';
                tgt[end++] = 'e';
                break;
            }
            case 0xF6:
            {
                tgt[end++] = 'o';
                tgt[end++] = 'e';
                break;
            }
            case 0xD6:
            {
                tgt[end++] = 'O';
                tgt[end++] = 'e';
                break;
            }
            case 0xDC: //Ü
            {
                tgt[end++] = 'U';
                tgt[end++] = 'e';
                break;
            }
            }
        }
    }
    tgt[end] = '\0';
}

//======================================================================
//
// Description:
//	receive colormap message
//
//
// Use: private
//======================================================================
void
InvCoviseViewer::addColormap(const char *name, const char *colormap)
{

    char annotation[255];
    char thename[255];

    char *token, *message;
    float map_min = 0.f;
    float map_max = 1.f;
    int num_color = 0;
    int num_steps = 2;
    float *r, *g, *b, *a;
    //
    // parse colormap
    //
    //cerr << colormap << endl;

    char chch[255];
    char lastCh[255][20];

    char *redName = new char[strlen(name) + 1];
    char *orgName = new char[strlen(name) + 1];
    strcpy(orgName, name);
    //++++++++++++++++++++++++++++++++++++++++++++++++++++
    strcpy(chch, name);
    char del[3];
    strcpy(del, "_");
    char *tok;
    tok = strtok(chch, del);
    int cnt = 0;
    while (tok)
    {
        strcpy(lastCh[cnt], tok);
        tok = strtok(NULL, del);
        cnt++;
    }
    int ii;
    strcpy(redName, lastCh[0]);
    strcat(redName, del);
    for (ii = 1; ii < cnt - 1; ++ii)
    {
        strcat(redName, lastCh[ii]);
        if (ii != cnt - 2)
            strcat(redName, del);
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++
    message = new char[strlen(colormap) + 1];
    strcpy(message, colormap);

    token = strtok(message, "\n");

    if (token)
        strcpy(thename, token);
    else
        strcpy(thename, "");
    if (token && strlen(token) < 255)
        strcpy(thename, token);
    else
        strcpy(thename, "Name out of range");

    token = strtok(NULL, "\n");

    if (token != NULL)
    {
        if (strlen(token) < 255)
        {
            // look for non-displayed ascii characters
            checkAscii(annotation, token);
        }
        //strcpy(annotation,token);
        else
            strcpy(annotation, "Annotation out of range");
    }
    else
    {
        // cerr << "Broken colormap" << endl;
        return;
    }
    char *p = NULL;
    p = strtok(NULL, "\n");
    if (p)
        map_min = atof(p);
    p = strtok(NULL, "\n");
    if (p)
        map_max = atof(p);
    p = strtok(NULL, "\n");
    if (p)
        num_color = atoi(p);
    p = strtok(NULL, "\n");
    if (p)
        num_steps = atoi(p);

    //  cerr << "COLORMAP: " << map_min << " : "
    //       << map_max << " : "
    //       << num_color << " : "
    //       << num_steps << endl;

    r = new float[num_color];
    g = new float[num_color];
    b = new float[num_color];
    a = new float[num_color];

    for (int i = 0; i < num_color; i++)
    {
        p = strtok(NULL, "\n");
        if (p)
            *(r + i) = atof(p);
        p = strtok(NULL, "\n");
        if (p)
            *(g + i) = atof(p);
        p = strtok(NULL, "\n");
        if (p)
            *(b + i) = atof(p);
        p = strtok(NULL, "\n");
        if (p)
            *(a + i) = atof(p);

        //   cerr << r[i] << " : " << g[i] << " : " << b[i] << endl;
    }

    // cerr << "Adding a colormap in InvCoviseViewer::addColormap" << endl;

    // we have to add the colormap with its original name in order to maintain the connection
    // to the obj of which it defines the colors (RM. 06.04.01)
    char *format;
    if (currentViewer->getUseNumberFormat())
    {
        format = currentViewer->getNumberFormat();
    }
    else
    {
        format = NULL;
    }
    colormap_manager->addColormap(format, orgName, num_color, r, g, b, a,
                                  map_min, map_max, num_steps, annotation,
                                  cmap_x_0, cmap_y_0, cmap_size);

    delete[] r;
    delete[] g;
    delete[] b;
    delete[] message;

    //
    // add to colormap list
    // internally we keep track only of the reduced name (RM. 06.04.01)
    //
    addToColormapList(redName);

#ifdef _AIRBUS
    colormap_manager->showColormap(redName, (SoXtExaminerViewer *)(this->currentViewer));
#endif
    delete[] orgName;
    delete[] redName;
}

//======================================================================
//
// Description:
//	build a colormap entry for the new colormap.
//
// Use: private
//
//
//======================================================================
void InvCoviseViewer::selectColormapListItem(int onORoff, const char *name)
{

    int itempos = XmListItemPos(colormaplist, XmStringCreateSimple((char *)name));
    if (onORoff)
    {
        XmListSelectPos(colormaplist, itempos, False);
        XmUpdateDisplay(colormaplist);
    }
    else
    {
        XmListDeselectPos(colormaplist, itempos);
        XmUpdateDisplay(colormaplist);
    }
}

//======================================================================
//
// Description:
//	build a colormap entry for the new colormap.
//
// Use: private
//
//
//======================================================================
void InvCoviseViewer::addColormapMenuEntry(const char *name)
{

    //
    // create the motif menu entry
    //
    Arg args[12];
    WidgetClass widgetClass;
    //String  	    	callbackReason;
    Widget subMenu = menuItems[SV_COLORMAP].widget;
    Widget subButton;
    int n;
    // makes sure menu has been built
    if (subMenu == NULL)
        return;

    // create the submenu widget, adding a callback to update the radio buttons
    int argnum = 0;
#ifdef MENUS_IN_POPUP
    SoXt::getPopupArgs(XtDisplay(subMenu), 0, args, &argnum);
#endif

    widgetClass = xmToggleButtonGadgetClass;
    /**** callbackReason =XmNvalueChangedCallback; ****/
    XtSetArg(args[0], XmNindicatorType, XmONE_OF_MANY);
    n = 1;
    XmString xmstr = XmStringCreate((char *)name, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
    (void)xmstr;
    subButton = XtCreateWidget(
        name,
        widgetClass,
        subMenu, args, n);

    //    XtAddCallback(subButton, callbackReason,
    //		(XtCallbackProc)InvCoviseViewer::toggleColormapMenu,
    //		(XtPointer) &menuItems[SV_COLORMAP]);
    // manage children
    XtManageChild(subButton);
}

//======================================================================
//
// Description:
//	build a colormap entry for the new colormap.
//
// Use: private
//
//
//======================================================================
void InvCoviseViewer::toggleColormapMenu(Widget theWidget, XtPointer, XtPointer)
{
    (void)theWidget;
    // cerr << "toggling buttons in colormap menu" << endl;
}

//======================================================================
//
// Description:
//	receive colormap message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::replaceColormap(const char *name, const char *colormap)
{
    char annotation[255];
    char thename[255];
    char *token, *message;
    float map_min;
    float map_max;
    int num_color;
    int num_steps;
    float *r, *g, *b, *a;

    //cerr << "InvCoviseViewer::replaceColormap(..) called" << endl;

    //cerr << "Deleting a colormap in InvCoviseViewer::replacingColormap" << endl;
    colormap_manager->removeColormap(name);

    removeFromColormapList(name);

    // cerr << colormap << endl;

    message = new char[strlen(colormap) + 1];
    strcpy(message, colormap);

    token = strtok(message, "\n");

    strcpy(thename, token);
    if (strlen(token) < 255)
        strcpy(thename, token);
    else
        strcpy(thename, "Name out of range");

    token = strtok(NULL, "\n");

    if (token != NULL)
    {
        if (strlen(token) < 255)
            strcpy(annotation, token);
        else
            strcpy(annotation, "Annotation out of range");
    }
    else
    {
        // cerr << "Broken colormap" << endl;
        return;
    }
    map_min = atof(strtok(NULL, "\n"));
    map_max = atof(strtok(NULL, "\n"));
    num_color = atoi(strtok(NULL, "\n"));
    num_steps = atoi(strtok(NULL, "\n"));

    r = new float[num_color];
    g = new float[num_color];
    b = new float[num_color];
    a = new float[num_color];

    for (int i = 0; i < num_color; i++)
    {
        *(r + i) = atof(strtok(NULL, "\n"));
        *(g + i) = atof(strtok(NULL, "\n"));
        *(b + i) = atof(strtok(NULL, "\n"));
        *(a + i) = atof(strtok(NULL, "\n"));
    }

    // cerr << "Adding a colormap in InvCoviseViewer::addColormap" << endl;
    char *format;
    if (currentViewer->getUseNumberFormat())
    {
        format = currentViewer->getNumberFormat();
    }
    else
    {
        format = NULL;
    }

    colormap_manager->addColormap(format, name, num_color, r, g, b, a, map_min, map_max, num_steps, annotation, cmap_x_0, cmap_y_0, cmap_size);

    //cerr << "InvCoviseViewer::replaceColormap(..) DRAW cmap" << endl;
    colormap_manager->showColormap(name, (SoXtExaminerViewer *)(this->currentViewer));

    addToColormapList(name);

    delete[] r;
    delete[] g;
    delete[] b;
    delete[] message;
}

//======================================================================
//
// Description:
//	receive colormap message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::deleteColormap(const char *name)
{

    // delete colormap
    //
    // ...
    //

    // convert name to the displayed one COLLECT_1_OUT_001 -> COLLECT_1_OUT
    char *orgName = new char[1 + strlen(name)];
    char *redName = new char[1 + strlen(name)];
    strcpy(orgName, name);
    char chch[255];
    char lastCh[255][20];

    strcpy(chch, name);
    char del[3];
    strcpy(del, "_");
    char *tok;
    tok = strtok(chch, del);
    int cnt = 0;
    while (tok)
    {
        strcpy(lastCh[cnt], tok);
        tok = strtok(NULL, del);
        cnt++;
    }
    int ii;
    strcpy(redName, lastCh[0]);
    strcat(redName, del);
    for (ii = 1; ii < cnt - 1; ++ii)
    {
        strcat(redName, lastCh[ii]);
        if (ii != cnt - 2)
            strcat(redName, del);
    }

    if (colormap_manager->removeColormap(orgName))
    {
        removeFromColormapList(redName);
    }
    else
    {
        //    cerr << "No colormap to delete for object " << redName << endl;
    }

    delete[] orgName;
    delete[] redName;
}

//======================================================================
//
// Description:
//	add a part
//
//
// Use: private
//======================================================================
void InvCoviseViewer::addPart(const char *name, int partId, SoSwitch *s)
{
    std::string *objName = new std::string(name);

    if (s)
    {
        multiHash.insert(partId, *objName);
        /*
      cout << "\nMultiHash table:" << endl;
      for (int i=0;i<10;i++) {
        iter=multiHash[i]; // !! i is key-variable
        if (iter) {
      while (iter) {
       cout <<"("<<i<<"/"<<iter()<< " [" << iter.hash() << "])  ";
       ++iter;
        }
      cout << endl;
        }
      }
      cout << "--------------------" << endl;
      */
        switchHash.insert(partId, s);
    }

    delete objName;
}

//======================================================================
//
// Description:
//	replace a part
//
//
// Use: private
//======================================================================
void
InvCoviseViewer::replacePart(const char *name, int partId, SoSwitch *s)
{
    deletePart(name);
    addPart(name, partId, s);
}

//======================================================================
//
// Description:
//      delete a part
//
//
// Use: private
//======================================================================
void
InvCoviseViewer::deletePart(const char *name)
{
    if (NULL == name)
    {
        return;
    }
    std::string *objName = new std::string(name);

    switchIter = switchHash.first();
    iter = multiHash.first();
    for (; iter; ++iter)
    {
        if (iter() == *objName)
        {
            multiHash.remove(iter);
            switchHash.remove(switchIter);
        }
        ++switchIter;
    }
    delete objName;
}

//======================================================================
//
// Description:
//	add a time-part
//
//
// Use: private
//======================================================================
void
InvCoviseViewer::addTimePart(const char *name, int timeStep, int partId, SoSwitch *s)
{
    std::string *objName = new std::string(name);
    TimePart *tp = new TimePart(timeStep, partId);

    if (s)
    {
        nameHash.insert(*tp, *objName);
        /* only for testing
      cout << "\nnameHash table:" << endl;

      for (int i=0; i<3; i++){
        for (int j=0;j<10;j++){
      TimePart *tiPa = new TimePart(i, j);
      nameIter=nameHash[*tiPa];
      if (nameIter) {
       while (nameIter) {
         cout <<"("<<i<<","<<j<<"/"<<nameIter()<< " [" << nameIter.hash() << "])  ";
         ++nameIter;
      }
      cout << endl;
      }
      delete tiPa;
      }
      }
      cout << "--------------------" << endl;
      */
        referHash.insert(*tp, s);
    }

    delete objName;
    delete tp;
}

//======================================================================
//
// Description:
//	replace a time-part
//
//
// Use: private
//======================================================================
void
InvCoviseViewer::replaceTimePart(const char *name, int timeStep, int partId, SoSwitch *s)
{
    deleteTimePart(name);
    addTimePart(name, timeStep, partId, s);
}

//======================================================================
//
// Description:
//      delete a time-part
//
//
// Use: private
//======================================================================
void InvCoviseViewer::deleteTimePart(const char *name)
{

    std::string *objName = new std::string(name);

    nameIter = nameHash.first();
    referIter = referHash.first();
    for (; nameIter; ++nameIter)
    {
        if (nameIter() == *objName)
        {
            nameHash.remove(nameIter);
            referHash.remove(referIter);
        }
        ++referIter;
    }
    delete objName;
}

//======================================================================
//
// Description:
//	receive viewing message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveDecoration(char *message)
{
    int onoroff;

    if (sscanf(message, "%d", &onoroff) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveDecoration: sscanf failed\n");
    }

    // set decoration mode in viewer ...
    print_comment(__LINE__, __FILE__, "Setting new decoration");
    setDecoration(onoroff);
}

//======================================================================
//
// Description:
//	receive viewing message
//
//
// Use: private
//======================================================================
void InvCoviseViewer::receiveProjection(char *message)
{
    int onoroff;

    if (sscanf(message, "%d", &onoroff) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveProjection: sscanf failed\n");
    }

    // set camera projection mode in viewer ...
    print_comment(__LINE__, __FILE__, "Setting new projection");

    if (onoroff == 1) // perspective
    {
        currentViewer->toggleCameraType();
        //currentViewer->setCameraType(SoPerspectiveCamera::getClassTypeId());
    }
    else // ortho
    {
        currentViewer->toggleCameraType();
        //currentViewer->setCameraType(SoOrthographicCamera::getClassTypeId());
    }
}

//======================================================================
//
// Description:
//	receive synchronisation mode  message
//
//
// Use: public
//======================================================================
void InvCoviseViewer::receiveSyncMode(char *message)
{

    if (strcmp("LOOSE", message) == 0)
    {
        sync_flag = SYNC_LOOSE;
        // update menu availability for loose coupling
        setSyncMode(sync_flag);
    }
    else if (strcmp("SYNC", message) == 0)
    {
        sync_flag = SYNC_SYNC;
        setSyncMode(sync_flag);
    }
    else
    {
        sync_flag = SYNC_TIGHT;
        setSyncMode(sync_flag);
    }
}

//======================================================================
//
// Description:
//	receive light mode mode  message
//
//
// Use: public
//======================================================================
void InvCoviseViewer::receiveLightMode(char *message)
{

    int type;

    if (sscanf(message, "%d", &type) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveLightMode: sscanf failed\n");
    }

    if (type == SoLightModel::BASE_COLOR)
    {
        // update menu availability for light mode
        setLightMode(SoLightModel::BASE_COLOR);
    }
    else
    {
        setLightMode(SoLightModel::PHONG);
    }
}

//======================================================================
//
// Description:
//	receive light selection  message
//
//
// Use: public
//======================================================================
void
InvCoviseViewer::receiveSelection(char *name)
{
    // cerr << "InvCoviseViewer::receiveSelection (Slave) " << name << endl;
    setSelection(name);
}

//======================================================================
//
// Description:
//	receive deselection message
//
//
// Use: public
//======================================================================
void
InvCoviseViewer::receiveDeselection(char *name)
{
    // cerr << "InvCoviseViewer::receiveDeselection (Slave) " << name << endl;
    setDeselection(name);
}

//======================================================================
//
// Description:
//	receive part switching message
//
//
// Use: public
//======================================================================
void
InvCoviseViewer::receivePart(char *message)
{
    // cerr << "InvCoviseViewer::receiveDeselection (Slave) " << name << endl;
    int partId, switchTag;

    if (sscanf(message, "%d %d", &partId, &switchTag) != 2)
    {
        fprintf(stderr, "InvCoviseViewer::receivePart: sscanf failed\n");
    }

    // switching part on/off
    switchPart(partId, switchTag);
}

//======================================================================
//
// Description:
//	receive message to set a reference part
//
//
// Use: public
//======================================================================
void InvCoviseViewer::receiveReferencePart(char *message)
{
    int refPartId;

    if (sscanf(message, "%d", &refPartId) != 1)
    {
        fprintf(stderr, "InvCoviseViewer::receiveReferencePart: sscanf failed\n");
    }

    // set a reference part for the animation
    setReferencePoint(refPartId);
    transformScene(refPartId);
}

//======================================================================
//
// Description:
//	receive message to reset scene
//
//
// Use: public
//======================================================================
void InvCoviseViewer::receiveResetScene()
{
    // reset tarnsformed scene for animation
    resetTransformedScene();
}

//======================================================================
//
// Description:
//	receive transparency message
//
//
// Use: public
//======================================================================
void InvCoviseViewer::receiveTransparency(char *name)
{

    setTransparency(name);
}

//======================================================================
//
// Description:
//	user starts editing in viewer.
//
//
// Use: private
//======================================================================
void InvCoviseViewer::viewerStartEditCB(void *data, InvViewer *)
{

    InvCoviseViewer *r = (InvCoviseViewer *)data;

    if (r->master == TRUE)
    {
        r->viewer_edit_state = CO_ON;
    }
}

//======================================================================
//
// Description:
//	user finishs editing in viewer.
//
//
// Use: private
//======================================================================
void InvCoviseViewer::viewerFinishEditCB(void *data, InvViewer *)
{

    InvCoviseViewer *r = (InvCoviseViewer *)data;

    // let's send a transformation
    if (r->master == TRUE && r->sync_flag > SYNC_LOOSE)
    {
        r->viewer_edit_state = CO_OFF;

        //
        // get the current camera and pass the values to the communication
        // manager
        //
        //   longer messages for NaN
        char message[300];
        float pos[3];
        float ori[4];
        int view;
        float aspect;
        float near;
        float far;
        float focal;
        float angleORheightangle; // depends on camera type !

        r->getTransformation(pos, ori, &view, &aspect, &near, &far, &focal, &angleORheightangle);

        //
        // pack into character string

        sprintf(message, "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %5d %7.3f %7.3f %7.3f %7.3f %7.3f",
                pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3], view, aspect, near, far, focal, angleORheightangle);

        rm_sendCamera(message);
    }
}

#ifdef CO_hp1020

static float pos[4][3] = {
    { 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 }
};
static float colors[3][3] = {
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 }
};
static int32_t indices[6] = {
    1, 0, 2, -1, 0, 3
};
#endif

//======================================================================
//
// Description:
//	axis setup.
//
//
// Use: private
//======================================================================
SoNode *InvCoviseViewer::makeAxis()
{
#ifdef CO_hp1020
    //SoInput in;

    //in.setBuffer(axis,strlen(axis));
    //SoNode *result=NULL;

    //SoDB::read(&in,result);
    SoSeparator *root = new SoSeparator;
    //SoMaterial *myMaterials = new SoMaterial;
    //myMaterials->diffuseColor.setValues(0, 3, colors);
    //root->addChild(myMaterials);
    //SoMaterialBinding *myMaterialBinding = new SoMaterialBinding;
    //myMaterialBinding->value = SoMaterialBinding::PER_PART;
    //root->addChild(myMaterialBinding);
    SoCoordinate3 *myCoords = new SoCoordinate3;
    myCoords->point.setValues(0, 4, pos);
    root->addChild(myCoords);
    SoIndexedLineSet *myLineSet = new SoIndexedLineSet;
    myLineSet->coordIndex.setValues(0, 6, indices);
    root->addChild(myLineSet);

    SoSeparator *s1 = new SoSeparator;
    SoMaterial *redMaterial = new SoMaterial;
    // Red
    redMaterial->diffuseColor.setValue(1.0, 0.0, 0.0);
    SoCone *c1 = new SoCone();
    c1->bottomRadius.setValue(0.1);
    c1->height.setValue(0.3);
    SoTranslation *t1 = new SoTranslation;
    t1->translation.setValue(1, 0, 0);
    SoRotationXYZ *r1 = new SoRotationXYZ;
    r1->axis.setValue(SoRotationXYZ::Z);
    r1->angle.setValue(-1.570796327);
    s1->addChild(redMaterial);
    s1->addChild(t1);
    s1->addChild(r1);
    s1->addChild(c1);
    SoSeparator *s2 = new SoSeparator;
    SoMaterial *greenMaterial = new SoMaterial;
    // Green
    greenMaterial->diffuseColor.setValue(0.0, 1.0, 0.0);
    SoCone *c2 = new SoCone();
    c2->bottomRadius.setValue(0.1);
    c2->height.setValue(0.3);
    SoTranslation *t2 = new SoTranslation;
    t2->translation.setValue(0, 1, 0);
    SoRotationXYZ *r2 = new SoRotationXYZ;
    r2->axis.setValue(SoRotationXYZ::Y);
    r2->angle.setValue(-1.570796327);
    s2->addChild(greenMaterial);
    s2->addChild(t2);
    s2->addChild(c2);
    SoSeparator *s3 = new SoSeparator;
    SoMaterial *blueMaterial = new SoMaterial;
    // Blue
    blueMaterial->diffuseColor.setValue(0.0, 0.0, 1.0);
    SoCone *c3 = new SoCone();
    c3->bottomRadius.setValue(0.1);
    c3->height.setValue(0.3);
    SoTranslation *t3 = new SoTranslation;
    t3->translation.setValue(0, 0, 1);
    SoRotationXYZ *r3 = new SoRotationXYZ;
    r3->axis.setValue(SoRotationXYZ::X);
    r3->angle.setValue(1.570796327);
    s3->addChild(blueMaterial);
    s3->addChild(t3);
    s3->addChild(r3);
    s3->addChild(c3);

    root->addChild(s1);
    root->addChild(s2);
    root->addChild(s3);

    return root;
#else
    SoInput in;

    in.setBuffer((void *)axis, strlen(axis));
    SoNode *result;

    SoDB::read(&in, result);

    return result;
#endif
}

//======================================================================
//
// Description:
//	ask if Master
//
// Use: public
//======================================================================
int InvCoviseViewer::isMaster()
{
    return master;
}

//======================================================================
//
// Description:
//	ask if Master
//
// Use: public
//======================================================================
int InvCoviseViewer::isSynced()
{
    return sync_flag;
}

//======================================================================
//
// Description:
//	Master setup.
//
// Use: public
//======================================================================
void InvCoviseViewer::setMaster()
{
    master = TRUE;
    setMasterSlaveMenu(master);
    XtSetSensitive(MasterRequest, !master);
}

//======================================================================
//
// Description:
//	Slave setup.
//
// Use: public
//======================================================================
void InvCoviseViewer::setSlave()
{
    master = FALSE;
    setMasterSlaveMenu(master);
    XtSetSensitive(MasterRequest, !master);
}

//======================================================================
//
// Description:
//	Master/Slave switch .
//
//
// Use: public
//======================================================================
void InvCoviseViewer::setMasterSlave()
{
    if (master == FALSE)
    {
        master = TRUE;
        setMasterSlaveMenu(master);
    }
    else
    {
        master = FALSE;
        setMasterSlaveMenu(master);
    }
    XtSetSensitive(MasterRequest, !master);
}

//======================================================================
//
// Description:
//	New data is going to be coming into the viewer.  Time to disconnect all
//  manipulators and picking, and wait for new information.  Might as well go
//  into a viewing mode as well, this gets rid of the manipulators, and puts
//  the user in control of viewing when new data shows up.
//
// Use: public
//
//======================================================================
void InvCoviseViewer::newData()
{
    selection->deselectAll();
}

//======================================================================
//
// Description:
//	This sets the user mode callack routine
//
// Use: public
//
//======================================================================
void
InvCoviseViewer::setUserModeEventCallback(SoXtRenderAreaEventCB *fcn)
{
    userModeCB = fcn;
    userModedata = currentViewer->getSceneGraph();
    if (userModeFlag)
    {
        currentViewer->setEventCallback(userModeCB, userModedata);
        cerr << "InvCoviseViewer::setUserModeEventCallback(..) setEventCallback userModeCB" << endl;
    }
}

//======================================================================
//
// Description:
//	Move up the picked path to the parent group.
//
// Use: public
//
//======================================================================
void InvCoviseViewer::pickParent()
{
    SoFullPath *pickPath;
    int parentIndex = 0;

    // We'll pick the parent of the last selection in the list...
    pickPath = (SoFullPath *)(*selection)[selection->getNumSelected() - 1];
    if (pickPath == NULL || pickPath->getLength() < 2)
        return;

    // Get actual node that is the current selection:
    SoNode *tail = pickPath->getTail();
    SoNode *kitTail = ((SoNodeKitPath *)pickPath)->getTail();
    SoType nkt = SoBaseKit::getClassTypeId();
    if (kitTail->isOfType(nkt))
        tail = kitTail;
    else
        kitTail = NULL;

    // If kitTail is at top of path, we've already gone as high as we can go.
    if (kitTail == pickPath->getHead())
        return;

    // Get index of parent of selection.
    if (kitTail != NULL)
    {
        // Look for first kit above tail. If none, use direct parent of kitTail.
        SoNode *aboveTail = ((SoNodeKitPath *)pickPath)->getNodeFromTail(1);
        SbBool aboveIsKit = aboveTail->isOfType(nkt);
        for (int i = pickPath->getLength() - 1; i >= 0; i--)
        {
            if (aboveIsKit)
            {
                if (pickPath->getNode(i) == aboveTail)
                {
                    parentIndex = i;
                    break;
                }
            }
            else if (pickPath->getNode(i) == kitTail)
            {
                parentIndex = i - 1;
                break;
            }
        }
    }
    else
    {
        // If tail is not a nodekit, parentIndex is just parent of tail...
        parentIndex = pickPath->getLength() - 2;
    }

    // cannot select the selection node (make sure we're not)
    if (pickPath->getNode(parentIndex) == selection)
    {
        fprintf(stderr, "No more parents to pick (cannot pick above the selection node)\n");
        return;
    }

    pickPath->ref(); // need to ref it, because
    // selection->clear unref's it
    selection->deselectAll();
    pickPath->truncate(parentIndex + 1); // Make path end at parentIndex
    selection->select(pickPath); // add path back in
    pickPath->unref(); // now we can unref it, again
}

//======================================================================
//
// Description:
//	Pick all group nodes and shapes under selection.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::pickAll()
{
    selection->deselectAll();

#ifdef DEBUG
    assert(selection != NULL);
#endif

    SoPathList myPaths;

    // Our callbacks on the selection's 'select' method may add
    // more children to the selection node (by making a trackball or handlebox)
    // Therefore, we must first determine the selections by storing
    // paths to them.
    // Following this, we call 'select' on each path, in turn.

    //
    // Create paths from the selection node to all of it's children
    // that are groups or shapes.
    //
    for (int i = 0; i < selection->getNumChildren(); i++)
    {
        SoNode *node = selection->getChild(i);
        if ((node->isOfType(SoGroup::getClassTypeId()) || node->isOfType(SoShape::getClassTypeId())))
        {
            SoPath *thisPath = new SoPath(selection);
            thisPath->append(i);

            myPaths.append(thisPath);
        }
    }

    //
    // Select each path in 'myPaths'
    //
    for (int j = 0; j < myPaths.getLength(); j++)
        selection->select(myPaths[j]);
}

//======================================================================
//
// Description:
//	This routine first detaches manipulators from all selected objects,
//      then attaches a manipulator to all selected objects.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::replaceAllManips(
    InvEManipMode manipMode) // Current manipulator
{
    detachManipFromAll();
    attachManipToAll(manipMode);
}

//======================================================================
//
// Description:
//	This routine attaches a manipulator to all selected objects.
//
// Use: private
//
//======================================================================
// Current manipulator
void InvCoviseViewer::attachManipToAll(InvEManipMode manipMode)
{
    int i;

    for (i = 0; i < selection->getNumSelected(); i++)
    {
        SoPath *p = (*selection)[i];
        attachManip(manipMode, p);
    }
}

//======================================================================
//
// Description:
//	This routine attaches and activates a manipulator.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::attachManip(
    InvEManipMode manipMode, // Current manipulator
    SoPath *selectedPath) // Which selection to attach to
{
    SoTransformManip *theXfManip;
    SoPath *xfPath;

    //
    // Attach to a manipulator.
    //

    if (manipMode == SV_NONE)
        return;

    xfPath = findTransformForAttach(selectedPath);
    xfPath->ref();
    theXfManip = NULL;

    switch (manipMode)
    {
    case SV_TRACKBALL:
        theXfManip = new SoTrackballManip;
        break;

    case SV_HANDLEBOX:
        theXfManip = new SoHandleBoxManip;

        break;

    case SV_JACK:
        theXfManip = new SoJackManip;
        break;

    case SV_CENTERBALL:
        theXfManip = new SoCenterballManip;
        break;

    case SV_XFBOX:
        theXfManip = new SoTransformBoxManip;
        break;

    case SV_TABBOX:
        theXfManip = new SoTabBoxManip;
        break;

    case SV_NONE:
        return;
    }

    if (theXfManip)
    {

        SoFullPath *fp = (SoFullPath *)xfPath;

#ifdef DEBUG
        if (!fp->getTail()->isOfType(SoTransform::getClassTypeId()))
        {
            fprintf(stderr, "DBG> Fatal Error: in InvCoviseViewer::attachManip\n");
            fprintf(stderr, "   > end of path is not a transform\n");
        }
#endif
        SoTransform *oldXf = (SoTransform *)fp->getTail();
        oldXf->ref();
        theXfManip->ref();

        if (!theXfManip->replaceNode(xfPath))
        {
            theXfManip->unref();
#ifdef DEBUG
            fprintf(stderr, "DBG> Fatal Error: in InvCoviseViewer::attachManip\n");
            fprintf(stderr, "   > manip->replaceNode() failed!\n");
#endif
        }

        // If the transformSliderSet is attached to the oldXf, then attach
        // it to the new manip instead.
        if (transformSliderSet && transformSliderSet->isVisible()
            && transformSliderSet->getNode() == oldXf)
        {
            transformSliderSet->setNode(theXfManip);
        }

        // Add manip and paths to the maniplist (maniplist will ref/unref)
        ///	maniplist->append(selectedPath, theXfManip, xfPath );
        maniplist->append(selectedPath, theXfManip, (SoPath *)xfPath);
        theXfManip->unref();
        oldXf->unref();

        if (manipMode == SV_TABBOX)
        {
            // Special case!  When using a  tab box, we want to adjust the
            // scale tabs upon viewer finish.
            currentViewer->addFinishCallback(
                &InvCoviseViewer::adjustScaleTabSizeCB, theXfManip->getDragger());
        }

        if (manipMode == SV_JACK)
        {
            // Special case! For jack manip, we want it so that clicking on the
            // selected object initiates 2-dimensional translation.  Other
            // parts of the jack manip should use the default resource geometry.
            // So, we replace the parts that do planar motion.
            // We need to replace a total of six parts:
            //     'translator.yzTranslator.translator':
            //     'translator.xzTranslator.translator':
            //     'translator.xyTranslator.translator':
            //     'translator.yzTranslator.translatorActive':
            //     'translator.xzTranslator.translatorActive':
            //     'translator.xyTranslator.translatorActive':
            // In the SoJackDragger, 'translator' is an SoDragPointDragger,
            // which takes care of all translations in 3 dimensions for jack.
            // In the SoDragPointDragger there are 3 planar translation parts
            // (each is an SoTranslate2Dragger) and 3 linear translation parts
            // (each is an SoTranslate1Dragger).  At any given time, dragPoint
            // displays one of each kind of dragger. We leave the linear
            // translators as the default geometry (a cylinder along the axis
            // of motion), but replace the geometry in the planar translators.
            // Within the SoDragPointDragger, the planar translators are named
            // 'yzTranslator', 'xzTranslator', and 'xyTranslator'.
            // Each of these is an SoTranslate2Dragger, which has two parts
            // for its geometry, 'translator' and 'translatorActive.' Clicking
            // on the 'translator' is what initiates the 2D translation.
            // Once begun, the 'translatorActive' part is displayed. We
            // replace both of these parts with a path to the selected object.

            // When we call setPartAsPath, we need to prune the path if our
            // selected geometry lays inside a nodekit.
            // For example, let's say a dumbBellKit contains 1 bar (a cylinder)
            // and 2 end (spheres).  Since this is the lowest-level kit
            // containing these 3 shapes, they are considered a single object
            // by the SceneViewer.
            // So we need to pass a path to the kit, not the individual piece
            // that is selected.  This way, subsequent clicks on any piece of
            // the dumbbell will cause 2D translation.
            // First, is a nodekit on the path? If so, find the last one.
            SoFullPath *jackP = (SoFullPath *)selectedPath;
            SoType bkType = SoBaseKit::getClassTypeId();
            int lastKitInd = -1;
            for (int i = jackP->getLength() - 1; i >= 0; i--)
            {
                if (jackP->getNode(i)->isOfType(bkType))
                {
                    lastKitInd = i;
                    break;
                }
            }
            // If there's a lastKitInd, make jackP be a copy of
            // selectedPath, but only up to lastKitInd.
            if (lastKitInd != -1)
                jackP = (SoFullPath *)selectedPath->copy(0, lastKitInd + 1);

            // Get the dragger from the manip (the manip contains the dragger,
            // and the dragger has the parts).
            SoDragger *d = theXfManip->getDragger();

            // Use jackP to set the translator parts, then discard (unref) it:
            jackP->ref();
            d->setPartAsPath("translator.yzTranslator.translator", jackP);
            d->setPartAsPath("translator.xzTranslator.translator", jackP);
            d->setPartAsPath("translator.xyTranslator.translator", jackP);
            d->setPartAsPath(
                "translator.yzTranslator.translatorActive", jackP);
            d->setPartAsPath(
                "translator.xzTranslator.translatorActive", jackP);
            d->setPartAsPath(
                "translator.xyTranslator.translatorActive", jackP);
            jackP->unref();
        }
    }

    xfPath->unref();
}

//======================================================================
//
// Description:
//	This routine detaches the manipulators from all selected objects.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::detachManipFromAll()
{
    //
    // Loop from the end of the list to the start.
    //
    for (int i = selection->getNumSelected() - 1; i >= 0; i--)
    {
        SoPath *p = (SoPath *)(*selection)[i];
        detachManip(p);
    }
}

//======================================================================
//
// Description:
//	This routine detaches a manipulator.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::detachManip(
    SoPath *p) // Selection object that is being removed
{
    //
    // Detach manip and remove from scene graph.
    //
    int which = maniplist->find(p);
    // See if this path is registered in the manip list.
    if (which != -1)
    {
        // remove from scene graph
        SoTransformManip *manip = (SoTransformManip *)maniplist->getManip(which);

        if (manip->isOfType(SoTabBoxManip::getClassTypeId()))
        {
            // Special case!  When using a  tab box, we want to adjust the
            // scale tabs upon viewer finish.
            currentViewer->removeFinishCallback(
                &InvCoviseViewer::adjustScaleTabSizeCB, manip->getDragger());
        }

        SoPath *xfPath = maniplist->getXfPath(which);
        SoTransform *newXf = new SoTransform;
        newXf->ref();
        manip->ref();

        // replace the manip
        manip->replaceManip(xfPath, newXf);

        // If the transformSliderSet is attached to the manip, then attach
        // it to the new node instead.
        if (transformSliderSet && transformSliderSet->isVisible()
            && transformSliderSet->getNode() == manip)
            transformSliderSet->setNode(newXf);

        manip->unref();
        newXf->unref();

        // remove from maniplist
        maniplist->remove(which);
    }
}

//======================================================================
//
// Description:
//	Added as a finish callback to the current viewer. It makes sure
//      the scale tab size gets changed when a viewer gesture is
//      completed.
//
// Use: public
//
//======================================================================
void InvCoviseViewer::adjustScaleTabSizeCB(void *userData, InvViewer *)
{
    SoTabBoxDragger *dragger = (SoTabBoxDragger *)userData;
    dragger->adjustScaleTabSize();
}

//======================================================================
//
// Description:
//	See the selection from the camera
//
// Use: public
//
//======================================================================
void InvCoviseViewer::viewSelection()
{

    if (selection->getNumSelected() == 0)
    {
        viewAll();
        return;
    }

    SoPath *path = (*selection)[0];
    if (path != NULL)
    {
        getCamera()->viewAll(path, currentViewer->getViewportRegion());
    }
    else
    {
        viewAll();
        return;
    }
}

//======================================================================
//
// Description:
//	Create a color editor for the currently selected object.
//      Attachment code copied from SoXformManip.c++
//
// Use: private
//
//======================================================================
void InvCoviseViewer::createColorEditor()
{
    if (colorEditor == NULL)
    {
        colorEditor = new MyColorEditor;
        colorEditor->setWYSIWYG(TRUE);
        colorEditor->setTitle("Diffuse Color");
    }

    SoMaterial *editMaterial = findMaterialForAttach(NULL);

    if (editMaterial != NULL)
    {
        colorEditor->attach(&(editMaterial->diffuseColor), 0, editMaterial);
    }

    colorEditor->show();
}

//======================================================================
//
// Description:
//   Find the appropriate material node in the scene graph to attach a material
//   editor to.
//
//   Two possible cases:
//        [1] The path-tail is NOT a group.  We search the siblings of the path
//            tail (including the tail itself) from right to left for a node
//	      that is affected by materials (shapes or groups).
//            We stop the search if we come to a material node to the left of the
//	      pathTail.  If we find a node that IS affected by material, we will
//	      insert a material node just before the path-tail. This is
//            because the editor should not affect nodes that appear
//            before attachPath in the scene graph.
//        [2] The path-tail IS a group.  We search the children from left to
//            right for material nodes.
//            We stop the search if we come to a material node.
//            If we find a node that is affected by materials, we will insert a
//	      material just before this node. This is because the editor for a
//	      group should affect ALL nodes within that group.
//
// NOTE: For the purposes of this routine, we consider SoSwitch as different
//       from other types of group. This is because we don't want to put
//       the new node underneath the switch, but next to it.
//
// Use: private
//
//
//======================================================================
SoMaterial *InvCoviseViewer::findMaterialForAttach(
    const SoPath *target) // path to start search from
{
    int pathLength;
    SoPath *selectionPath;
    SoMaterial *editMtl = NULL;

    SbBool madeNewMtl = FALSE; // did we create a new material
    // node within this method?

    if ((selectionPath = (SoPath *)target) == NULL)
    {
        //
        //  If no selection path is specified, then use the LAST path in the
        //  current selection list.
        //
        // last guy
        selectionPath = (*selection)[selection->getNumSelected() - 1];
    }
    pathLength = selectionPath->getLength();

    if (pathLength <= 0)
    {
        fprintf(stderr, "No objects currently selected...\n");
        return NULL;
    }

#ifdef DEBUG
    if (pathLength < 2)
    {
        fprintf(stderr, "Picked object has no parent...\n");
        return NULL;
    }
#endif

    // find 'group' and try to find 'editMtl'
    SoGroup *group = NULL;
    SoNode *node;
    int index, i;
    SbBool ignoreNodekit = FALSE;

    if (selectionPath->getTail()->isOfType(SoBaseKit::getClassTypeId()))
    {
        // Nodekits have their own built in policy for creating new material
        // nodes. Allow them to contruct and return it.
        // Get the last nodekit in the path:
        SoBaseKit *kit = (SoBaseKit *)((SoNodeKitPath *)selectionPath)->getTail();
        // SO_CHECK_PART returns NULL if the part doesn't exist yet...
        editMtl = SO_GET_PART(kit, "material", SoMaterial);
        if (editMtl == NULL)
        {
            // This nodekit does not have a material part.
            // Ignore the fact that this is a nodekit.
            ignoreNodekit = TRUE;
        }
    }

    SbBool isTailGroup = selectionPath->getTail()->isOfType(SoGroup::getClassTypeId()) && (!selectionPath->getTail()->isOfType(SoSwitch::getClassTypeId()));

    if ((editMtl == NULL) && (!isTailGroup))
    {
        //
        //    CASE 1: The path-tail is not a group.
        //    'group'      becomes the second to last node in the path.
        //    We search the path tail and its siblings from right to left for a
        //    mtl node.
        //    We stop the search if we come to a shape node or a group node
        //    to the left of the pathTail.  If we find a shape or group, we
        //    will insert a mtl just before the path-tail. This is
        //    because the manipulator should not affect objects that appear
        //    before selectionPath in the scene graph.
        //
        group = (SoGroup *)selectionPath->getNode(pathLength - 2);
        index = group->findChild(selectionPath->getTail());

        for (i = index; (i >= 0) && (editMtl == NULL); i--)
        {
            node = group->getChild(i);
            // found SoMaterial
            if (node->isOfType(SoMaterial::getClassTypeId()))
                editMtl = (SoMaterial *)node;
            else if (i != index)
            {
                if (isAffectedByMaterial(node))
                    break;
            }
        }

        if (editMtl == NULL)
        {
            editMtl = new SoMaterial;
            group->insertChild(editMtl, index);
            madeNewMtl = TRUE;
        }
    }
    else if (editMtl == NULL)
    {
        //    CASE 2: The path-tail is a group.
        //    'group'      becomes the path tail
        //      We search the children from left to right for mtl nodes.
        //      We stop the search if we come to a shape node or a group node.
        //      If we find a shape or group, we will insert a mtl just
        //      before this shape or group. This is because the editor
        //      for a group should affect ALL objects within that group.
        //
        group = (SoGroup *)selectionPath->getTail();
        for (i = 0; (i < group->getNumChildren()) && (editMtl == NULL); i++)
        {
            node = group->getChild(i);
            if (node->isOfType(SoMaterial::getClassTypeId()))
                editMtl = (SoMaterial *)node;
            else if (isAffectedByMaterial(node))
                break;
        }

        if (editMtl == NULL)
        {
            editMtl = new SoMaterial;
            group->insertChild(editMtl, i);
            madeNewMtl = TRUE;
        }
    }

    // If we just created the material node here, then set the ignore
    // flags for all fields in the node.  This will cause the fields
    // to be inherited from their ancestors. The material editor will
    // undo these flags whenever it changes the value of a field
    if (madeNewMtl == TRUE)
    {
        editMtl->ambientColor.setIgnored(TRUE);
        editMtl->diffuseColor.setIgnored(TRUE);
        editMtl->specularColor.setIgnored(TRUE);
        editMtl->emissiveColor.setIgnored(TRUE);
        editMtl->shininess.setIgnored(TRUE);
        editMtl->transparency.setIgnored(TRUE);
    }

    // If any of the fields is ignored, then fill the value with the value
    // inherited from the rest of the scene graph
    if (editMtl->ambientColor.isIgnored()
        || editMtl->diffuseColor.isIgnored()
        || editMtl->specularColor.isIgnored()
        || editMtl->emissiveColor.isIgnored()
        || editMtl->shininess.isIgnored()
        || editMtl->transparency.isIgnored())
    {

        // Create a path to the material
        SoPath *mtlPath;
        if ((!ignoreNodekit) && selectionPath->getTail()->isOfType(SoBaseKit::getClassTypeId()))
        {
            SoBaseKit *kit = (SoBaseKit *)((SoNodeKitPath *)selectionPath)->getTail();
            mtlPath = kit->createPathToPart("material", TRUE, selectionPath);
            mtlPath->ref();
        }
        else
        {
            if (!isTailGroup)
            {
                // CASE 1: path-tail was NOT 'group' -- copy all but last entry
                mtlPath = selectionPath->copy(0, pathLength - 1);
            }
            else
            {
                // CASE 2: path-tail was 'group' -- copy all of editPath
                mtlPath = selectionPath->copy(0, pathLength);
            }
            mtlPath->ref();
            // add the material to the end of the path
            if (group)
            {
                int mtlIndex = group->findChild(editMtl);
                mtlPath->append(mtlIndex);
            }
            else
                fprintf(stderr, "InvCoviseViewer.cpp::findMaterialForAttach(): group used uninitialized\n");
        }

        // Pass the material node to an accumulate state callback
        // that will load any 'ignored' values with their inherited values.
        SoCallbackAction cba;
        cba.addPreTailCallback(InvCoviseViewer::findMtlPreTailCB, editMtl);
        cba.apply(mtlPath);

        mtlPath->unref();
    }

    return (editMtl);
}

//======================================================================
//
// Description:
//   Callback used by 'findMaterialForAttach' as part of the accumulate state
//   action. Returns 'PRUNE', which tells the action not to draw the
//   shape as part of the accum state action.
//   editor to.
//
// Use: private
//
//
//======================================================================
SoCallbackAction::Response
InvCoviseViewer::findMtlPreTailCB(void *data, SoCallbackAction *accum,
                                  const SoNode *)
//
////////////////////////////////////////////////////////////////////////
{
    SoMaterial *mtl = (SoMaterial *)data;

    SbColor ambient, diffuse, specular, emissive;
    float shininess, transparency;

    accum->getMaterial(ambient, diffuse, specular, emissive,
                       shininess, transparency);

    // inherit the accumulated values only in those fields being ignored.
    if (mtl->ambientColor.isIgnored())
        mtl->ambientColor.setValue(ambient);
    if (mtl->diffuseColor.isIgnored())
        mtl->diffuseColor.setValue(diffuse);
    if (mtl->specularColor.isIgnored())
        mtl->specularColor.setValue(specular);
    if (mtl->emissiveColor.isIgnored())
        mtl->emissiveColor.setValue(emissive);
    if (mtl->shininess.isIgnored())
        mtl->shininess.setValue(shininess);
    if (mtl->transparency.isIgnored())
        mtl->transparency.setValue(transparency);

    return SoCallbackAction::ABORT;
}

//======================================================================
//
// Description:
//   Find the appropriate transform node in the scene graph for attaching a
//   transform editor or manipulator.
//
//   How we treat the 'center' field of the transform node:
//   If we need to create a new transform node:
//       set the 'center' to be the geometric center of all objects
//       affected by that transform.
//   If we find a transform node that already exists:
//       'center' will not be changed.
//
//   Three possible cases:
//        [1] The path-tail is a node kit. Just ask the node kit for a path
//            to the part called "transform"
//        [2] The path-tail is NOT a group.  We search the siblings of the path
//            tail (including the tail itself) from right to left for a node
//	      that is affected by transforms (shapes, groups, lights,cameras).
//            We stop the search if we come to a transform node to the left of
//	      the pathTail.  If we find a node that IS affected by transform,
//	      we will insert a transform node just before the path-tail. This is
//            because the editor should not affect nodes that appear
//            before attachPath in the scene graph.
//        [3] The path-tail IS a group.  We search the children from left to
//            right for transform nodes.
//            We stop the search if we come to a transform node.
//            If we find a node that is affected by transform, we will insert a
//	      transform just before this node. This is because the editor for a
//	      group should affect ALL nodes within that group.
//
// NOTE: For the purposes of this routine, we consider SoSwitch as different
//       from other types of group. This is because we don't want to put
//       the new node underneath the switch, but next to it.
//
// Use: private
//
//
//======================================================================
SoPath *InvCoviseViewer::findTransformForAttach(
    const SoPath *target) // path to start search from
{
    int pathLength;
    SoPath *selectionPath;
    SoTransform *editXform;

    // fprintf(stderr,"In findTransformForAttach");

    if ((selectionPath = (SoPath *)target) == NULL)
    {
        //
        //  If no selection path is specified, then use the LAST path in the
        //  current selection list.
        //
        selectionPath = (*selection)[selection->getNumSelected() - 1];
    }
    pathLength = selectionPath->getLength();

    if (pathLength <= 0)
    {
        fprintf(stderr, "No objects currently selected...\n");
        return NULL;
    }

#ifdef DEBUG
    if (pathLength < 2)
    {
        fprintf(stderr, "Picked object has no parent...\n");
        return NULL;
    }
#endif

    // find 'group' and try to find 'editXform'
    SoGroup *group = NULL;
    SoNode *node;
    int index, i;
    SbBool isTailGroup, isTailKit;
    SbBool existedBefore = FALSE;
    SoPath *pathToXform = NULL;

    editXform = NULL;

    isTailGroup = (selectionPath->getTail()->isOfType(SoGroup::getClassTypeId())
                   && !selectionPath->getTail()->isOfType(SoSwitch::getClassTypeId()));

    isTailKit = selectionPath->getTail()->isOfType(SoBaseKit::getClassTypeId());

    //    CASE 1: The path-tail is a node kit.
    if (isTailKit)
    {

        // Nodekits have their own built in policy for creating new transform
        // nodes. Allow them to contruct and return a path to it.
        SoBaseKit *kit = (SoBaseKit *)((SoNodeKitPath *)selectionPath)->getTail();

        // Before creating path, see if the transform part exists yet:
        if (SO_CHECK_PART(kit, "transform", SoTransform) != NULL)
            existedBefore = TRUE;

        if ((editXform = SO_GET_PART(kit, "transform", SoTransform)) != NULL)
        {
            pathToXform = kit->createPathToPart("transform", TRUE, selectionPath);
            pathToXform->ref();
        }
        else
        {
            // This nodekit has no transform part.
            // Treat the object as if it were not a nodekit.
            isTailKit = FALSE;
        }
    }

    if (!isTailGroup && !isTailKit)
    {
        //
        //    CASE 2: The path-tail is not a group.
        //    'group'      becomes the second to last node in the path.
        //    We search the path tail and its siblings from right to left for a
        //    transform node.
        //    We stop the search if we come to a 'movable' node
        //    to the left of the pathTail.  If we find a movable node, we
        //    will insert a transform just before the path-tail. This is
        //    because the manipulator should not affect objects that appear
        //    before selectionPath in the scene graph.
        //
        group = (SoGroup *)selectionPath->getNode(pathLength - 2);
        index = group->findChild(selectionPath->getTail());

        for (i = index; (i >= 0) && (editXform == NULL); i--)
        {
            node = group->getChild(i);
            // found an SoMaterial
            if (node->isOfType(SoTransform::getClassTypeId()))
                editXform = (SoTransform *)node;
            else if (i != index)
            {
                if (isAffectedByTransform(node))
                    break;
            }
        }

        if (editXform == NULL)
        {
            existedBefore = FALSE;
            editXform = new SoTransform;
            group->insertChild(editXform, index);
        }
        else
            existedBefore = TRUE;
    }
    else if (!isTailKit)
    {
        //    CASE 3: The path-tail is a group.
        //    'group'      becomes the path tail
        //      We search the children from left to right for transform nodes.
        //      We stop the search if we come to a movable node.
        //      If we find a movable node, we will insert a transform just
        //      before this node. This is because the editor
        //      for a group should affect ALL objects within that group.
        //
        group = (SoGroup *)selectionPath->getTail();
        for (i = 0; (i < group->getNumChildren()) && (editXform == NULL); i++)
        {
            node = group->getChild(i);
            if (node->isOfType(SoTransform::getClassTypeId()))
                editXform = (SoTransform *)node;
            else if (isAffectedByTransform(node))
                break;
        }

        if (editXform == NULL)
        {
            existedBefore = FALSE;
            editXform = new SoTransform;
            group->insertChild(editXform, i);
        }
        else
            existedBefore = TRUE;
    }

    // If we don't have a path yet (i.e., we weren't handed a nodekit path)
    // create the 'pathToXform'
    // by copying editPath and making the last node in the path be editXform
    if (pathToXform == NULL)
    {
        if (!isTailGroup)
            // CASE 2: path-tail was NOT 'group' -- copy all but last entry
            pathToXform = selectionPath->copy(0, pathLength - 1);
        else
            // CASE 3: path-tail was 'group' -- copy all of editPath
            pathToXform = selectionPath->copy(0, pathLength);
        pathToXform->ref();

        // add the transform to the end
        if (group)
        {
            int xfIndex = group->findChild(editXform);
            pathToXform->append(xfIndex);
        }
        else
            fprintf(stderr, "InvCoviseViewer::findTransformForAttach(): group used uninitialized\n");
    }

    // Now. If we created the transform node right here, right now, then
    // we will set the 'center' field based on the geometric center. We
    // don't do this if we didn't create the transform, because "maybe it
    // was that way for a reason."
    if (existedBefore == FALSE)
    {
        // First, find 'applyPath' by popping nodes off the path until you
        // reach a separator. This path will contain all nodes affected by
        // the transform at the end of 'pathToXform'
        SoFullPath *applyPath = (SoFullPath *)pathToXform->copy();
        applyPath->ref();
        for (int i = (applyPath->getLength() - 1); i > 0; i--)
        {
            if (applyPath->getNode(i)->isOfType(SoSeparator::getClassTypeId()))
                break;
            applyPath->pop();
        }

        // Next, apply a bounding box action to applyPath, and reset the
        // bounding box just before the tail of 'pathToXform' (which is just
        // the editXform). This will assure that the only things included in
        // the resulting bbox will be those affected by the editXform.
        SoGetBoundingBoxAction bboxAction(currentViewer->getViewportRegion());
        bboxAction.setResetPath(pathToXform, TRUE, SoGetBoundingBoxAction::BBOX);
        bboxAction.apply(applyPath);

        applyPath->unref();

        // Get the center of the bbox in world space...
        SbVec3f worldBoxCenter = bboxAction.getBoundingBox().getCenter();

        // Convert it into local space of the transform...
        SbVec3f localBoxCenter;
        SoGetMatrixAction ma(currentViewer->getViewportRegion());
        ma.apply(pathToXform);
        ma.getInverse().multVecMatrix(worldBoxCenter, localBoxCenter);

        // Finally, set the center value...
        editXform->center.setValue(localBoxCenter);
    }

    pathToXform->unrefNoDelete();
    return (pathToXform);
}

//======================================================================
//
// Description:
//   Find the appropriate label node in the scene graph for sending to
//   slave renderers.
//   The given target path goes from the selection node to a selected
//   shape node
//
// Use: private
//
//
//======================================================================
// path to start search from
void InvCoviseViewer::findObjectName(char *objName, const SoPath *selectionPath)
{
    int j;
    const char *name;
    SbName string;

    SoNode *node = selectionPath->getTail();

    //  fprintf(stderr,"In findObjectName");

    //
    // look on the left side if there is
    //
    SoGroup *sep = (SoGroup *)selectionPath->getNodeFromTail(1);
    (void)node;
    //
    // should be a separator !

    if (sep->isOfType(SoSeparator::getClassTypeId()))
    {
        // look for the label under the separator

        for (j = 0; j < sep->getNumChildren(); j++)
        {
            SoNode *n = sep->getChild(j);

            if (n->isOfType(SoLabel::getClassTypeId()))
            {
                // look into the label
                SoLabel *l = (SoLabel *)sep->getChild(j);
                string = l->label.getValue();
                name = string.getString();
                strcpy(objName, name);
                break;
            }
        }
    }
}

//======================================================================
//
// Description:
//   Update Object View of the master & its slaves
//
//
// Use: public
//
//
//======================================================================

void InvCoviseViewer::updateSlaves()
{
    updateObjectView();
}

//======================================================================
//
// Description:
//   Find the appropriate transform node in the scene graph for a
//   given object name.
//
// Use: private
//
//
//======================================================================
void InvCoviseViewer::updateObjectView()
{

    SoTransform *transform;
    const char *name;
    char objName[255];
    SbName string;
    int i, j;

    SoSearchAction saLabel;
    SoPathList listLabel;
    SoLabel *label;

    saLabel.setFind(SoSearchAction::TYPE);
    saLabel.setInterest(SoSearchAction::ALL);
    saLabel.setType(SoLabel::getClassTypeId());

    saLabel.apply(selection);

    // get the list of paths
    listLabel = saLabel.getPaths();

    // cycle through the list and find a match
    if (listLabel.getLength() != 0)
    {
        for (i = 0; i < listLabel.getLength(); i++)
        {
            label = (SoLabel *)(listLabel[i]->getTail());
            string = label->label.getValue();
            name = string.getString();
            strcpy(&objName[0], name);

            SoGroup *group = (SoGroup *)(listLabel[i]->getNodeFromTail(1));

            for (j = 0; j < group->getNumChildren(); j++)
            {
                SoNode *n = group->getChild(j);
                if (n->isOfType(SoTransform::getClassTypeId()))
                {
                    transform = (SoTransform *)group->getChild(j);
                    // send Transformation to the slave renderers
                    sendTransformation(objName, transform);
                }
            }
        }
    }
    else
        print_comment(__LINE__, __FILE__, "Currently no objects in renderer");

    if (c_first_time)
    {
        int sel = 0;
        cmapSelected_.get(c_oldname, sel);
        if (sel == 1)
        {
            char buffer[255];
            sprintf(buffer, "%s", colormap_manager->currentColormap());
            sendColormap(buffer);
            sleep(1); //  redirection problem
        }
    }

    InvCoviseViewer::cameraCallback(this, NULL);
}

//======================================================================
//
// Description:
//   Find the appropriate switch node in the scene graph for a
//   given object name.
//
// Use: private
//
//
//======================================================================
SoSwitch *InvCoviseViewer::findSwitchNode(char *Name)
{
    //print_comment(__LINE__,__FILE__,"in findSwitchNode");
    char name[255];

    strcpy(name, "S_");
    strcat(name, Name);

    SoSwitch *top_switch;
    SoSearchAction saSwitch;

    saSwitch.setFind(SoSearchAction::NAME);
    saSwitch.setInterest(SoSearchAction::FIRST);
    saSwitch.setName(name);
    saSwitch.setSearchingAll(TRUE);

    //cerr << "Start searching" << endl;
    saSwitch.apply(selection);
    //cerr << "Stop searching" << endl;

    SoPath *path = saSwitch.getPath();
    if (!path)
        cerr << "switch node not found" << endl;

    top_switch = (SoSwitch *)(path->getTail());

    return top_switch;
}

//======================================================================
//
// Description:
//   Find the appropriate transform node in the scene graph for a
//   given object name.
//
// Use: private
//
//
//======================================================================
SoTransform *InvCoviseViewer::findTransformNode(char *Name)
{
    //print_comment(__LINE__,__FILE__,"in findTransformNode");

    SoTransform *transform;
    const char *name;
    char objName[255];
    SbName string;
    int i, j;

    SoSearchAction saLabel;
    SoPathList listLabel;
    SoLabel *label;

    saLabel.setFind(SoSearchAction::TYPE);
    saLabel.setInterest(SoSearchAction::ALL);
    saLabel.setType(SoLabel::getClassTypeId());
    saLabel.setSearchingAll(TRUE);
    saLabel.apply(selection);

    // get the list of paths
    listLabel = saLabel.getPaths();

    transform = NULL;
    // cycle through the list and find (first) match
    if (listLabel.getLength() != 0)
    {
        for (i = 0; i < listLabel.getLength(); i++)
        {
            label = (SoLabel *)(listLabel[i]->getTail());
            string = label->label.getValue();
            name = string.getString();
            strcpy(objName, name);

            if (strcmp(objName, Name) == 0)
            {

                SoGroup *group = (SoGroup *)(listLabel[i]->getNodeFromTail(1));

                for (j = 0; j < group->getNumChildren(); j++)
                {
                    SoNode *n = group->getChild(j);
                    if (n->isOfType(SoTransform::getClassTypeId()))
                    {
                        transform = (SoTransform *)group->getChild(j);
                        break;
                    }
                }
                break;
            }
        }
    }
    else
    {
        print_comment(__LINE__, __FILE__, "ERROR: findTransformNode : no object with this name found");
        return NULL;
    }

    return transform;
}

//======================================================================
//
// Description:
//   Find the appropriate shape node in the scene graph for a
//   given object name.
//
// Use: private
//
//
//======================================================================
SoShape *InvCoviseViewer::findShapeNode(char *Name)
{
    //print_comment(__LINE__,__FILE__,"in findShapeNode");

    SoShape *shape = NULL;
    const char *name;
    char objName[255];
    SbName string;
    int i, j;

    SoSearchAction saLabel;
    SoPathList listLabel;
    SoLabel *label;

    //  fprintf(stderr,"In findShapeNode");

    saLabel.setFind(SoSearchAction::TYPE);
    saLabel.setInterest(SoSearchAction::ALL);
    saLabel.setType(SoLabel::getClassTypeId());

    saLabel.apply(selection);

    // get the list of paths
    listLabel = saLabel.getPaths();

    // cycle through the list and find (first) match
    if (listLabel.getLength() != 0)
    {
        for (i = 0; i < listLabel.getLength(); i++)
        {
            label = (SoLabel *)(listLabel[i]->getTail());
            string = label->label.getValue();
            name = string.getString();
            strcpy(objName, name);

            if (strcmp(objName, Name) == 0)
            {

                SoGroup *group = (SoGroup *)(listLabel[i]->getNodeFromTail(1));

                for (j = 0; j < group->getNumChildren(); j++)
                {
                    SoNode *n = group->getChild(j);
                    if (n->isOfType(SoShape::getClassTypeId()))
                    {
                        shape = (SoShape *)group->getChild(j);
                        break;
                    }
                }
                break;
            }
        }
    }
    else
    {
        print_comment(__LINE__, __FILE__, "ERROR: findShapeNode : no object with this name found");
        return NULL;
    }

    return shape;
}

//======================================================================
//
// Description:
//	Create an object editor for the currently selected object.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::createObjectEditor()
{
    // not implemented yet
}

//======================================================================
//
// Description:
//	Create a part editor.
//
// Use: private
//
//======================================================================
void
InvCoviseViewer::createPartEditor()
{
    char id[255];
    int maxKey;
    int numEntries;

    if (partEditor_ == NULL)
    {
        partEditor_ = new InvPartEditor;
    }

    partEditor_->setViewer(this);

    // delete all items from list widget
    partEditor_->deleteAllItems();

    // fill list widget
    if (partEditor_)
    {
        numEntries = multiHash.getNumEntries();
        if (numEntries > 0)
        {
            maxKey = multiHash.getMaxKey();
        }
        else
        {
            maxKey = -1;
        }

        //Determine how many parts are neeeded in the
        //Part editor
        int count = 0;
        int i;
        for (i = 0; i <= maxKey; i++)
        {
            if (multiHash[i])
            {
                count++;
            }
        }
        partEditor_->allocPartList(count);
        for (i = 0; i <= maxKey; i++)
        {
            iter = multiHash[i];
            if (iter)
            {
                count++;
                id[0] = '\0';
                strcat(id, partNames[i]);
                partEditor_->addToPartList(id, i);
            }
        }
    }
    //  partEditor->constructorCommon();
    partEditor_->show();
}

void
InvCoviseViewer::createAnnotationEditor(const InvAnnoFlag *af)
{

    if (annoEditor_ == NULL)
    {
        annoEditor_ = new InvAnnotationEditor;
    }

    annoEditor_->setViewer(this);
    annoEditor_->setFlagObject(af);

    annoEditor_->show();
}

//======================================================================
//
// Description:
//	Switch a part on/off.
//
// Use: public
//
//======================================================================
void InvCoviseViewer::switchPart(int key, int tag)
{
    SoSwitch *s = NULL;

    assert(tag == SO_SWITCH_ALL || tag == SO_SWITCH_NONE);

    switchIter = switchHash[key];
    if (switchIter)
    {
        while (switchIter)
        {
            s = switchIter();
            if (s)
            {
                s->whichChild.setValue(tag);
                sendPart(key, tag);
            }
            ++switchIter;
        }
    }
}

//======================================================================
//
// Description:
//    Set reference point for a part of ID "partID"
//
// Use: public
//
//======================================================================
void
InvCoviseViewer::setReferencePoint(int partID)
{

    SoSwitch *s = NULL;
    SoGroup *partRoot = NULL;
    SoNode *node = NULL;
    TimePart *tp = new TimePart(0, partID);

    referIter = referHash[*tp];
    if (referIter)
    {
        partRoot = new SoGroup;
        while (referIter)
        {
            s = referIter();
            if (s)
            {
                node = s->getChild(0);
                if (node->isOfType(SoSeparator::getClassTypeId()))
                {
                    partRoot->addChild((SoSeparator *)node);
                }
                else
                    cerr << "under switch node no separator node!" << endl;
            }
            ++referIter;
        }
    }
    if (partRoot)
    {
        SoGetBoundingBoxAction bba(currentViewer->getViewportRegion());
        bba.apply(partRoot);
        SbBox3f bbox = bba.getBoundingBox();
        refPoint = bbox.getCenter();
        //cerr << "P: " << refPoint[0] << "\t" << refPoint[1] << "\t" << refPoint[2] << endl;
        sendResetScene();
        sendReferencePart(partID);
    }
    delete tp;
}

//======================================================================
//
// Description:
//    Translate parts with regard to the reference part.
//
// Use: public
//
//======================================================================
void InvCoviseViewer::transformScene(int part)
{
    int i, j; //,k;
    int numEntries;
    int maxTime;
    int maxPart;
    SoSwitch *s = NULL;
    SoNode *node = NULL;
    SoSeparator *sep = NULL;
    SoNode *child = NULL;
    SoTransform *t = NULL;
    TimePart tp;
    TimePart temp;

    numEntries = referHash.getNumEntries();
    if (numEntries > 0)
    {
        maxTime = referHash.getMaxTime();
        maxPart = referHash.getMaxPart();
    }
    else
    {
        maxTime = -1;
        maxPart = -1;
    }

    SoGroup **timeRoot = new SoGroup *[maxTime];
    SbVec3f *transVec = new SbVec3f[maxTime];

    for (i = 1; i <= maxTime; i++)
    {
        tp.set(i, part);
        timeRoot[i - 1] = NULL;

        referIter = referHash[tp];
        if (referIter)
        {
            timeRoot[i - 1] = new SoGroup;
            while (referIter)
            {
                s = referIter();
                if (s)
                {
                    node = s->getChild(0);
                    if (node->isOfType(SoSeparator::getClassTypeId()))
                    {
                        timeRoot[i - 1]->addChild((SoSeparator *)node);
                    }
                    else
                        cerr << "under switch node no separator node!" << endl;
                }
                ++referIter;
            }
        }

        if (timeRoot[i - 1])
        {
            SoGetBoundingBoxAction bba(currentViewer->getViewportRegion());
            bba.apply(timeRoot[i - 1]);
            SbBox3f bbox = bba.getBoundingBox();
            SbVec3f seekPoint = bbox.getCenter();
            //cerr << i << " : TP: " << seekPoint[0] << "\t" << seekPoint[1] << "\t" << seekPoint[2] << endl;
            transVec[i - 1].setValue(refPoint[0] - seekPoint[0], refPoint[1] - seekPoint[1], refPoint[2] - seekPoint[2]);

            for (j = 0; j <= maxPart; j++)
            {
                temp.set(i, j);
                referIter = referHash[temp];
                if (referIter)
                {
                    while (referIter)
                    {
                        s = referIter();
                        if (s)
                        {
                            node = s->getChild(0);
                            if (node->isOfType(SoSeparator::getClassTypeId()))
                            {
                                sep = (SoSeparator *)node;
                                child = sep->getChild(1);
                                if (child->isOfType(SoTransform::getClassTypeId()))
                                {
                                    t = (SoTransform *)child;
                                    t->translation.setValue(transVec[i - 1]);
                                }
                                else
                                    cerr << "under separator node second child no transform node!" << endl;
                            }
                        }
                        ++referIter;
                    }
                }
            }
        }
    }
    // free
    delete[] transVec;
    delete[] timeRoot;
}

//======================================================================
//
// Description:
//    Reset the translations.
//
// Use: public
//
//======================================================================
void InvCoviseViewer::resetTransformedScene()
{
    int i, j;
    int numEntries;
    int maxTime;
    int maxPart;
    SoSwitch *s = NULL;
    SoNode *node = NULL;
    SoSeparator *sep = NULL;
    SoNode *child = NULL;
    SoTransform *t = NULL;
    TimePart temp;

    SbVec3f *nullVec = new SbVec3f(0.0, 0.0, 0.0);

    numEntries = referHash.getNumEntries();
    if (numEntries > 0)
    {
        maxTime = referHash.getMaxTime();
        maxPart = referHash.getMaxPart();
    }
    else
    {
        maxTime = -1;
        maxPart = -1;
    }

    for (i = 1; i <= maxTime; i++)
    {
        for (j = 0; j <= maxPart; j++)
        {
            temp.set(i, j);
            referIter = referHash[temp];
            if (referIter)
            {
                while (referIter)
                {
                    s = referIter();
                    if (s)
                    {
                        node = s->getChild(0);
                        if (node->isOfType(SoSeparator::getClassTypeId()))
                        {
                            sep = (SoSeparator *)node;
                            child = sep->getChild(1);
                            if (child->isOfType(SoTransform::getClassTypeId()))
                            {
                                t = (SoTransform *)child;
                                // reset translation
                                t->translation.setValue(*nullVec);
                            }
                            else
                                cerr << "under separator node second child no transform node!" << endl;
                        }
                    }
                    ++referIter;
                }
            }
        }
    }
    sendResetScene();
}

//======================================================================
//
// Description:
//	Create a material editor for the currently selected object.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::createMaterialEditor()
{
    if (materialEditor == NULL)
        materialEditor = new SoXtMaterialEditor;
    materialEditor->show();

    materialEditor->attach(findMaterialForAttach(NULL));
}

//======================================================================
//
// Description:
//	Create a transform editor for the currently selected object
//
// Use: private
//
//======================================================================
void InvCoviseViewer::createTransformSliderSet()
{
    SoPath *editTransformPath;
    SoTransform *editTransform;

    // get path to a transform to edit
    if ((editTransformPath = findTransformForAttach(NULL)) == NULL)
        return;

    // the tail of the path is a transform for us!
    editTransformPath->ref();
    editTransform = (SoTransform *)((SoFullPath *)editTransformPath)->getTail();
    editTransformPath->unref();

    // Nuke the old slider set and get a new one
    if (transformSliderSet == NULL)
        transformSliderSet = new SoXtTransformSliderSet();
    transformSliderSet->setNode(editTransform);
    transformSliderSet->show();
}

//======================================================================
//
// Description:
//      Set fog on/off.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::setFog(SbBool flag)
{
    fogFlag = flag;

    if (fogFlag)
        // purple ?
        environment->fogType.setValue(SoEnvironment::HAZE);
    else
        environment->fogType.setValue(SoEnvironment::NONE);
}

//======================================================================
//
// Description:
//      Set AA-ing on/off.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::setAntialiasing(SbBool flag)
{
    antialiasingFlag = flag;

    if (antialiasingFlag)
    {

#ifdef __linux__
        // on GeForce2 PC graphics hardware devices this seems
        // to be the best compromise (smoothing only)
        currentViewer->setAntialiasing(TRUE, 1);
#else
        // SGI and others (smoothing plus multi pass rendering)
        SoGLRenderAction *ra = currentViewer->getGLRenderAction();
        ra->setPassUpdate(FALSE);
        currentViewer->setGLRenderAction(ra);
        currentViewer->setAntialiasing(FALSE, 3);
#endif
    }
    else
    {
        currentViewer->setAntialiasing(FALSE, 1);
    }
}

//======================================================================
//
// Description:
//      Invokes color editor on ambient lighting color.
//
//======================================================================
void InvCoviseViewer::editAmbientColor()
{
    if (ambientColorEditor == NULL)
    {
        ambientColorEditor = new MyColorEditor;
        ambientColorEditor->setTitle("Ambient Lighting");
        ambientColorEditor->addColorChangedCallback(
            InvCoviseViewer::ambientColorCallback, this);
    }

    // Normalize ambient intensity
    SbColor ambCol;
    ambCol = environment->ambientColor.getValue();
    ambCol *= environment->ambientIntensity.getValue();
    environment->ambientIntensity.setValue(1.0);
    environment->ambientColor.setValue(ambCol);

    ignoreCallback = TRUE;
    ambientColorEditor->setColor(environment->ambientColor.getValue());
    ignoreCallback = FALSE;
    ambientColorEditor->show();
}

//======================================================================
//
// Description:
//  Callback proc invoked by the color editor, this changes the scene's
//  ambient lighting color.
//
//  Use: static, private
//
//
//======================================================================
void InvCoviseViewer::ambientColorCallback(void *userData, const SbColor *color)
{
    InvCoviseViewer *sv = (InvCoviseViewer *)userData;

    if (sv->ignoreCallback)
        return;

    sv->environment->ambientColor.setValue(*color);
}

//======================================================================
//
// Description:
//      Invokes color editor on background color.
//
//======================================================================
void InvCoviseViewer::editBackgroundColor()
{
    if (backgroundColorEditor == NULL)
    {
        backgroundColorEditor = new MyColorEditor;
        backgroundColorEditor->setTitle("Background Color");
        backgroundColorEditor->addColorChangedCallback(
            InvCoviseViewer::backgroundColorCallback, this);
    }
    ignoreCallback = TRUE;
    backgroundColorEditor->setColor(getBackgroundColor());
    ignoreCallback = FALSE;
    backgroundColorEditor->show();
}

//======================================================================
//
// Description:
//  Callback proc invoked by the color editor, this changes the current
//  viewer's background color.
//
//  Use: static, private
//
//
//======================================================================
void InvCoviseViewer::backgroundColorCallback(void *userData, const SbColor *c)
{
    float r, g, b;

    InvCoviseViewer *sv = (InvCoviseViewer *)userData;

    if (sv->ignoreCallback)
        return;

    sv->currentViewer->setBackgroundColor(*c);

    // keep fog color up to date with bkg color
    sv->environment->fogColor.setValue(*c);

    // send new color to slaves
    c->getValue(r, g, b);
    sv->sendBackcolor(r, g, b);
}

//======================================================================
//
// Description:
//	This will remove any cameras under root.
//
// Use: private
//
//======================================================================
void InvCoviseViewer::removeCameras(SoGroup *root)
{
    SoSearchAction sa;
    sa.setType(SoCamera::getClassTypeId());
    sa.setInterest(SoSearchAction::ALL);
    sa.apply(root);

    // remove those cameras!
    SoPathList paths = sa.getPaths();
    for (int i = 0; i < paths.getLength(); i++)
    {
        SoPath *p = paths[i];
        SoCamera *cam = (SoCamera *)p->getNodeFromTail(0);
        SoGroup *group = (SoGroup *)p->getNodeFromTail(1);
        group->removeChild(cam);
    }
}

//======================================================================
//
// Description:
//	Reads the given file and insert the geometry under the selection
//  node. If the node didn't have any children, the viewAll() method is
//  automatically called.
//
// Use: private
//
//======================================================================
SbBool InvCoviseViewer::readFile(const char *filename)
{
    SoInput in;
    if (!in.openFile(filename))
    {

        // display an error dialog
        char str[MAXPATHLEN + 21];
        strcpy(str, "Error opening file: ");
        strcat(str, filename);
        SoXt::createSimpleErrorDialog(mgrWidget, (char *)"File Error Dialog", str);
        return FALSE;
    }

    SbBool hadNoChildren = (selection->getNumChildren() == 0);

    // add nodes under selection, not sceneGraph
    SoNode *node;
    SbBool ok;
    while ((ok = SoDB::read(&in, node)) && (node != NULL))
        selection->addChild(node);

    // display error dialog if there were reading errors
    if (!ok)
    {
        char str[MAXPATHLEN + 21];
        strcpy(str, "Error reading file: ");
        strcat(str, filename);
        SoXt::createSimpleErrorDialog(mgrWidget, (char *)"File Error Dialog", str);
        return FALSE;
    }

    // remove any cameras under selection which were just added
    removeCameras(selection);

    if (hadNoChildren)
    {
        viewAll();
        saveHomePosition();
    }

    return TRUE;
}

//======================================================================
//
// Description:
//	Read environment data. We expect the following nodes:
//
//  Group {
//    Label { "COVISE Renderer Environment v1.0" }
//    Camera {}
//    Environment {}
//    LightGroup {
//      Switch { DirectionalLight }  # 1
//  	...
//      Switch { DirectionalLight }  # 6
//    }
//    DirectionalLight {}   	# optional headlight
//  }
//
// Use: private
SbBool InvCoviseViewer::readEnvFile(const char *filename)
//
//======================================================================
{
    SoInput in;
    if (!in.openFile(filename))
    {
        // display an error dialog
        char str[MAXPATHLEN + 21];
        strcpy(str, "Error opening file: ");
        strcat(str, filename);
        SoXt::createSimpleErrorDialog(mgrWidget, (char *)"File Error Dialog", str);
        return FALSE;
    }

    SoNode *n;
    SoLabel *l = NULL;
    SbBool isValid = FALSE;
    SbBool ok;

    if ((ok = SoDB::read(&in, n)) && n != NULL)
    {
        // we expect a label first
        n->ref();
        if (n->isOfType(SoLabel::getClassTypeId()))
        {
            l = (SoLabel *)n;
            isValid = (strcmp(l->label.getValue().getString(),
                              SV_ENV_LABEL) == 0);
        }
        n->unref();
    }
    else if (!ok)
    {
        // display error dialog if there were reading errors
        char str[MAXPATHLEN + 21];
        strcpy(str, "Error reading file: ");
        strcat(str, filename);
        SoXt::createSimpleErrorDialog(mgrWidget, (char *)"File Error Dialog", str);
        return FALSE;
    }

    // if ok, read the rest.
    if (isValid)
    {
        // Camera
        if (SoDB::read(&in, n) != FALSE && (n != NULL))
        {
            n->ref();
            if (n->isOfType(SoCamera::getClassTypeId()))
            {
                // replace the old camera with the new camera and
                // re-attach the viewer.
                SoCamera *newCamera = (SoCamera *)n;
                SoCamera *oldCamera = getCamera();
                SoSearchAction sa;
                sa.setNode(oldCamera);
                sa.apply(sceneGraph);
                SoFullPath *fullCamPath = (SoFullPath *)sa.getPath();
                if (fullCamPath)
                {
                    SoGroup *parent = (SoGroup *)fullCamPath->getNode(fullCamPath->getLength() - 2);
                    parent->insertChild(newCamera, parent->findChild(oldCamera));
                    setCamera(newCamera);
                    if (parent->findChild(oldCamera) >= 0)
                        parent->removeChild(oldCamera);
                }
#if DEBUG
                else
                {
#ifndef _AIRBUS
                    SoDebugError::post("COVISE Renderer method: readEnvFile",
                                       " cannot find camera in scene graph");
#else
                    SoDebugError::post("NS3D Renderer method: readEnvFile",
                                       " cannot find camera in scene graph");
#endif
#endif
                }
                n->unref();
            }
            // Environment
            if (SoDB::read(&in, n) != FALSE && (n != NULL))
            {
                n->ref();
                if (n->isOfType(SoEnvironment::getClassTypeId()))
                {
                    lightsCameraEnvironment->replaceChild(environment, n);
                    environment = (SoEnvironment *)n;
                }
                n->unref();
            }
            // Light group
            if (SoDB::read(&in, n) != FALSE && (n != NULL))
            {
                n->ref();
                if (n->isOfType(SoGroup::getClassTypeId()))
                {

                    // remove all of the existing lights
                    int i;
                    for (i = lightDataList.getLength(); i > 0; i--)
                        removeLight((InvLightData *)lightDataList[i - 1]);

                    lightsCameraEnvironment->replaceChild(lightGroup, n);
                    lightGroup = (SoGroup *)n;

                    // This was busted. It was looking for a light as child 0,
                    // but the scale and scaleInverse made it think no light was
                    // there. So now, we do this right...
                    // We'll just check for the light as any old child.
                    // This way it's okay to add a translation node under that
                    // switch too, so we can translate the manips as well.
                    // This allows as to place the directional light manips.
                    for (i = 0; i < lightGroup->getNumChildren(); i++)
                    {
                        SoNode *node = lightGroup->getChild(i);
                        if (node->isOfType(SoSwitch::getClassTypeId()))
                        {
                            SoSwitch *sw = (SoSwitch *)node;
                            SbBool addedIt = FALSE;
                            for (int j = 0;
                                 addedIt == FALSE && j < sw->getNumChildren();
                                 j++)
                            {
                                node = sw->getChild(j);
                                if (node->isOfType(SoLight::getClassTypeId()))
                                {
                                    addLightEntry((SoLight *)node, sw);
                                    addedIt = TRUE;
                                }
                            }
                        }
                    }
                }
                n->unref();
            }
            // Headlight (optional) - if not there, turn headlight off
            if (SoDB::read(&in, n) != FALSE && (n != NULL))
            {
                n->ref();
                if (n->isOfType(SoDirectionalLight::getClassTypeId()))
                {
                    SoDirectionalLight *headlight = getHeadlight();
                    SoDirectionalLight *newLight = (SoDirectionalLight *)n;
                    if (headlight != NULL)
                    {
                        headlight->intensity.setValue(newLight->intensity.getValue());
                        headlight->color.setValue(newLight->color.getValue());
                        headlight->direction.setValue(newLight->direction.getValue());
                        setHeadlight(TRUE);
                    }
                }
                n->unref();
            }
            else
                setHeadlight(FALSE);
        }
        else
        {
#ifndef _AIRBUS
            fprintf(stderr, "COVISE Renderer ERROR: Sorry, the environment file is not formatted correctly\n");
#else
        fprintf(stderr, "NS3D Renderer ERROR: Sorry, the environment file is not formatted correctly\n");
#endif
        }

        return TRUE;
    }

    //======================================================================
    //
    // Description:
    //  Adds the given geometry under the selection
    //  node.
    //
    // Use: public
    //
    //======================================================================
    void InvCoviseViewer::addToSceneGraph(SoGroup * child, const char *name, SoGroup *root)
    {

        newData();

        // add nodes under 'root', not sceneGraph
        if (root != NULL)
        {
            root->addChild(child);
        }
        else
        {
            selection->addChild(child);
        }

        addToObjectList((char *)name);
    }

    //======================================================================
    //
    // Description:
    //  Adds the given geometry under the selection
    //  node. If the node didn't have any children, the viewAll() method is
    //  automatically called.
    //
    // Use: public
    //
    //======================================================================
    void InvCoviseViewer::removeFromSceneGraph(SoGroup * delroot, const char *name)
    {

        selection->deselectAll();
        selection->removeChild(delroot);

        // now that we have removed the subgroup from our
        // selection we delete every child node under delroot

        removeFromObjectList(name);
    }

    //======================================================================
    //
    // Description:
    //  Replaces the given geometry under the selection
    //  node. Camera position does not change
    //
    // Use: public
    //
    //======================================================================
    void InvCoviseViewer::replaceSceneGraph(SoNode * root)
    {

        // remove old scene
        deleteScene();

        // add new nodes under selection, not sceneGraph
        selection->addChild(root);

        // if you want to view the whole scene when new data arrives
        // uncomment the four following lines
        //   SbBool hadNoChildren = (selection->getNumChildren() == 0);
        //   if (hadNoChildren) {
        //	   viewAll();
        //	   saveHomePosition(); }
    }

    //======================================================================
    //
    // Description:
    //  Adds the given texture to the texture list
    //
    // Use: public
    //
    //======================================================================
    void InvCoviseViewer::addToTextureList(SoTexture2 * tex)
    {

        // add texture to list
        if (textureList && tex)
        {
            textureList->append(tex);
        }
    }

    //======================================================================
    //
    // Description:
    //  Removes the given texture from the texture list
    //
    // Use: public
    //
    //======================================================================
    void InvCoviseViewer::removeFromTextureList(SoTexture2 * tex)
    {
        int index;

        if (textureList && tex)
        {
            index = textureList->find(tex);
            if (index > -1)
            {
                textureList->remove(index);
            }
        }
    }

    //======================================================================
    //
    // Description:
    //  Replaces the given geometry under the selection
    //  node. Camera position is set so that the whole object can be seen
    //
    // Use: public
    //
    //======================================================================
    void
        InvCoviseViewer::setSceneGraph(SoNode * root)
    {

        deleteScene();

        // add nodes under selection, not sceneGraph
        selection->addChild(root);

        // remove any cameras under selection which were just added
        // should be no cameras
        //    removeCameras(selection);

        viewAll();
        saveHomePosition();
    }

    //======================================================================
    //
    // Description:
    //	This routine is called to get a file name. Either a motif
    //  dialog or the showcase gizmo are used.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::getFileName()
    {
        // use a motif file selection dialog
        if (fileDialog == NULL)
        {
            Arg args[5];
            int n = 0;

            // unmanage when ok/cancel are pressed
            XtSetArg(args[n], XmNautoUnmanage, TRUE);
            n++;
            fileDialog = XmCreateFileSelectionDialog(
                XtParent(mgrWidget), (char *)"File Dialog", args, n);

            XtAddCallback(fileDialog, XmNokCallback,
                          (XtCallbackProc)InvCoviseViewer::fileDialogCB,
                          (XtPointer) this);
        }

        // manage the dialog
        XtManageChild(fileDialog);
    }

    //======================================================================
    //
    // Description:
    //	Motif file dialog callback.
    //
    //
    //======================================================================
    void InvCoviseViewer::fileDialogCB(Widget, InvCoviseViewer * sv,
                                       XmFileSelectionBoxCallbackStruct * data)
    {
        // Get the file name
        char *filename;
        XmStringGetLtoR(data->value,
                        (XmStringCharSet)XmSTRING_DEFAULT_CHARSET, &filename);

        // Use that file
        sv->doFileIO(filename);

        XtFree(filename);
    }

    //======================================================================
    //
    // Description:
    //	detach everything and nuke the existing scene.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::deleteScene()
    {
        // deselect everything (also detach manips)
        selection->deselectAll();

        // temporaly remove the light manips
        removeAttachedLightManipGeometry();

        // remove the geometry under the selection node
        for (int i = selection->getNumChildren(); i > 0; i--)
            selection->removeChild(i - 1);

        // add the light manips back in
        addAttachedLightManipGeometry();
    }

    //======================================================================
    //
    // Description:
    //	Read/Write to the given file name, given the current file mode.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::doFileIO(const char *file)
    {
        SbBool okFile = FALSE;

        switch (fileMode)
        {

        case SV_FILE_SAVE_AS:
            okFile = writeFile(file);
            break;

        case SV_FILE_READ_ENV:
            readEnvFile(file);
            break;

        case SV_FILE_SAVE_ENV:
        {

            // Run through the lights. If any light is a directionalLightManip,
            // transfer its translation into the InvLightData, since that
            // info will not write out.
            for (int i = 0; i < lightDataList.getLength(); i++)
                transferDirectionalLightLocation((InvLightData *)lightDataList[i]);

            writeEnvFile(file);
        }
        break;

        default:
            fprintf(stderr, "Wrong file mode %d passed!\n", fileMode);
            return;
        }

        // save the new file name so we can simply use "Save" instead of
        // "Save As" next time around.
        if (fileMode == SV_FILE_SAVE_AS)
        {

            // save the current file name
            delete fileName;
            if (okFile && file != NULL)
                fileName = strdup(file);
            else
                fileName = NULL;
        }

        // enable/disable cmd key shortcuts and menu items
        updateCommandAvailability();
    }

    //======================================================================
    //
    // Description:
    //	Saves the scene to the current file.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::save()
    {
        if (fileName != NULL)
        {
            SbBool ok = writeFile(fileName);
            if (!ok)
            {
                delete fileName;
                fileName = NULL;
            }
        }
        else
        {
            fileMode = SV_FILE_SAVE_AS;
            getFileName();
        }
    }

    //======================================================================
    //
    // Description:
    //	Removes the attached light manips geometry from the scene. This
    //  is used for file writting,...
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::removeAttachedLightManipGeometry()
    {
        for (int i = 0; i < lightDataList.getLength(); i++)
        {

            InvLightData *data = (InvLightData *)lightDataList[i];

            // We'll be putting everything back later, so make a note of this...
            data->shouldBeManip = data->isManip;

            if (data->isManip == TRUE)
                editLight(data, FALSE);
        }
    }

    //======================================================================
    //
    // Description:
    //	Add the attached light manips geometry back into the scene. This
    // is called after the geometry has been temporaly revomed (used for file
    // writting).
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::addAttachedLightManipGeometry()
    {
        for (int i = 0; i < lightDataList.getLength(); i++)
        {

            InvLightData *data = (InvLightData *)lightDataList[i];

            if (data->isManip != data->shouldBeManip)
                editLight(data, data->shouldBeManip);
        }
    }

    //======================================================================
    //
    // Description:
    //	Write the nodes under the selection node to the given file name.
    //
    // Use: private
    //
    //======================================================================
    SbBool InvCoviseViewer::writeFile(const char *filename)
    {
        SoWriteAction wa;

        if (!wa.getOutput()->openFile(filename))
        {

            // display an error dialog
            char str[MAXPATHLEN + 21];
            strcpy(str, "Error creating file: ");
            strcat(str, filename);
            SoXt::createSimpleErrorDialog(mgrWidget, (char *)"File Error Dialog", str);

            return FALSE;
        }

        // temporarily replace all manips with regular transform nodes.
        removeManips();

        // Do the same for all the light manips
        removeAttachedLightManipGeometry();

        // write out all the children of the selection node
        for (int i = 0; i < selection->getNumChildren(); i++)
            wa.apply(selection->getChild(i));
        wa.getOutput()->closeFile();

        // Now put the manips back in the scene graph.
        restoreManips();

        // put the light manips back in the scene graph.
        addAttachedLightManipGeometry();

        return TRUE;
    }

    //======================================================================
    //
    // Description:
    //	Write the Enviroment nodes (camera and lights) to the given
    //  file name.
    //
    // Use: private
    SbBool InvCoviseViewer::writeEnvFile(const char *filename)
    //
    //======================================================================
    {
        SoWriteAction wa;

        if (!wa.getOutput()->openFile(filename))
        {

            // display an error dialog
            char str[MAXPATHLEN + 21];
            strcpy(str, "Error creating file: ");
            strcat(str, filename);
            SoXt::createSimpleErrorDialog(mgrWidget, (char *)"File Error Dialog", str);

            return FALSE;
        }

        // write out the environment including the headlight
        wa.apply(envLabel);
        wa.apply(getCamera());
        wa.apply(environment);
        wa.apply(lightGroup);
        if (isHeadlight())
            wa.apply(getHeadlight());

        wa.getOutput()->closeFile();

        return TRUE;
    }

    //======================================================================
    //
    // Description:
    //	Send VRML camera msg to VRML_Renderer
    //  file name.
    //
    // Use: private
    //
    //======================================================================
    SbBool InvCoviseViewer::sendVRMLCamera()
    {
        SoWriteAction wa;
        char buffer[400];

        wa.getOutput()->setBuffer(buffer, 400, NULL);
        SoCamera *vwrCamera = getCamera();
        if (vwrCamera != NULL)
        {
            wa.apply(getCamera());
            rm_sendVRMLCamera(buffer);
            return TRUE;
        }

        return FALSE;
    }

    //======================================================================
    //
    // Description:
    //	Print the scene using a custom print dialog.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::print()
    {
        if (printDialog == NULL)
        {
            printDialog = new SoXtPrintDialog;
            printDialog->setTitle("SceneViewer Printing");
            printDialog->setBeforePrintCallback(
                InvCoviseViewer::beforePrintCallback, (void *)this);
            printDialog->setAfterPrintCallback(
                InvCoviseViewer::afterPrintCallback, (void *)this);
        }

        //
        // Send the render area size and scene graph to the print dialog
        //
        Widget widget = getRenderAreaWidget();
        if (widget != NULL)
        {
            Arg args[2];
            int n = 0;
            SbVec2s sz;
            XtSetArg(args[n], XtNwidth, &sz[0]);
            n++;
            XtSetArg(args[n], XtNheight, &sz[1]);
            n++;
            XtGetValues(widget, args, n);
            printDialog->setPrintSize(sz);
        }

        printDialog->show();
    }

    //======================================================================
    //
    // Description:
    //	Print the scene to an rgb file called "snap.rgb"
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::snap(const char *filename)
    {
        Widget widget = getRenderAreaWidget();
#if 1
        if (widget != NULL) // works for us
        {
            Arg args[2];
            int n = 0;
            SbVec2s sz;
            XtSetArg(args[n], XtNwidth, &sz[0]);
            n++;
            XtSetArg(args[n], XtNheight, &sz[1]);
            n++;
            XtGetValues(widget, args, n);

            snap(filename, sz[0], sz[1]);
        }
#else
    currentViewer->render();
    x11SnapTIFF(widget, filename);
#endif
    }

    //======================================================================
    //
    // Description:
    //	Print the scene to an rgb file
    //
    // Use: private
    //
    //======================================================================
    void
    InvCoviseViewer::snap(const char *filename, int sx, int sy)
    {
        // use OIV 's offscreen renderer

        sx = coCoviseConfig::getInt("width", "Renderer.Snap", sx);
        sy = coCoviseConfig::getInt("height", "Renderer.Snap", sy);
        int numpasses = coCoviseConfig::getInt("passes", "Renderer.Snap", 5);
        std::string format = coCoviseConfig::getEntry("format", "Renderer.Snap", "tiff");

        SbViewportRegion viewport(sx, sy);

        //InvOffscreenRenderer printer(currentViewer->getGLRenderAction());
        InvOffscreenRenderer printer(viewport);
        printer.setViewportRegion(viewport);
        viewport = printer.getViewportRegion();

        //printer.setBackgroundColor( SbColor(1.0,1.0,1.0) );
        printer.setBackgroundColor(getBackgroundColor());
        SoGLRenderAction *GLRendAct = printer.getGLRenderAction();
        if (GLRendAct)
        {
            GLRendAct->setNumPasses(numpasses);
            printer.setGLRenderAction(GLRendAct);
        }
        printer.setComponents(InvOffscreenRenderer::RGB);

        if (printer.render(sceneGraph) == FALSE)
        {
            if (GLRendAct)
            {
                GLRendAct->setNumPasses(1);
                GLRendAct->setSmoothing(False);
                GLRendAct->apply(sceneGraph);
            }
            SoXt::createSimpleErrorDialog(mgrWidget, (char *)"Snap Error Dialog",
                                          (char *)"problem with offscreen rendering see console output");
            return;
        }

        if (format == "rgb")
            printer.writeToRGB(filename);
        else
            printer.writeToTiff(filename);

        if (GLRendAct)
        {
            GLRendAct->setNumPasses(1);
            GLRendAct->setSmoothing(False);
        }
    }

    //=======================================================================
    //
    // Description:
    //	Temporarily remove manips from the scene.
    // Restore them with a call to restoreManips().
    //
    // Use: private
    //
    //=======================================================================
    void InvCoviseViewer::removeManips()
    {
        // temporarily replace all manips with regular transform nodes.
        for (int m = 0; m < maniplist->getLength(); m++)
        {
            SoTransformManip *manip = (SoTransformManip *)maniplist->getManip(m);
            SoPath *xfPath = maniplist->getXfPath(m);
            manip->replaceManip(xfPath, NULL);
        }
    }

    //=======================================================================
    //
    // Description:
    //	Restore manips that were removed with removeManips().
    //
    // Use: private
    //
    //=======================================================================
    void InvCoviseViewer::restoreManips()
    {
        // Now put the manips back in the scene graph.
        for (int m = 0; m < maniplist->getLength(); m++)
        {
            SoTransformManip *manip = (SoTransformManip *)maniplist->getManip(m);
            SoPath *xfPath = maniplist->getXfPath(m);
            manip->replaceNode(xfPath);
        }
    }

    //======================================================================
    //
    // Description:
    //	Temporarily remove highlighting and manips from the scene. They
    // will all be restored after the printing is done.
    //
    // Use: private, static
    //
    //======================================================================
    void InvCoviseViewer::beforePrintCallback(void *uData, SoXtPrintDialog *)
    {
        InvCoviseViewer *sv = (InvCoviseViewer *)uData;

        // temporarily replace all manips with regular transforms.
        sv->removeManips();

        // Do the same for all the light manips
        sv->removeAttachedLightManipGeometry();

        // because the  viewer we use is the examiner viewer, turn the
        // feedback axis off while we print
        InvExaminerViewer *exam = (InvExaminerViewer *)sv->currentViewer;
        sv->feedbackShown = exam->isFeedbackVisible();
        exam->setFeedbackVisibility(FALSE);

        // set the scene to print
        sv->printDialog->setSceneGraph(sv->sceneGraph);
    }

    //======================================================================
    //
    // Description:
    //	Called after printing is done. Add the manips back into the
    // scene and restore the highliting style.
    //
    // Use: private, static
    //
    //======================================================================
    void InvCoviseViewer::afterPrintCallback(void *uData, SoXtPrintDialog *)
    {
        InvCoviseViewer *sv = (InvCoviseViewer *)uData;

        // put the manips back in the scene graph.
        sv->restoreManips();

        // put the light manips back in the scene graph.
        sv->addAttachedLightManipGeometry();

        // restor the examiner feedback
        InvExaminerViewer *exam = (InvExaminerViewer *)sv->currentViewer;
        exam->setFeedbackVisibility(sv->feedbackShown);
    }

    //======================================================================
    //
    // Description:
    //	Send a message to UI for showing help
    //
    // Use: private, static
    //
    //
    //======================================================================

    static void showHelp(const char *helpFile)
    {
        InvCommunication().sendShowHelpMessage(helpFile);
    }

    //======================================================================
    //
    // Description:
    //	Static routine for  processing topbar menu events.
    //  When the menu is created, it stores pointer to the Renderer
    //  in the client_data, so that we can tell which Renderer needs
    //  the event.
    //
    // Use: private, static
    //
    //
    //======================================================================
    void InvCoviseViewer::processTopbarEvent(
        Widget, // Which widget?  I don't care
        InvCoviseViewerData * data, // Pointer to button/Renderer
        XmAnyCallbackStruct * cb) // X garbage
    {
        InvCoviseViewer *sv = data->classPt;
        Time eventTime = cb->event->xbutton.time;

        switch (data->id)
        {

        //
        // ONLINE HELP
        // 30.01.2001
        //
        case SV_HELP_1:
        {
            showHelp("/usersguide/renderer/index.html");
            break;
        }

        case SV_HELP_2:
        {
            showHelp("/usersguide/index.html");
            break;
        }

        //
        // File
        //

        case SV_FILE_SAVE_AS:
        case SV_FILE_READ_ENV:
        case SV_FILE_SAVE_ENV:
            sv->fileMode = data->id;
            sv->getFileName();
            break;

        case SV_FILE_COPY:
            sv->removeManips();
            sv->clipboard->copy((SoPathList *)sv->selection->getList(), eventTime);
            sv->restoreManips();
            break;

        case SV_FILE_SAVE:
            sv->save();
            break;

        case SV_FILE_PRINT:
            // works for us
            sv->print();
            break;

        case SV_FILE_SNAP:
            //#       ifdef __sgi
            sv->snap("snap.tiff");
            //#       else
            // display an error dialog
            // 	SoXt::createSimpleErrorDialog(sv->mgrWidget, "Snap Error Dialog",
            //                                      "SNAP not available on this platform");
            // #       endif

            break;

        case SV_FILE_SNAP_ALL:
            if (sv->mySequencer)
                sv->mySequencer->snap();
            break;

        case SV_FILE_RESIZE_PAL:

            extern InvRenderManager *rm;
            rm->setSize(0, 0);
            break;
        //
        // Viewing
        //

        case SV_VIEW_PICK:
            sv->setViewing(!sv->isViewing());
            sv->sendViewing(sv->isViewing());
            break;

        case SV_VIEW_USER:
            sv->userModeFlag = !sv->userModeFlag;
            if (sv->userModeFlag)
            {
                sv->currentViewer->setEventCallback(sv->userModeCB, sv->userModedata);
                cerr << "InvCoviseViewer::processTopbarEvent(..) setEventCallback userModeCB" << endl;
            }
            else
            {
                sv->currentViewer->setEventCallback(NULL, NULL);
                cerr << "InvCoviseViewer::processTopbarEvent(..) UNSET CB" << endl;
            }
            break;

        case SV_VIEW_SELECTION:
            sv->viewSelection();
            break;

        case SV_VIEW_SCREEN_TRANSPARENCY:
            sv->setTransparencyType(SoGLRenderAction::SCREEN_DOOR);
            sv->sendTransparency(SoGLRenderAction::SCREEN_DOOR);
            break;
        case SV_VIEW_BLEND_TRANSPARENCY:
            sv->setTransparencyType(SoGLRenderAction::BLEND);
            sv->sendTransparency(SoGLRenderAction::BLEND);
            break;
        case SV_VIEW_DELAY_BLEND_TRANSPARENCY:
            sv->setTransparencyType(SoGLRenderAction::DELAYED_BLEND);
            sv->sendTransparency(SoGLRenderAction::DELAYED_BLEND);
            break;
        case SV_VIEW_SORT_BLEND_TRANSPARENCY:
            sv->setTransparencyType(SoGLRenderAction::SORTED_OBJECT_BLEND);
            sv->sendTransparency(SoGLRenderAction::SORTED_OBJECT_BLEND);
            break;

        case SV_VIEW_FOG:
            sv->setFog(!sv->fogFlag);
            sv->sendFog(sv->fogFlag);
            break;
        case SV_VIEW_ANTIALIASING:
            sv->setAntialiasing(!sv->antialiasingFlag);
            sv->sendAntialiasing(sv->antialiasingFlag);
            break;
        case SV_VIEW_BKG_COLOR:
            sv->editBackgroundColor();
            break;
        case SV_VIEW_AXIS:
            if (sv->axis_state == CO_ON)
            {
                sv->axis_state = CO_OFF;
                sv->axis_switch->whichChild.setValue(SO_SWITCH_NONE);
                sv->sendAxis(CO_OFF);
            }
            else
            {
                sv->axis_state = CO_ON;
                sv->axis_switch->whichChild.setValue(0);
                ;
                sv->sendAxis(CO_ON);
            }
            break;
        case SV_VIEW_CLIPPING:
            if (sv->clipState == CO_ON)
            {
                sv->clipState = CO_OFF;
                sv->clipSwitch->whichChild.setValue(SO_SWITCH_NONE);
                if (sv->clippingPlaneEditor)
                    sv->clippingPlaneEditor->hide();
                sv->sendClippingPlane(CO_OFF, sv->eqn);
            }
            else
            {
                sv->clipState = CO_ON;
                sv->clipSwitch->whichChild.setValue(0);
                sv->editClippingPlane();
                sv->sendClippingPlane(CO_ON, sv->eqn);
            }
            break;

        //
        // Editors
        //

        case SV_EDITOR_MATERIAL:
            sv->createMaterialEditor();
            break;

        case SV_EDITOR_TRANSFORM:
            sv->createTransformSliderSet();
            break;

        case SV_EDITOR_COLOR:
            sv->createColorEditor();
            break;

        case SV_EDITOR_PARTS:
            sv->createPartEditor();
            break;
        // toogle interactor state
        case SV_EDITOR_SNAPH:
            sv->toggleHandleState();
            break;
        case SV_EDITOR_FREEH:
            sv->toggleHandleState();
            break;

        //
        // Manips
        //

        case SV_MANIP_TRACKBALL:
            sv->highlightRA->setVisible(FALSE); // highlight visible when no manip
            sv->curManip = (sv->curManip == SV_TRACKBALL) ? SV_NONE : SV_TRACKBALL;
            if (sv->curManipReplaces)
                sv->replaceAllManips(sv->curManip);
            break;

        case SV_MANIP_HANDLEBOX:
            sv->highlightRA->setVisible(FALSE); // highlight visible when no manip
            sv->curManip = (sv->curManip == SV_HANDLEBOX) ? SV_NONE : SV_HANDLEBOX;
            if (sv->curManipReplaces)
                sv->replaceAllManips(sv->curManip);
            break;

        case SV_MANIP_JACK:
            sv->highlightRA->setVisible(FALSE); // highlight visible when no manip
            sv->curManip = (sv->curManip == SV_JACK) ? SV_NONE : SV_JACK;
            if (sv->curManipReplaces)
                sv->replaceAllManips(sv->curManip);
            break;

        case SV_MANIP_CENTERBALL:
            sv->highlightRA->setVisible(FALSE); // highlight visible when no manip
            sv->curManip = (sv->curManip == SV_CENTERBALL) ? SV_NONE : SV_CENTERBALL;
            if (sv->curManipReplaces)
                sv->replaceAllManips(sv->curManip);
            break;

        case SV_MANIP_XFBOX:
            sv->highlightRA->setVisible(FALSE); // highlight visible when no manip
            sv->curManip = (sv->curManip == SV_XFBOX) ? SV_NONE : SV_XFBOX;
            if (sv->curManipReplaces)
                sv->replaceAllManips(sv->curManip);
            break;

        case SV_MANIP_TABBOX:
            sv->highlightRA->setVisible(FALSE); // highlight visible when no manip
            sv->curManip = (sv->curManip == SV_TABBOX) ? SV_NONE : SV_TABBOX;
            if (sv->curManipReplaces)
                sv->replaceAllManips(sv->curManip);
            break;

        case SV_MANIP_NONE:
            sv->highlightRA->setVisible(TRUE); // highlight visible when no manip
            sv->curManip = SV_NONE;
            if (sv->curManipReplaces)
                sv->detachManipFromAll();
            break;
        case SV_MANIP_REPLACE_ALL:
            // Toggle the value of 'curManipReplaces'
            sv->curManipReplaces = (sv->curManipReplaces == TRUE) ? FALSE : TRUE;

            if (sv->curManipReplaces)
                sv->replaceAllManips(sv->curManip);
            break;

        //
        // Lights
        //

        case SV_LIGHT_MODEL:
        {
            int num;
            //int n;
            SoTexture2 *tex;
            if (sv->lightmodel_state == 0)
            {
                sv->lightmodel_state = 1;
                sv->currentViewer->setLightModelState(1);
                sv->lightmodel->model = SoLightModel::PHONG;
                sv->sendLightMode(SoLightModel::PHONG);
                if (sv->textureList)
                {
                    num = sv->textureList->getLength();
                    for (int n = 0; n < num; n++)
                    {
                        tex = (SoTexture2 *)(*(sv->textureList))[n];
                        tex->model.setValue(SoTexture2::MODULATE);
                    }
                }
            }
            else
            {
                sv->lightmodel_state = 0;
                sv->currentViewer->setLightModelState(0);
                sv->lightmodel->model = SoLightModel::BASE_COLOR;
                sv->sendLightMode(SoLightModel::BASE_COLOR);
                if (sv->textureList)
                {
                    num = sv->textureList->getLength();
                    for (int n = 0; n < num; n++)
                    {
                        tex = (SoTexture2 *)(*(sv->textureList))[n];
                        tex->model.setValue(SoTexture2::DECAL);
                    }
                }
            }
        }
        break;

        case SV_LIGHT_AMBIENT_EDIT:
            sv->editAmbientColor();
            break;
        case SV_LIGHT_ADD_DIRECT:
            sv->addLight(new SoDirectionalLight);
            break;
        case SV_LIGHT_ADD_POINT:
            sv->addLight(new SoPointLight);
            break;
        case SV_LIGHT_ADD_SPOT:
        {
            // Set the dropOffRate to be non-zero, or it will always work
            // like a point light.
            SoSpotLight *newSpot = new SoSpotLight;
            newSpot->dropOffRate = .01;
            sv->addLight(newSpot);
        }
        break;

        case SV_LIGHT_TURN_ON:
        case SV_LIGHT_TURN_OFF:
        {
            SbBool onFlag = (data->id == SV_LIGHT_TURN_ON);
            for (int i = 0; i < sv->lightDataList.getLength(); i++)
                sv->turnLightOnOff((InvLightData *)sv->lightDataList[i], onFlag);
            sv->turnLightOnOff(sv->headlightData, onFlag);
        }
        break;
        case SV_LIGHT_SHOW_ALL:
        case SV_LIGHT_HIDE_ALL:
        {
            SbBool onFlag = (data->id == SV_LIGHT_SHOW_ALL);
            for (int i = 0; i < sv->lightDataList.getLength(); i++)
                sv->editLight((InvLightData *)sv->lightDataList[i], onFlag);
        }
        break;

        case SV_SYNC_LOOSE:
            sv->sync_flag = SYNC_LOOSE;
            sv->setSyncMode(sv->sync_flag);
            sv->sendSyncMode();
            break;

        case SV_SYNC_MA_SL:
            sv->sync_flag = SYNC_SYNC;
            sv->setSyncMode(sv->sync_flag);
            sv->sendSyncMode();
            break;

        case SV_SYNC_TIGHT:
            sv->sync_flag = SYNC_TIGHT;
            sv->setSyncMode(sv->sync_flag);
            sv->sendSyncMode();
            break;

            /*
                 case SV_COLORMAP_BOTTOM_LEFT:

                     //cerr << "SV_COLORMAP_BOTTOM_LEFT" << endl;
                     sv->cmapPosition = COLORMAP_BOTTOM_LEFT;
                     sv->cmap_x_0 = -1.0;
                sv->cmap_y_0 = -0.9;
                sv->cmap_size=  0.7;
                     sv->colormap_manager->updateColormaps(sv->currentViewer,sv->cmap_x_0,sv->cmap_y_0,sv->cmap_size);
                break;

            case SV_COLORMAP_TOP_LEFT:

            //cerr << "SV_COLORMAP_TOP_LEFT" << endl;
            sv->cmapPosition = COLORMAP_TOP_LEFT;
            sv->cmap_x_0 = -1.0;
            sv->cmap_y_0 =  0.2;
            sv->cmap_size=  0.7;
            sv->colormap_manager->updateColormaps(sv->currentViewer,sv->cmap_x_0,sv->cmap_y_0,sv->cmap_size);
            break;

            case SV_COLORMAP_TOP_RIGHT:

            //cerr << "SV_COLORMAP_TOP_RIGHT" << endl;
            sv->cmapPosition = COLORMAP_TOP_RIGHT;
            sv->cmap_x_0 =  0.7;
            sv->cmap_y_0 =  0.2;
            sv->cmap_size=  0.7;
            sv->colormap_manager->updateColormaps(sv->currentViewer,sv->cmap_x_0,sv->cmap_y_0,sv->cmap_size);
            break;

            case SV_COLORMAP_BOTTOM_RIGHT:

            //cerr << "SV_COLORMAP_BOTTOM_RIGHT" << endl;
            sv->cmapPosition = COLORMAP_BOTTOM_RIGHT;
            sv->cmap_x_0 =  0.7;
            sv->cmap_y_0 = -0.9;
            sv->cmap_size=  0.7;
            sv->colormap_manager->updateColormaps(sv->currentViewer,sv->cmap_x_0,sv->cmap_y_0,sv->cmap_size);
            break;
            */

        } // endswitch( topbar button )
    }

    //======================================================================
    //
    // Description:
    //	Adds the given light to the scene and to the menu.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::addLight(SoLight * light)
    {
        // create the switch and light node and add it to the scene
        SoSwitch *lightSwitch = new SoSwitch;
        lightGroup->addChild(lightSwitch);
        lightSwitch->addChild(light);
        SWITCH_LIGHT_ON(lightSwitch);

        // add the light entry for the new light
        InvLightData *data = addLightEntry(light, lightSwitch);

        //
        // Try to come up with some meaningfull default position base
        // of the current camera view volume.
        //
        SoCamera *vwrCamera = getCamera(); // don't cache this in the class
        SbViewVolume vv = vwrCamera->getViewVolume(0.0);
        SbVec3f forward = -vv.zVector();
        SbVec3f center = vwrCamera->position.getValue() + forward * (vwrCamera->nearDistance.getValue() + vwrCamera->farDistance.getValue()) / 2.0f;
        SbVec3f position(vv.ulf + forward * vv.nearToFar * .25f);
        //XXX ??? XXX
        //XXX this algorithm should be replaced. Perhaps instead of using
        //XXX 'forward' we could go a little up and to the left?
        //XXX ??? XXX

        if (data->type == SoDirectionalLight::getClassTypeId())
        {
            SoDirectionalLight *myLight = (SoDirectionalLight *)data->light;
            // the position of the light can't be given to the light itself.
            // So we use the translation and translation inverse to
            // get it to go where we want.
            data->translation->translation = position;
            data->translationInverse->translation = -position;
            myLight->direction = center - position;
        }
        else
        {
            // The data->scale will influence the position we set.
            // So we need to prepare for this. Note, it's not a prolem for
            // directional lights since they use the translation node,
            // which is outside the scale and scaleInverse grouping
            SbVec3f invrs = data->scaleInverse->scaleFactor.getValue();
            SbVec3f scaledLoc = position;
            scaledLoc *= invrs[0];

            if (data->type == SoPointLight::getClassTypeId())
            {
                SoPointLight *myLight = (SoPointLight *)data->light;
                myLight->location = scaledLoc;
                // no direction for this light
            }
            else if (data->type == SoSpotLight::getClassTypeId())
            {
                SoSpotLight *myLight = (SoSpotLight *)data->light;
                myLight->location = scaledLoc;
                myLight->direction = center - position;
            }
        }
    }

    //======================================================================
    //
    // Description:
    //	Creates and append the light data struct, and adds a menu entry
    //  for the light.
    //
    // Use: private
    //
    //
    //======================================================================
    InvLightData *InvCoviseViewer::addLightEntry(SoLight * light, SoSwitch * lightSwitch)
    {
        //
        // create the light data
        //

        InvLightData *data = new InvLightData;
        lightDataList.append(data);

        light->ref();
        data->light = light;

        data->lightSwitch = lightSwitch;

        // Try and find the scale, scaleInverse, translation, and
        // translationInverse.
        data->scale = NULL;
        data->scaleInverse = NULL;
        data->translation = NULL;
        data->translationInverse = NULL;
        SbBool gotLight = FALSE;
        for (int i = 0; i < lightSwitch->getNumChildren(); i++)
        {
            SoNode *n = lightSwitch->getChild(i);
            if (n == light)
                gotLight = TRUE;
            else if (n->isOfType(SoScale::getClassTypeId()))
            {
                if (data->scale == NULL && gotLight == FALSE)
                    data->scale = (SoScale *)n;
                else if (data->scaleInverse == NULL && gotLight == TRUE)
                    data->scaleInverse = (SoScale *)n;
            }
            else if (n->isOfType(SoTranslation::getClassTypeId()))
            {
                if (data->translation == NULL && gotLight == FALSE)
                    data->translation = (SoTranslation *)n;
                else if (data->translationInverse == NULL && gotLight == TRUE)
                    data->translationInverse = (SoTranslation *)n;
            }
        }

        // Now install any missing nodes...
        if (data->scale == NULL)
        {
            data->scale = new SoScale;
            int lightInd = lightSwitch->findChild(light);
            lightSwitch->insertChild(data->scale, lightInd);
        }
        if (data->scaleInverse == NULL)
        {
            data->scaleInverse = new SoScale;
            int lightInd = lightSwitch->findChild(light);
            lightSwitch->insertChild(data->scaleInverse, lightInd + 1);
        }
        if (data->translation == NULL)
        {
            data->translation = new SoTranslation;
            int scaleInd = lightSwitch->findChild(data->scale);
            lightSwitch->insertChild(data->translation, scaleInd);
        }
        if (data->translationInverse == NULL)
        {
            data->translationInverse = new SoTranslation;
            int scaleInvInd = lightSwitch->findChild(data->scaleInverse);
            lightSwitch->insertChild(data->translationInverse, scaleInvInd + 1);
        }
        // See if the size was already calculated (this happens when we read
        // .env files)...
        SbVec3f oldScale = data->scale->scaleFactor.getValue();
        if (calculatedLightManipSize == FALSE
            && oldScale != SbVec3f(1, 1, 1))
        {
            lightManipSize = oldScale[0];
            calculatedLightManipSize = TRUE;
        }

        data->classPt = this;
        data->colorEditor = NULL;
        data->isManip = FALSE;
        data->type = light->getTypeId();

        // set the correct label name
        char *str;
        if (data->type == SoDirectionalLight::getClassTypeId())
            str = (char *)"Directional ";
        else if (data->type == SoPointLight::getClassTypeId())
            str = (char *)"Point ";
        else if (data->type == SoSpotLight::getClassTypeId())
            str = (char *)"Spot ";
        else
            str = (char *)"Loaded from Env";
        data->name = strdup(str);

        //
        // by default attach the light manipulator to show the light
        //
        editLight(data, TRUE);

        //
        // add the menu entry
        //
        addLightMenuEntry(data);

        return data;
    }

    //======================================================================
    //
    // Description:
    //	build the light menu entry for the given light.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::addLightMenuEntry(InvLightData * data)
    {
        //
        // create the motif menu entry
        //

        Widget menu = menuItems[SV_LIGHT].widget;

        // makes sure menu has been built
        if (menu == NULL)
            return;

        // create the submenu widget, adding a callback to update the toggles
        Arg args[8];
        int argnum = 0;
#ifdef MENUS_IN_POPUP
        SoXt::getPopupArgs(XtDisplay(menu), 0, args, &argnum);
#endif
        data->submenuWidget = XmCreatePulldownMenu(menu, (char *)"LightPulldown", args, argnum);

        XtAddCallback(data->submenuWidget, XmNmapCallback,
                      (XtCallbackProc)InvCoviseViewer::lightSubmenuDisplay,
                      (XtPointer)data);

        // create a cascade menu entry which will bring the submenu
        XtSetArg(args[0], XmNsubMenuId, data->submenuWidget);
        data->cascadeWidget = XtCreateWidget(data->name,
                                             xmCascadeButtonGadgetClass, menu, args, 1);

        // add "on/off" toggle
        data->onOffWidget = XtCreateWidget("On/Off", xmToggleButtonGadgetClass,
                                           data->submenuWidget, NULL, 0);
        XtAddCallback(data->onOffWidget, XmNvalueChangedCallback,
                      (XtCallbackProc)InvCoviseViewer::lightToggleCB, (XtPointer)data);

        // add "Icon" toggle
        data->iconWidget = XtCreateWidget("Icon", xmToggleButtonGadgetClass,
                                          data->submenuWidget, NULL, 0);
        XtAddCallback(data->iconWidget, XmNvalueChangedCallback,
                      (XtCallbackProc)InvCoviseViewer::editLightToggleCB, (XtPointer)data);

        // add "Edit Color" toggle
        data->editColorWidget = XtCreateWidget("Edit Color", xmPushButtonGadgetClass,
                                               data->submenuWidget, NULL, 0);
        XtAddCallback(data->editColorWidget, XmNactivateCallback,
                      (XtCallbackProc)InvCoviseViewer::editLightColorCB, (XtPointer)data);

        // add "Remove" entry
        data->removeWidget = XtCreateWidget("Remove", xmPushButtonGadgetClass,
                                            data->submenuWidget, NULL, 0);
        XtAddCallback(data->removeWidget, XmNactivateCallback,
                      (XtCallbackProc)InvCoviseViewer::removeLightCB, (XtPointer)data);

        // manage children
        XtManageChild(data->onOffWidget);
        XtManageChild(data->iconWidget);
        XtManageChild(data->editColorWidget);
        XtManageChild(data->removeWidget);
        XtManageChild(data->cascadeWidget);
    }

    //======================================================================
    //
    // Description:
    //	Called by "On/Off" light menu entry when toggle changes.
    //
    // Use: static private
    //
    //
    //======================================================================
    void InvCoviseViewer::lightToggleCB(Widget toggle, InvLightData * data, void *)
    {
        data->classPt->turnLightOnOff(data, XmToggleButtonGetState(toggle));
    }

    //======================================================================
    //
    // Description:
    //	Turn the given light on or off.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::turnLightOnOff(InvLightData * data, SbBool flag)
    {
        // check if it is the headlight
        if (data == headlightData)
        {
            setHeadlight(flag);
            sendHeadlight(flag);
        }
        else
        {
            if (flag)
                SWITCH_LIGHT_ON(data->lightSwitch);
            else
                SWITCH_LIGHT_OFF(data->lightSwitch);
        }
    }

    //======================================================================
    //
    // Description:
    //	"Edit" light menu entry callback.
    //
    // Use: static private
    //
    //
    //======================================================================
    void InvCoviseViewer::editLightToggleCB(Widget toggle, InvLightData * data, void *)
    {
        data->classPt->editLight(data, XmToggleButtonGetState(toggle));
    }

    //======================================================================
    //
    // Description:
    //	Attach/detach the correct manipulator on the given light.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::editLight(InvLightData * data, SbBool flag)
    {
        // ??? check if this is for the headlight, which is special cased
        // ??? since a manipulator cannot be used (aligned to camera).
        SbBool forHeadlight = (data == data->classPt->headlightData);

        //
        // attach the manip to the light and add it to the scene
        //
        if (flag)
        {

            if (forHeadlight)
            {

                if (headlightEditor == NULL)
                {
                    headlightEditor = new SoXtDirectionalLightEditor;
                    headlightEditor->setTitle("Headlight Editor");
                }

                // Make sure we have the current viewer's headlight
                SoLight *l = data->classPt->getHeadlight();
                l->ref();
                if (data->light)
                    data->light->unref();
                data->light = l;

                // attach the dir light editor
                // ??? don't use the path from the root to the headlight
                // ??? since we want the light to be relative to the
                // ??? camera (i.e. moving the camera shouldn't affect
                // ??? the arrow in the editor since that direction
                // ??? is relative to the camera).
                SoPath *littlePath = new SoPath(data->light);
                headlightEditor->attach(littlePath);
                headlightEditor->show();
            }
            else if (data->isManip == FALSE)
            {

                // NOTE: if isManip == TRUE, then the light is already a manip
                // and doesn't need to be changed.

                SoLight *newManip = NULL;

                // allocate the right manipulator type if needed
                if (data->type == SoDirectionalLight::getClassTypeId())
                {
                    newManip = new SoDirectionalLightManip;
                    newManip->ref();
                }
                else if (data->type == SoPointLight::getClassTypeId())
                {
                    newManip = new SoPointLightManip;
                    newManip->ref();
                }
                else if (data->type == SoSpotLight::getClassTypeId())
                {
                    newManip = new SoSpotLightManip;
                    newManip->ref();
                    // Set dropOffRate non-zero, or it will look like a pointLight.
                    ((SoSpotLightManip *)newManip)->dropOffRate = .01;
                }

                // get the path from the root to the light node
                SoSearchAction sa;
                sa.setNode(data->light);
                sa.apply(currentViewer->getSceneGraph());
                SoPath *path = sa.getPath();
                // ??? light is probably turned off so we don't
                // ??? need to print a warning message. Just don't
                // ??? do anything
                if (path == NULL)
                {
                    newManip->unref();
                    return;
                }

                path->ref();

                // Set the size for the manip. If this is the first one,
                // then calculate a good size, based on the size of the scene.
                // Once this size is determined, use it for all other light manips.
                // (We need to save the value because the scene size will change
                //  over time, but we want all light manips to be the same size.

                if (!calculatedLightManipSize)
                {
                    // Run a bounding box action on the scene...
                    SoGetBoundingBoxAction ba(currentViewer->getViewportRegion());
                    ba.apply(currentViewer->getSceneGraph());
                    SbBox3f sceneBox = ba.getBoundingBox();
                    SbVec3f size = sceneBox.getMax() - sceneBox.getMin();
                    //XXX pick a good size!
                    lightManipSize = .025 * size.length();

                    calculatedLightManipSize = TRUE;
                }
                data->scale->scaleFactor.setValue(lightManipSize, lightManipSize,
                                                  lightManipSize);
                float invSz = (lightManipSize == 0.0) ? 1.0 : 1.0 / lightManipSize;
                data->scaleInverse->scaleFactor.setValue(invSz, invSz, invSz);

                // Put the manip into the scene.
                if (data->type == SoPointLight::getClassTypeId())
                    ((SoPointLightManip *)newManip)->replaceNode(path);
                else if (data->type == SoDirectionalLight::getClassTypeId())
                    ((SoDirectionalLightManip *)newManip)->replaceNode(path);
                else if (data->type == SoSpotLight::getClassTypeId())
                    ((SoSpotLightManip *)newManip)->replaceNode(path);

                // Okay, now that we stuck that manip in there,
                // we better make a note of it...
                path->unref();
                data->isManip = TRUE;
                data->light->unref();
                data->light = newManip;
            }
        }
        //
        // detach the manip from the light and remove it from the scene
        //
        else
        {
            if (forHeadlight)
            {
                // detach editor from light
                if (headlightEditor != NULL)
                {
                    headlightEditor->detach();
                    headlightEditor->hide();
                }
            }
            else if (data->isManip == TRUE)
            {
                // replace the lightManip node with a regular light node
                // get the path from the root to the lightManip node
                SoSearchAction sa;
                sa.setNode(data->light);
                sa.apply(currentViewer->getSceneGraph());
                SoPath *path = sa.getPath();

                if (path != NULL)
                {
                    path->ref();
                    SoLight *newLight = NULL;
                    if (data->type == SoPointLight::getClassTypeId())
                    {
                        newLight = new SoPointLight;
                        newLight->ref();
                        ((SoPointLightManip *)data->light)->replaceManip(path, (SoPointLight *)newLight);
                    }
                    else if (data->type == SoDirectionalLight::getClassTypeId())
                    {
                        newLight = new SoDirectionalLight;
                        newLight->ref();
                        ((SoDirectionalLightManip *)data->light)->replaceManip(path, (SoDirectionalLight *)newLight);
                    }
                    else if (data->type == SoSpotLight::getClassTypeId())
                    {
                        newLight = new SoSpotLight;
                        newLight->ref();
                        ((SoSpotLightManip *)data->light)->replaceManip(path, (SoSpotLight *)newLight);
                    }
                    else
                    {
                        fprintf(stderr, "InvCoviseViewer::editLight(): newLight used uninitialized\n");
                    }
                    path->unref();
                    data->light->unref();
                    data->light = newLight;
                    data->isManip = FALSE;
                }
            }
        }
    }

    //======================================================================
    //
    // Description:
    //     When removing or writing directional light manip,
    //     we don't want to lose the translation of the manip.
    //     Since the regular light can't hold this value,
    //     get it from the manip and move it to the translation.
    //
    // Use: private
    //
    void InvCoviseViewer::transferDirectionalLightLocation(InvLightData * data)
    //
    //======================================================================
    {
        if (!data->light)
            return;

        if (!data->light->isOfType(SoDirectionalLightManip::getClassTypeId()))
            return;

        // when removing a directional light manip,
        // we don't want to lose the translation of the manip.
        // Since the regular light can't hold this value,
        // get it from the manip and move it to the translation.

        SoDirectionalLightManip *manip
            = (SoDirectionalLightManip *)data->light;
        SoDirectionalLightDragger *dragger
            = (SoDirectionalLightDragger *)manip->getDragger();

        SbVec3f lightTrans(0, 0, 0);

        if (dragger)
            lightTrans += dragger->translation.getValue();

        SbVec3f scl = data->scale->scaleFactor.getValue();
        lightTrans *= scl[0];

        lightTrans += data->translation->translation.getValue();
        data->translation->translation = lightTrans;
        data->translationInverse->translation = -lightTrans;

        // Now zero out the translation in the dragger itself:
        dragger->translation = SbVec3f(0, 0, 0);
    }

    //======================================================================
    //
    // Description:
    //	Called by "Edit Color" light menu entry.
    //
    // Use: static private
    //
    //
    //======================================================================
    void InvCoviseViewer::editLightColorCB(Widget, InvLightData * data, void *)
    {
        // create the color editor with the right title
        if (data->colorEditor == NULL)
        {
            data->colorEditor = new MyColorEditor;
            char str[MAXPATHLEN + 21];
            strcpy(str, data->name);
            strcat(str, " Light Color");
            data->colorEditor->setTitle(str);
        }

        if (!data->colorEditor->isAttached())
        {
            // if this is for the headlight, make sure we have the
            // current viewer headlight
            if (data == data->classPt->headlightData)
            {
                SoLight *l = data->classPt->getHeadlight();
                l->ref();
                if (data->light)
                    data->light->unref();
                data->light = l;
            }

            // normalize the light intensity
            SbColor col;
            col = data->light->color.getValue();
            col *= data->light->intensity.getValue();
            data->light->intensity.setValue(1.0);
            data->light->color.setValue(col);

            data->colorEditor->attach(&data->light->color, data->light);
        }

        data->colorEditor->show();
    }

    //======================================================================
    //
    // Description:
    //	remove button menu entry callback.
    //
    // Use: static private
    //
    //
    //======================================================================
    void InvCoviseViewer::removeLightCB(Widget, InvLightData * data, void *)
    {
        data->classPt->removeLight(data);
    }

    //======================================================================
    //
    // Description:
    //	removes the light from the scene, and removes the light data
    //  and pulldown menu entry.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::removeLight(InvLightData * data)
    {
        // delete the color editor and manip
        delete data->colorEditor;

        // note: deleted  code that dealt with the manip.
        // Since the light and the manip are one and the same now.
        // unrefing the light also removes the manip

        // unref the light (or manip) for this entry
        if (data->light)
            data->light->unref();

        // remove the light from the scene
        lightGroup->removeChild(data->lightSwitch);

        // nuke the menu entry
        if (data->cascadeWidget != NULL)
            XtDestroyWidget(data->cascadeWidget);

        // remove from list and delete the struct
        lightDataList.remove(lightDataList.find(data));
        delete data;
    }

    //======================================================================
    //
    // Description:
    //	Called whenever a light submenu is mapped on screen (update
    //  the toggles)
    //
    // Use: static private
    //
    //
    //======================================================================
    void InvCoviseViewer::lightSubmenuDisplay(Widget, InvLightData * data, void *)
    {
        InvCoviseViewer *sv = data->classPt;
        SbBool set;

        //
        // update the "on/off" toggle
        //
        if (data == sv->headlightData)
            set = sv->isHeadlight();
        else
            set = IS_LIGHT_ON(data->lightSwitch);
        if (set)
            TOGGLE_ON(data->onOffWidget);
        else
            TOGGLE_OFF(data->onOffWidget);

        //
        // update the "Edit" toggle
        //
        if (data == sv->headlightData)
            set = (sv->headlightEditor != NULL && sv->headlightEditor->isVisible());
        else
            set = (data->isManip == TRUE);
        if (set)
            TOGGLE_ON(data->iconWidget);
        else
            TOGGLE_OFF(data->iconWidget);
    }

    //======================================================================
    //
    // Description:
    //	Called after a paste operation has completed.
    //
    // Use: static, private
    //
    //
    //======================================================================
    void InvCoviseViewer::pasteDoneCB(void *userData, SoPathList *pathList)
    {
        InvCoviseViewer *sv = (InvCoviseViewer *)userData;
        sv->pasteDone(pathList);
    }

    //======================================================================
    //
    // Description:
    //	Called after a paste operation has completed, this adds the
    // pasted data to our scene graph.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::pasteDone(SoPathList * pathList)
    {
        if (pathList->getLength() <= 0)
            return;

        // first, detach manips from all selected objects
        detachManipFromAll();

        // now, turn off the sel/desel callbacks.
        // we'll turn them on again after we've adjusted the selection
        selection->removeSelectionCallback(InvCoviseViewer::selectionCallback, this);
        selection->removeDeselectionCallback(InvCoviseViewer::deselectionCallback, this);
        selectionCallbackInactive = TRUE;

        // now deselect all, and build up a selection from the pasted paths
        selection->deselectAll();

        // Add every path in the path list as a child under selection.
        // Then select each of these paths.
        for (int i = 0; i < pathList->getLength(); i++)
        {

            // if the head of the path is a selection node, then don't
            // paste the head - rather, paste all of its children.
            // this makes sure we don't have more than 1 selection node.
            // While we're adding the paths as children, select each path.
            SoPath *p = (*pathList)[i];
            SoNode *head = p->getHead();
            SoPath *selpath;
            if (head->isOfType(SoSelection::getClassTypeId()))
            {
                for (int j = 0; j < ((SoSelection *)head)->getNumChildren(); j++)
                {
                    selection->addChild(((SoSelection *)head)->getChild(j));

                    // create a path from selection to this child
                    // and select the path.
                    selpath = new SoPath(selection);
                    selpath->append(selection->getNumChildren() - 1);
                    selection->select(selpath);
                }
            }
            else
            {
                // not a selection node, so just add it.
                selection->addChild(p->getHead());

                // create a path from selection to this child
                // and select the path.
                selpath = new SoPath(selection);
                selpath->append(selection->getNumChildren() - 1);
                selection->select(selpath);
            }
        }

        // now add manips to all the selected objects
        attachManipToAll(curManip);

        // and turn the sel/desel callbacks back on
        selection->addSelectionCallback(InvCoviseViewer::selectionCallback, this);
        selection->addDeselectionCallback(InvCoviseViewer::deselectionCallback, this);
        selectionCallbackInactive = FALSE;

        // enable/disable keyboard shortcuts
        updateCommandAvailability();

        delete pathList;
    }

    //======================================================================
    //
    // Description:
    //	Show the component
    //
    // Use: public, virtual
    //
    //======================================================================
    void InvCoviseViewer::show()
    {

        XtManageChild(mgrWidget);
    }

    //======================================================================
    //
    // Description:
    //	Show the component
    //
    // Use: public, virtual
    //
    //
    //======================================================================
    void InvCoviseViewer::hide()
    {

        XtUnmanageChild(mgrWidget);
    }

    //======================================================================
    //
    // Description:
    //	Build routine for SceneViewer.  This creates all of the X widgets
    //
    // Use: public, virtual
    //
    //
    //======================================================================
    Widget
    InvCoviseViewer::buildWidget(Widget parent, const char *title)
    {
        // create a form to hold everything together
        mgrWidget = XtVaCreateWidget("RendererForm",
                                     xmFormWidgetClass, parent,
                                     XmNshadowType, XmSHADOW_ETCHED_OUT,
                                     NULL);

        registerWidget(mgrWidget);

        // create the topbar menu
        if (showMenuFlag == TRUE)
        {
            buildAndLayoutTopbarMenu(mgrWidget);
            //
            // PAGEIN specific motif menu stuff
            buildAndLayoutCoviseMenu(currentViewer, mgrWidget);
        }

        // build and layout the current viewer
        setTitle(title);

        //////////////////////
        ////TRY
        //    Arg args[5];
        //
        //    XmString xmstring = XmStringCreateLocalized("Pro5555555555555555555555555555555555555555555555555be");
        //    XtSetArg(args[0], XmNlabelString, xmstring);
        //
        //   Widget title2 = XmCreateLabel(mgrWidget, "description2", args, 1);
        //   XmStringFree(xmstring);
        //   XtManageChild(title2);
        //////////////////////

        currentViewer = new InvExaminerViewer(mgrWidget);
        currentViewer->setRenderer(this); // give our address to the viewer obj

        currentViewer->setDrawStyle(InvViewer::STILL,
                                    InvViewer::VIEW_AS_IS);
        currentViewer->setDrawStyle(InvViewer::INTERACTIVE,
                                    InvViewer::VIEW_SAME_AS_STILL);

        // insert the top draw-style node of the current viewer
        drawStyle->addChild(currentViewer->getDrawStyleSwitch());

        // setting the texture list
        currentViewer->setTextureList(textureList);

        //    cerr << "    InvCoviseViewer::buildWidget(..) get RenderAction"

        // handle the spacemouse input
        spacemouse = NULL;
#ifndef CO_hp1020

        if (coCoviseConfig::isOn("InputDevices.Spaceball", true))
        {
            spacemouse = new SoXtMagellan();
            spacemouse->init(SoXt::getDisplay(), SoXtMagellan::ALL);
            if (spacemouse->exists())
            {
                spacemouse->setTranslationScaleFactor(0.002);
                spacemouse->setRotationScaleFactor(0.002);
                currentViewer->registerDevice(spacemouse);
            }
            else
            {
                delete spacemouse;
                spacemouse = NULL;

                // Check for Magellan device with Linux driver:
                spacemouse = new SoXtLinuxMagellan();
                ((SoXtLinuxMagellan *)spacemouse)->init(SoXt::getDisplay(), SoXtLinuxMagellan::ALL);
                if (spacemouse->exists())
                {
                    currentViewer->registerDevice(spacemouse);
                }
                else
                {
                    delete spacemouse;
                    spacemouse = NULL;
                }
            }
        }
#endif
        //
        // add start and finish edit callback
        currentViewer->addStartCallback(InvCoviseViewer::viewerStartEditCB, this);
        currentViewer->addFinishCallback(InvCoviseViewer::viewerFinishEditCB, this);

        // disable quitting of the renderer via 4DWM
        currentViewer->setWindowCloseCallback(InvCoviseViewer::winCloseCallback, this);

        // get spiining animation status from  covise.config
        int asStatus;

        if (coCoviseConfig::isOn("Renderer.Autospin", false))
        {
            asStatus = CO_ON;
        }
        else
        {
            asStatus = CO_OFF;
        }
        currentViewer->setAnimationEnabled(asStatus);
        currentViewer->setSceneGraph(sceneGraph);

        // since we want no crash traffic in the network we set the
        // ExaminerViewer animation facility OFF
        // currentViewer->setAnimationEnabled( (SbBool)CO_OFF );

        // since we created the camera, do a view all and save this
        // as the starting point (don't want default camera values).
        viewAll();

        saveHomePosition();

        currentViewer->setEventCallback(appEventHandler, this);

        // fog
        fogFlag = FALSE;
        environment->fogType.setValue(SoEnvironment::NONE);
        environment->fogColor.setValue(getBackgroundColor());

        buildAndLayoutViewer(currentViewer);

        // manage those children
        if (showMenuFlag == TRUE)
        {
            XtManageChild(pageinMenuWidget);
            XtManageChild(topbarMenuWidget);
        }

#ifdef _COLLAB_VIEWER
        showMenu(FALSE);
        extern InvRenderManager *rm;
        rm->setSize(600, 600);
#endif

// init Telepointer
#ifndef __linux__
        tpHandler = new TPHandler(currentViewer);
#endif

        currentViewer->setGLRenderAction(highlightRA);
        currentViewer->redrawOnSelectionChange(selection);
        currentViewer->show();

        //
        // master/slave default setup
        //
        if (showMenuFlag == TRUE)
            setMasterSlaveMenu(master);

        // clipboard is for copy/paste of 3d data.
        //??? what if this SceneViewer had its widget destroyed and rebuilt?
        //??? we need to destroy the clipboards when that happens.
        clipboard = new SoXtClipboard(mgrWidget);

        //
        // current state
        //   updateStateLabel("READY");
        //   updateTimeLabel(0.0);
        updateSyncLabel();

        // screen door transparency sucks
        currentViewer->setTransparencyType(SoGLRenderAction::BLEND);

        return mgrWidget;
    }

    //======================================================================
    //
    // Description:
    //	Builds and layout the given viewer.
    //
    // Use: private
    //
    //======================================================================
    void
        InvCoviseViewer::buildAndLayoutViewer(InvExaminerViewer * vwr)
    {
        if (mgrWidget == NULL)
            return;

        // layout the viewer to be attached under the topbar menu
        // (if the pulldown menu is shown)
        Arg args[12];
        int n = 0;
        if (showMenuFlag == TRUE)
        {
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
            n++;
            XtSetArg(args[n], XmNtopWidget, pageinMenuWidget);
            n++;
        }
        else
        {
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
            n++;
        }

        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNtraversalOn, False);
        n++;

        XtSetValues(vwr->getWidget(), args, n);
    }

    //===================================================================
    //
    // After realization, we can set up the color map for the popup menu windows.
    //
    // Use: protected
    //
    void
    InvCoviseViewer::afterRealizeHook()
    //
    //===================================================================
    {
        SoXtComponent::afterRealizeHook();

#ifdef MENUS_IN_POPUP
        if (popupWidget)
            SoXt::addColormapToShell(popupWidget, SoXt::getShellWidget(getWidget()));
#endif
    }

    //======================================================================
    //
    // Description:
    //	Create topbar menu.  Invalid buttons are rendered gray.
    //      Each button's callback include a structure with the ID
    //      of the button and a pointer to the Renderer that created
    //      it.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::buildAndLayoutTopbarMenu(Widget parent)
    {
        if (topbarMenuWidget != NULL)
            return;

        Arg args[12];
        int i, j, n, id;
        WidgetList buttons, subButtons;
        int itemCount, subItemCount;
        WidgetClass widgetClass;
        String callbackReason;

        // create topbar menu
        topbarMenuWidget = XmCreateMenuBar(parent, (char *)"menuBar", NULL, 0);

        itemCount = XtNumber(pulldownData);
        buttons = (WidgetList)XtMalloc(itemCount * sizeof(Widget));

        for (i = 0; i < itemCount; i++)
        {
            // Make Topbar menu button
            Widget subMenu = XmCreatePulldownMenu(topbarMenuWidget, (char *)"LayoutTopbarMenu", NULL, 0);

            id = pulldownData[i].id;
            menuItems[id].widget = subMenu;
            XtAddCallback(subMenu, XmNmapCallback,
                          (XtCallbackProc)InvCoviseViewer::menuDisplay,
                          (XtPointer)&menuItems[id]);

            XtSetArg(args[0], XmNsubMenuId, subMenu);
            buttons[i] = XtCreateWidget(pulldownData[i].name,
                                        xmCascadeButtonGadgetClass, topbarMenuWidget, args, 1);

            // Make subMenu buttons
            subItemCount = pulldownData[i].subItemCount;
            subButtons = (WidgetList)XtMalloc(subItemCount * sizeof(Widget));

            for (j = 0; j < subItemCount; j++)
            {
                if (pulldownData[i].subMenu[j].id == SV_SEPARATOR)
                    subButtons[j] = XtCreateWidget(NULL, xmSeparatorGadgetClass,
                                                   subMenu, NULL, 0);
                else
                {
                    switch (pulldownData[i].subMenu[j].buttonType)
                    {
                    case SV_PUSH_BUTTON:
                        widgetClass = xmPushButtonGadgetClass;
                        callbackReason = XmNactivateCallback;
                        n = 0;
                        break;
                    case SV_TOGGLE_BUTTON:
                        widgetClass = xmToggleButtonGadgetClass;
                        callbackReason = XmNvalueChangedCallback;
                        n = 0;
                        break;
                    case SV_RADIO_BUTTON:
                        widgetClass = xmToggleButtonGadgetClass;
                        callbackReason = XmNvalueChangedCallback;
                        XtSetArg(args[0], XmNindicatorType, XmONE_OF_MANY);
                        n = 1;
                        break;
                    default:
                        print_comment(__LINE__, __FILE__, "SceneViewer INTERNAL ERROR: bad buttonType");
                        widgetClass = 0;
                        callbackReason = 0;
                        n = 0;
                        break;
                    }

                    // check for keyboard accelerator
                    const char *accel = pulldownData[i].subMenu[j].accelerator;
                    const char *accelText = pulldownData[i].subMenu[j].accelText;
                    if (accel != NULL)
                    {
                        XtSetArg(args[n], XmNaccelerator, accel);
                        n++;

                        if (accelText != NULL)
                        {
                            XmString xmstr = XmStringCreate((char *)accelText,
                                                            (char *)XmSTRING_DEFAULT_CHARSET);
                            XtSetArg(args[n], XmNacceleratorText, xmstr);
                            n++;
                            //??? can we ever free the xmstr?
                        }
                    }

                    subButtons[j] = XtCreateWidget(
                        pulldownData[i].subMenu[j].name,
                        widgetClass,
                        subMenu, args, n);

                    id = pulldownData[i].subMenu[j].id;
                    menuItems[id].widget = subButtons[j];
                    XtAddCallback(subButtons[j], callbackReason,
                                  (XtCallbackProc)InvCoviseViewer::processTopbarEvent,
                                  (XtPointer)&menuItems[id]);
                }
            }
            XtManageChildren(subButtons, subItemCount);
            XtFree((char *)subButtons);
        }

        XtManageChildren(buttons, itemCount);
        XtFree((char *)buttons);
        //
        // layout the menu bar
        //
        n = 0;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        XtSetValues(topbarMenuWidget, args, n);

        //
        // Add the light items which are dynamically created
        //

        // first add the headlight
        addLightMenuEntry(headlightData);
        XtUnmanageChild(headlightData->removeWidget);
        XtUnmanageChild(headlightData->editColorWidget);
        XmString xmstr;
        xmstr = XmStringCreate((char *)"Edit", (char *)XmSTRING_DEFAULT_CHARSET);
        XtSetArg(args[0], XmNlabelString, xmstr);
        XtSetValues(headlightData->iconWidget, args, 1);
        XmStringFree(xmstr);

        // now the regular lights
        for (i = 0; i < lightDataList.getLength(); i++)
            addLightMenuEntry((InvLightData *)lightDataList[i]);

        XtSetArg(args[0], XmNsensitive, false);
        XtSetValues(menuItems[SV_FILE_SNAP_ALL].widget, args, 1);
    }

    Widget InvCoviseViewer::getSeqParent()
    {
        return leftRow;
    }

    //======================================================================
    //
    // Description:
    //	Builds and layout the Covise menu.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::buildAndLayoutCoviseMenu(InvExaminerViewer *,
                                                   Widget parent)
    {
        if (mgrWidget == NULL)
            return;

        if (pageinMenuWidget != NULL)
            return;

        //Widget form;
        //Widget rowcol1;
        Widget rowcol2;
        Widget rowcol3;
        Widget rowcol4;
        //Widget label1, label2, label3, label4;
        //  Widget colormaparea;

        Arg args[12];
        int n = 0;
        char text[32];

        ///    // build the viewer if necessary
        ///    if (vwr->getWidget() == NULL)
        ///	vwr->build(mgrWidget);

        // layout the menu to be attached under the topbar menu
        // (if the pulldown menu is shown)

        // create a covise menu
        XtSetArg(args[n], XmNorientation, XmVERTICAL);
        n++;
        pageinMenuWidget = XtCreateWidget("pageinMenu", xmRowColumnWidgetClass, parent, args, n);

        //  first a horizontal row column
        //
        n = 0;
        topForm_ = XtVaCreateManagedWidget("RowColumn",
                                           xmRowColumnWidgetClass, pageinMenuWidget,
                                           XmNpacking, XmPACK_TIGHT,
                                           XmNnumColumns, 3,
                                           XmNshadowType, XmSHADOW_ETCHED_OUT,
                                           XmNtraversalOn, False,
                                           XmNorientation, XmHORIZONTAL,
                                           NULL);

        //  colormap list
        //
        rowcol3 = XtVaCreateManagedWidget("RowColumn3",
                                          xmRowColumnWidgetClass, topForm_,
                                          XmNpacking, XmPACK_TIGHT,
                                          XmNorientation, XmVERTICAL,
                                          XmNtraversalOn, False,
                                          //  XmNsensitive ,      True,
                                          XmNnumColumns, 1,
                                          NULL);

        //strcpy(text,"      ");
        // colormaparea = XtVaCreateManagedWidget("Colormap",
        //	          xmLabelWidgetClass, rowcol3,
        //	          XmNlabelString,  XmStringCreateSimple(text),
        //	          NULL) ;
        n = 0;
        // create scrolled list
        XmString xmstr1[2];
        createColormapList((char *)"           Colormaps             ", xmstr1, 0, rowcol3);

        //
        // object list
        //
        rowcol4 = XtVaCreateManagedWidget("RowColumn4",
                                          xmRowColumnWidgetClass, topForm_,
                                          XmNshadowType, XmSHADOW_ETCHED_OUT,
                                          XmNpacking, XmPACK_TIGHT,
                                          XmNorientation, XmVERTICAL,
                                          XmNtraversalOn, False,
                                          NULL);
        n = 0;
        // create scrolled list
        XmString xmstr2[2];
        createList((char *)"        Geometry Objects         ", xmstr2, 0, rowcol4);

        rowcol2 = XtVaCreateManagedWidget("RowColumn2",
                                          xmRowColumnWidgetClass, topForm_,
                                          XmNpacking, XmPACK_TIGHT,
                                          XmNorientation, XmVERTICAL,
                                          XmNsensitive, True,
                                          XmNtraversalOn, False,
                                          XmNnumColumns, 1,
                                          NULL);
        n = 0;
        // left side : labels for info things
        //
        //strcpy(text," Mode ");
        ///* label1 = */ XtVaCreateManagedWidget("Label1",
        //		    xmDrawnButtonWidgetClass, rowcol1,
        //		    XmNlabelString, XmStringCreateSimple(text),
        //                   XmNshadowType, XmSHADOW_ETCHED_IN,
        //		    NULL) ;

        // strcpy(text," Sync ");
        ///* label2 = */ XtVaCreateManagedWidget("Label2",
        //	    xmDrawnButtonWidgetClass, rowcol1,
        //	    XmNlabelString, XmStringCreateSimple(text),
        //                XmNshadowType, XmSHADOW_ETCHED_IN,
        //		    NULL) ;

        // strcpy(text,"      ");
        // /* label3 = */ XtVaCreateManagedWidget("Label3",
        //		    xmDrawnButtonWidgetClass, rowcol1,
        //		    XmNlabelString, XmStringCreateSimple(text),
        //                    XmNshadowType, XmSHADOW_ETCHED_IN,
        //		    NULL) ;

        //  strcpy(text,"      ");
        //  /* label4 = */ XtVaCreateManagedWidget("Label4",
        //		    xmDrawnButtonWidgetClass, rowcol1,
        //	    XmNlabelString, XmStringCreateSimple(text),
        //              XmNshadowType, XmSHADOW_ETCHED_IN,
        //    NULL) ;

        // right side : the info labels

        strcpy(text, "      ");
        masterlabel = XtVaCreateManagedWidget("Masterlabel",
                                              xmDrawnButtonWidgetClass, rowcol2,
                                              XmNlabelString, XmStringCreateSimple(text),
                                              XmNshadowType, XmSHADOW_IN,
                                              XmNtraversalOn, False,
                                              NULL);

        strcpy(text, "      ");
        synclabel = XtVaCreateManagedWidget("Synclabel",
                                            xmDrawnButtonWidgetClass, rowcol2,
                                            XmNlabelString, XmStringCreateSimple(text),
                                            XmNshadowType, XmSHADOW_IN,
                                            XmNtraversalOn, False,
                                            NULL);

        strcpy(text, "      ");
        timelabel = XtVaCreateManagedWidget("Timelabel",
                                            xmDrawnButtonWidgetClass, rowcol2,
                                            XmNlabelString, XmStringCreateSimple(text),
                                            XmNshadowType, XmSHADOW_IN,
                                            XmNtraversalOn, False,
                                            NULL);

        strcpy(text, "      ");
        statelabel = XtVaCreateManagedWidget("Statelabel",
                                             xmDrawnButtonWidgetClass, rowcol2,
                                             XmNlabelString, XmStringCreateSimple(text),
                                             XmNshadowType, XmSHADOW_IN,
                                             XmNtraversalOn, False,
                                             NULL);
        // XmStringFree(xmstr[0]);  !!!!!

        //
        // two rowcolumns for input lines and strings
        //
        leftRow = XtVaCreateManagedWidget("RowColumn1",
                                          xmFormWidgetClass, parent,
                                          //XmNpacking,		XmPACK_TIGHT,
                                          //XmNorientation,	XmVERTICAL,
                                          //XmNsensitive ,      True,
                                          XmNnumColumns, 1,
                                          XmNtopAttachment, XmATTACH_WIDGET,
                                          XmNtopWidget, topbarMenuWidget,
                                          XmNleftAttachment, XmATTACH_FORM,
                                          XmNtraversalOn, False,
                                          NULL);

        n = 0;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNtopWidget, topbarMenuWidget);
        n++;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNleftWidget, leftRow);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        XtSetValues(pageinMenuWidget, args, n);

#ifdef _AIRBUS
        //XtUnmanageChild(rowcol2);
        XtUnmanageChild(rowcol4);
#endif
    }

    //======================================================================
    //
    // Description:
    //	create object list
    //
    // Use : private
    //
    //======================================================================
    Widget InvCoviseViewer::createText(Widget w, char *l, char *text)
    {

        // create main container widget
        Widget part = XtVaCreateWidget("RowColumn",
                                       xmRowColumnWidgetClass, w,
                                       XmNpacking, XmPACK_TIGHT,
                                       XmNnumColumns, 1,
                                       XmNorientation, XmHORIZONTAL,
                                       NULL);

        // create the label widget
        Widget label1 = XtVaCreateManagedWidget("Label",
                                                xmLabelWidgetClass, part,
                                                XmNlabelString,
                                                XmStringCreateSimple(l),
                                                NULL);

        // create the  text widget
        Widget label2 = XtVaCreateManagedWidget("Text",
                                                xmTextWidgetClass, part,
                                                XmNcolumns, 40,
                                                XmNlabelString,
                                                XmStringCreateSimple(text),
                                                NULL);
        (void)label1;
        (void)label2;
        XtManageChild(part);

        return part;
    }

    //======================================================================
    //
    // Description:
    //	create object list
    //
    // Use : private
    //
    //========================================================================
    void InvCoviseViewer::createList(char *title, XmStringTable,
                                     int, Widget parent)
    {

        const int citem = 5; /* no. of visible items */

        /****** create list title ******/
        Widget label = XtVaCreateManagedWidget("Label",
                                               xmLabelWidgetClass, parent,
                                               XmNshadowType, XmSHADOW_ETCHED_OUT,
                                               XmNlabelString, XmStringCreateSimple(title),
                                               NULL);

        /****** create scrolled list ******/
        objectlist = XmCreateScrolledList(parent, (char *)"List", NULL, 0);
        XtManageChild(objectlist);
        XtVaSetValues(objectlist,
                      XmNsensitive, True, /// Changed from false
                      //    XmNitemCount,		nitem,
                      //    XmNitems,			items,
                      XmNselectionPolicy, XmSINGLE_SELECT,
                      XmNvisibleItemCount, citem,
                      XmNscrollBarDisplayPolicy, XmSTATIC,
                      // XmNsensitive,		Master,
                      NULL);
        (void)label;
        XtAddCallback(objectlist, XmNsingleSelectionCallback,
                      (XtCallbackProc)InvCoviseViewer::objectListCB, (XtPointer) this);
    }

    //======================================================================
    //
    // Description:
    //	select object according to pick in list
    //
    // Use: private, static
    //
    //======================================================================
    void InvCoviseViewer::objectListCB(Widget, XtPointer user_data,
                                       XmListCallbackStruct * list_data)
    {
        char buffer[255];
        static char oldname[255];
        static int first_time = 0;
        static int selected = 0;

        InvCoviseViewer *r = (InvCoviseViewer *)user_data;

        // now, turn off the sel/desel callbacks.
        // we'll turn them on again after we've adjusted the selection
        r->selection->removeSelectionCallback(InvCoviseViewer::selectionCallback, (void *)r);
        r->selection->removeDeselectionCallback(InvCoviseViewer::deselectionCallback, (void *)r);
        r->selectionCallbackInactive = TRUE;

        if (first_time == 0)
        {
            strcpy(oldname, "DEFAULT");
            first_time = 1;
        }

        r->selection->deselectAll();

        // if item was selected select it, if item was deselected select it

        char name[255];
        char *itemText;
        XmStringGetLtoR(list_data->item, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET, &itemText);
        strcpy(name, itemText);

        if (!strcmp(oldname, name)) // toggle
        {
            if (selected)
            {
                XmListDeselectItem(r->objectlist, list_data->item);
                selected = 0;
                // update slaves
                if ((r->isMaster() == TRUE) && (r->isSynced() != SYNC_LOOSE))
                {
                    sprintf(buffer, "G_%s", name);
                    r->sendDeselection(buffer);
                    // cerr << "Sending a deselection for " << name << "to slaves" << endl;
                }
            }
            else
            {
                XmListSelectItem(r->objectlist, list_data->item, False);
                selected = 1;
                SoShape *shape = r->findShapeNode(name);
                if (shape != NULL)
                {
                    r->selection->select((SoNode *)shape);
                    // update slaves
                    if ((r->isMaster() == TRUE) && (r->isSynced() != SYNC_LOOSE))
                    {
                        sprintf(buffer, "G_%s", name);
                        r->sendSelection(buffer);
                        // cerr << "Sending a selection for " << name << "to slaves" << endl;
                    }
                }
            }
        }
        else // select other item
        {
            XmListSelectItem(r->objectlist, list_data->item, False);
            selected = 1;
            SoShape *shape = r->findShapeNode(name);
            if (shape != NULL)
            {
                r->selection->select((SoNode *)shape);
                // update slaves
                if ((r->isMaster() == TRUE) && (r->isSynced() != SYNC_LOOSE))
                {
                    sprintf(buffer, "G_%s", name);
                    r->sendSelection(buffer);
                    // cerr << "Sending a selection for " << name << "to slaves" << endl;
                }
            }
        }
        strcpy(oldname, name);

        //
        // turn on again
        r->selection->addSelectionCallback(InvCoviseViewer::selectionCallback, (void *)r);
        r->selection->addDeselectionCallback(InvCoviseViewer::deselectionCallback, (void *)r);
        r->selectionCallbackInactive = FALSE;
    }

    //======================================================================
    //
    // Description:
    //	update object list according to selected objects.
    //
    // Use: private, static
    //
    //======================================================================
    void InvCoviseViewer::addToObjectList(const char *name)
    {

        XmListAddItem(objectlist, XmStringCreateSimple((char *)name), 0);
    }

    //======================================================================
    //
    // Description:
    //	update object list according to selected objects.
    //
    // Use: private, static
    //
    //======================================================================
    void InvCoviseViewer::removeFromObjectList(const char *name)
    {

        //
        // first deselect all items
        XmListDeselectAllItems(objectlist);

        // remove entry in list
        XmListDeleteItem(objectlist, XmStringCreateSimple((char *)name));
    }

    //======================================================================
    //
    // Description:
    //	create colormap list
    //
    // Use : private
    //
    //========================================================================
    void InvCoviseViewer::createColormapList(char *title, XmStringTable,
                                             int, Widget parent)
    {

        const int citem = 5; /* no. of visible items */

        /****** create list title ******/
        Widget label = XtVaCreateManagedWidget("ColorLabel",
                                               xmLabelWidgetClass, parent,
                                               XmNshadowType, XmSHADOW_ETCHED_OUT,
                                               XmNlabelString, XmStringCreateSimple(title),
                                               NULL);

        /****** create scrolled list ******/
        colormaplist = XmCreateScrolledList(parent, (char *)"ColormapList", NULL, 0);
        XtManageChild(colormaplist);
        XtVaSetValues(colormaplist,
                      XmNsensitive, True, /// Changed from false
                      //    XmNitemCount,		nitem,
                      //    XmNitems,			items,
                      XmNselectionPolicy, XmSINGLE_SELECT,
                      XmNvisibleItemCount, citem,
                      XmNscrollBarDisplayPolicy, XmSTATIC,
                      // XmNsensitive,		Master,
                      NULL);
        (void)label;
        XtAddCallback(colormaplist, XmNsingleSelectionCallback,
                      (XtCallbackProc)InvCoviseViewer::colormapListCB, (XtPointer) this);
    }

    //======================================================================
    //
    // Description:
    //	select item according to pick in colormap list
    //
    // Use: private, static
    //
    //======================================================================
    void
    InvCoviseViewer::colormapListCB(Widget, XtPointer user_data,
                                    XmListCallbackStruct * list_data)
    {
        char buffer[255];

        //    cerr << "               InvCoviseViewer::colormapListCB(..) called" << endl;

        InvCoviseViewer *r = (InvCoviseViewer *)user_data;

        if (c_first_time == 0)
        {
            strcpy(r->c_oldname, "DEFAULT");
            c_first_time = 1;
        }

        // if item was selected select it, if item was deselected select it

        char name[255];
        char *itemText;
        XmStringGetLtoR(list_data->item, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET, &itemText);

        strcpy(name, itemText);
        int sel;
        if (!r->cmapSelected_.get(itemText, sel))
        {
            sel = 0;
        }

        if (!strcmp(r->c_oldname, name)) // toggle
        {
            if (sel == 1)
            {
                XmListDeselectItem(r->colormaplist, list_data->item);
                r->cmapSelected_.remove(itemText);
                r->colormap_manager->hideAllColormaps();
            }
            else
            {
                XmListSelectItem(r->colormaplist, list_data->item, False);
                r->cmapSelected_.removeAll();
                r->cmapSelected_.add(itemText, 1);
                r->colormap_manager->showColormap(name, (SoXtExaminerViewer *)(r->currentViewer));
            }
        }
        else // select other item
        {
            XmListSelectItem(r->colormaplist, list_data->item, False);
            r->cmapSelected_.removeAll();
            r->cmapSelected_.add(itemText, 1);
            r->colormap_manager->showColormap(name, (SoXtExaminerViewer *)(r->currentViewer));
        }
        strcpy(r->c_oldname, name);

        // update slaves
        if ((r->isMaster() == TRUE) && (r->isSynced() != SYNC_LOOSE))
        {

            sprintf(buffer, "%s", r->colormap_manager->currentColormap());
            r->sendColormap(buffer);
        }
    }

    //======================================================================
    //
    // Description:
    //	update object list according to selected objects.
    //
    // Use: private, static
    //
    //======================================================================
    void InvCoviseViewer::addToColormapList(const char *name)
    {

        if (!XmListItemExists(colormaplist, XmStringCreateSimple((char *)name)))
        {
            XmListAddItem(colormaplist, XmStringCreateSimple((char *)name), 0);
        }
        int sel = 0;
        cmapSelected_.get(name, sel);
        if (sel == 1)
        {
            colormap_manager->hideAllColormaps();
            colormap_manager->showColormap(name, (SoXtExaminerViewer *)(this->currentViewer));
            XmListSelectItem(colormaplist, XmStringCreateSimple((char *)name), False);
        }
    }

    //======================================================================
    //
    // Description:
    //	update object list according to selected objects.
    //
    // Use: private, static
    //
    //======================================================================
    void InvCoviseViewer::removeFromColormapList(const char *name)
    {

        //
        // first deselect all items
        // live is not that easy
        //XmListDeselectAllItems(colormaplist);
        XmListDeselectItem(colormaplist, XmStringCreateSimple((char *)name));

        // remove entry in list
        XmListDeleteItem(colormaplist, XmStringCreateSimple((char *)name));
    }

    //======================================================================
    //
    // Description:
    //
    //
    // Use: private
    //
    //======================================================================
    SoPath *InvCoviseViewer::findShapeByName(char *shapeName)
    {
        //print_comment(__LINE__,__FILE__,"in findShapeByName ");

        const char *name;
        char objName[255];
        SbName string;
        int i;

        SoSearchAction saLabel;
        SoSearchAction saShape;

        SoPathList listLabel;
        SoPathList listShape;

        SoLabel *label;

        SoPath *shapePath = NULL;

        saLabel.setFind(SoSearchAction::TYPE);
        saLabel.setInterest(SoSearchAction::ALL);
        saLabel.setType(SoLabel::getClassTypeId());

        /// saLabel.setType( SoLabel::getClassTypeId() );
        /// saLabel.setExact(TRUE);
        /// saLabel.setFindAll(TRUE);
        saLabel.apply(selection);

        saShape.setFind(SoSearchAction::TYPE);
        saShape.setInterest(SoSearchAction::ALL);
        saShape.setType(SoLabel::getClassTypeId());

        /// saShape.setType( SoShape::getClassTypeId() );
        ///!!????? saShape.setExact(FALSE);
        /// saShape.setFindAll(TRUE);
        saShape.apply(selection);

        // get the list of paths
        listLabel = saLabel.getPaths();
        listShape = saShape.getPaths();

        // cycle through the list and find (first) match
        if (listLabel.getLength() == listShape.getLength())
        {
            for (i = 0; i < listLabel.getLength(); i++)
            {
                label = (SoLabel *)(listLabel[i]->getTail());
                string = label->label.getValue();
                name = string.getString();
                strcpy(objName, name);

                if (strcmp(objName, shapeName) == 0)
                {
                    shapePath = listShape[i];
                    break;
                }
            }
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: findShapeByName : num labels != num shapes");

        return shapePath;
    }

    //======================================================================
    //
    // Description:
    //	update object list according to selected objects.
    //      is called from selection and deselection callbacks
    //
    // Use: private
    //
    //======================================================================
    void
    InvCoviseViewer::updateObjectList(SoPath * selectionPath, SbBool isSelection)
    {
        int j;
        const char *name;
        char objName[255];
        SbName string;

        SoNode *node = selectionPath->getTail();

        //
        // look on the left side if there is
        //
        SoGroup *sep = (SoGroup *)selectionPath->getNodeFromTail(1);
        (void)node;
        //
        // should be a separator !

        if (sep->isOfType(SoSeparator::getClassTypeId()))
        {
            // look for the label under the separator
            for (j = 0; j < sep->getNumChildren(); j++)
            {
                SoNode *n = sep->getChild(j);
                if (n->isOfType(SoLabel::getClassTypeId()))
                {
                    // look into the label
                    SoLabel *l = (SoLabel *)sep->getChild(j);
                    string = l->label.getValue();
                    name = string.getString();
                    strcpy(objName, name);
                    //  with this name we are going into the motif list and
                    //  search for a match

                    int itempos = XmListItemPos(objectlist, XmStringCreateSimple(objName));

                    // select or deselect the itempos

                    if (isSelection)
                    {
                        XmListSelectPos(objectlist, itempos, False);
                        XmUpdateDisplay(objectlist);
                        break;
                    }
                    else
                    {
                        XmListDeselectPos(objectlist, itempos);
                        XmUpdateDisplay(objectlist);
                        break;
                    }
                }
            }
        }
        else
        {
            sep = (SoGroup *)selectionPath->getNodeFromTail(2);
            if (sep->isOfType(SoSeparator::getClassTypeId()))
            {

                for (j = 0; j < sep->getNumChildren(); j++)
                {
                    SoNode *n = sep->getChild(j);
                    if (n->isOfType(SoLabel::getClassTypeId()))
                    {
                        // look into the label
                        SoLabel *l = (SoLabel *)sep->getChild(j);
                        string = l->label.getValue();
                        name = string.getString();

                        strcpy(objName, name);
                        //  with this name we are going into the motif list and
                        //  search for a match

                        int itempos = XmListItemPos(objectlist, XmStringCreateSimple(objName));

                        // select or deselect the itempos

                        if (isSelection)
                        {
                            XmListSelectPos(objectlist, itempos, False);
                            XmUpdateDisplay(objectlist);
                            break;
                        }
                        else
                        {
                            XmListDeselectPos(objectlist, itempos);
                            XmUpdateDisplay(objectlist);
                            break;
                        }
                    }
                }
            }
        }
    }

    //======================================================================
    //
    // Description:
    //	set the correct label for master/slave mode.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::updateObjectListItem(int onORoff, char *name)
    {

        int itempos = XmListItemPos(objectlist, XmStringCreateSimple(name));
        if (onORoff)
        {
            XmListSelectPos(objectlist, itempos, False);
            XmUpdateDisplay(objectlist);
        }
        else
        {
            XmListDeselectPos(objectlist, itempos);
            XmUpdateDisplay(objectlist);
        }
    }

    //======================================================================
    //
    // Description:
    //	set the correct label for master/slave mode.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::updateMasterLabel(short master)
    {
        char label[20];
        // Arg args[2];
        Arg args[1];

        if (master == TRUE)
            strcpy(label, "MASTER");
        else
            strcpy(label, "SLAVE ");

        XtSetArg(args[0], XmNlabelString, XmStringCreateSimple(label));
        XtSetValues(masterlabel, args, 1);
    }

    //======================================================================
    //
    // Description:
    //	set the correct label for renderer state.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::updateStateLabel(char *string)
    {

        Arg args[1];

        XtSetArg(args[0], XmNlabelString, XmStringCreateSimple(string));
        XtSetValues(statelabel, args, 1);
    }

    //======================================================================
    //
    // Description:
    //	set the correct time for time information label.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::updateTimeLabel(float time)
    {
        char label[20];
        Arg args[1];

        sprintf(label, "%f", time);

        XtSetArg(args[0], XmNlabelString, XmStringCreateSimple(label));
        XtSetValues(timelabel, args, 1);
    }

    //======================================================================
    //
    // Description:
    //	set the correct sync mode for sync information label.
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::updateSyncLabel()
    {
        char label[20];
        Arg args[1];

        if (sync_flag == SYNC_LOOSE)
            sprintf(label, "%s", "LOOSE");
        else if (sync_flag == SYNC_SYNC)
            sprintf(label, "%s", "SYNC");
        else
            sprintf(label, "%s", "TIGHT");

        XtSetArg(args[0], XmNlabelString, XmStringCreateSimple(label));
        XtSetValues(synclabel, args, 1);
    }

    //======================================================================
    //
    // Description:
    //	set the time for rendering.
    //
    // Use: public
    //
    //======================================================================
    void InvCoviseViewer::setRenderTime(float time)
    {
        // for now show the rendering time
        updateTimeLabel(time);
    }

    //======================================================================
    //
    // Description:
    //	Show/hide the pulldown menu bar.
    //
    // Use: public
    //
    //======================================================================
    void InvCoviseViewer::showMenu(SbBool flag)
    {
        if (showMenuFlag == flag || mgrWidget == NULL)
        {
            showMenuFlag = flag;
            return;
        }

        showMenuFlag = flag;

        if (showMenuFlag == TRUE)
        {

            // turn topbar menu on
            if (topbarMenuWidget == NULL)
                buildAndLayoutTopbarMenu(mgrWidget);
            if (pageinMenuWidget == NULL)
                buildAndLayoutCoviseMenu(currentViewer, mgrWidget);

            XtManageChild(topbarMenuWidget);
            XtManageChild(pageinMenuWidget);

            //
            // master/slave setup
            setMasterSlaveMenu(master);

            // attach viewer to bottom of menu
            Arg args[2];
            int n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
            n++;
            XtSetArg(args[n], XmNtopWidget, pageinMenuWidget);
            n++;
            XtSetValues(currentViewer->getWidget(), args, n);
        }
        else
        {
            // attach viewer to form
            Arg args[2];
            int n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
            n++;
            XtSetValues(currentViewer->getWidget(), args, n);

            // turn topbar menu off
            if (topbarMenuWidget != NULL)
                XtUnmanageChild(topbarMenuWidget);
            if (pageinMenuWidget != NULL)
                XtUnmanageChild(pageinMenuWidget);
        }
    }

    //======================================================================
    //
    //  Called when user tried to quit Renderer via double clicking on
    //  on the upper left corner  '-' of the border
    //
    //  Use: private, static
    //
    //
    //======================================================================
    // my data
    void InvCoviseViewer::winCloseCallback(void *userData,
                                           SoXtComponent *comp) // object
    {
        (void)userData;
        (void)comp;
        // we do nothing here since we want our Renderer to stay alive
    }

    //===================================================================
    //
    //  Manage the changes in the selected node(s)
    //
    //  Use: private, static
    //
    SoPath *
    InvCoviseViewer::pickFilterCB(void *userData, const SoPickedPoint *pick)
    //
    //===================================================================
    {
        InvCoviseViewer *sv = (InvCoviseViewer *)userData;
        SoPath *filteredPath = NULL;

        // If there are any transform manips along the path, check if they
        // belong to our personal set of manips.
        // If so, change the path so it points to the object the manip
        // is attached to.

        SoFullPath *fullP = (SoFullPath *)pick->getPath();
        //cerr << "InvCoviseViewer::pickFilterCB(..) org Path length " << fullPP->getLength() << endl;

        //    SoFullPath *fullPP = (SoFullPath *) InvPlaneMover::pickFilterCB(sv->pm_,pick);
        //    cerr << "InvCoviseViewer::pickFilterCB(..) after  InvPlaneMover::pickFilterCB Path length " << fullP->getLength() << endl;

        if (Annotations->isActive())
        {
            return Annotations->pickFilterCB(Annotations, pick);
        }

        SbVec3f point = pick->getPoint();

        sv->pm_->setPosition(point);

        SoNode *n;
        for (int i = 0; i < fullP->getLength(); i++)
        {
            n = fullP->getNode(i);
            if (n->isOfType(SoTransformManip::getClassTypeId()))
            {
                int which = sv->maniplist->find((SoTransformManip *)n);
                if (which != -1)
                {
                    filteredPath = sv->maniplist->getSelectionPath(which);
                    return filteredPath;
                }
            }
        }

        // If we didn't pick one of our manipulators, then return the pickPath
        filteredPath = pick->getPath();
        return filteredPath;
    }

    //======================================================================
    //
    //  Manage the changes in the selected node(s)
    //
    //  Use: private, static
    //
    //
    //======================================================================
    // my data
    void InvCoviseViewer::deselectionCallback(void *userData,
                                              SoPath *deselectedObject) // object
    {
        char objName[255];
        int num;
        SbName string;
        const SoPathList *deselectedObjectsList;

        InvCoviseViewer *sv = (InvCoviseViewer *)userData;

        if (sv->selectionCallbackInactive)
            return; // don't do anything (destructive)

        // now, turn off the sel/desel callbacks.
        // we'll turn them on again after we've adjusted the selection
        sv->selection->removeSelectionCallback(InvCoviseViewer::selectionCallback, (void *)sv);
        sv->selection->removeDeselectionCallback(InvCoviseViewer::deselectionCallback, (void *)sv);
        sv->selectionCallbackInactive = TRUE;

        InvPlaneMover::deSelectionCB(sv->pm_, deselectedObject);

        // remove the manip
        sv->detachManip(deselectedObject);

        // Remove editors
        if (sv->materialEditor)
            sv->materialEditor->detach();

        if (sv->colorEditor)
            sv->colorEditor->detach();

        if (sv->transformSliderSet)
            sv->transformSliderSet->setNode(NULL); // ??? same as detach ??? rc

        // reset current transform node to NULL
        if (sv->currTransformNode != NULL)
        {
            // sv->transformSensor->detach(sv->currTransformNode->transform);
            sv->transformSensor->detach();
            // sv->transformSensor->detach(sv->currTransformNode);
            sv->currTransformPath = NULL;
            sv->currTransformNode = NULL;
            sv->transformNode = NULL;
            sv->transformPath = NULL;
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: Cannot detach transformCallback from node");

        // get the object-names from the selected objects
        num = sv->selection->getNumSelected();
        deselectedObjectsList = sv->selection->getList();

        // enable/disable cmd key shortcuts and menu items
        sv->updateCommandAvailability();

        // update the object list
        sv->updateObjectList(deselectedObject, (SbBool)FALSE);

        // send to other renderer's
        for (int i = 0; i < num; i++)
        {
            string = (*deselectedObjectsList)[i]->getTail()->getName();
            strcpy(objName, string.getString());
            // cerr << "InvCoviseViewer::deselectionCallback: Deselected object is : " << objName << endl;
            sv->sendDeselection(objName);
        }
        string = deselectedObject->getTail()->getName();
        strcpy(objName, string.getString());
        sv->sendDeselection(objName);

        // turn on callbacks again
        sv->selection->addSelectionCallback(InvCoviseViewer::selectionCallback, (void *)sv);
        sv->selection->addDeselectionCallback(InvCoviseViewer::deselectionCallback, (void *)sv);
        sv->selectionCallbackInactive = FALSE;
    }

    //======================================================================
    //
    //  Manage the changes in the selected node(s)
    //
    //  Use: private, static
    //
    //
    //======================================================================
    void
        // my data
        InvCoviseViewer::selectionCallback(void *userData,
                                           SoPath *selectedObject) // object
    {

        InvCoviseViewer *sv = (InvCoviseViewer *)userData;
        char objName[255];
        int num;
        SbName string;
        SoPath *editTransformPath;
        const SoPathList *selectedObjectsList;
        SoNodeList *nodeList;
        SoPathList *pathList;

        if (sv->selectionCallbackInactive)
            return; // don't do anything (destructive)

        if (Annotations->isActive())
        {

            Widget widget = sv->getRenderAreaWidget();
            SbVec2s sz;
            if (widget != NULL)
            {
                Arg args[2];
                int n = 0;
                XtSetArg(args[n], XtNwidth, &sz[0]);
                n++;
                XtSetArg(args[n], XtNheight, &sz[1]);
                n++;
                XtGetValues(widget, args, n);
            }

            SbViewportRegion vpReg;
            vpReg.setWindowSize(sz);

            vpReg = sv->currentViewer->getCamera()->getViewportBounds(vpReg);

            SoGetBoundingBoxAction bBoxAct(vpReg);
            bBoxAct.apply(sv->selection);

            SbBox3f bb = bBoxAct.getBoundingBox();

            Annotations->setSize(bb);
            Annotations->selectionCB(Annotations, selectedObject);

            sv->selection->deselectAll();
            return;
        }

        // first we try to find an object with interaction capabilities
        // (at the moment (22.08.01) CuttingSurface is the only one)
        int len = selectedObject->getLength();
        int ii;
        char *selObjNm;
        int csFlg = 0;
        for (ii = 0; ii < len; ii++)
        {
            SoNode *obj = selectedObject->getNode(ii);
            char *tmp = (char *)obj->getName().getString();
            selObjNm = new char[1 + strlen(tmp)];
            strcpy(selObjNm, tmp);
            if (strncmp(selObjNm, "CuttingSurface", 14) == 0)
            {
                csFlg = 1;
                break;
            }
            if (strncmp(selObjNm, "VectorField", 11) == 0)
            {
                csFlg = 1;
                break;
            }
            delete[] selObjNm;
        }

        if (csFlg)
        {
            Widget widget = sv->getRenderAreaWidget();
            SbVec2s sz;
            if (widget != NULL)
            {
                Arg args[2];
                int n = 0;
                XtSetArg(args[n], XtNwidth, &sz[0]);
                n++;
                XtSetArg(args[n], XtNheight, &sz[1]);
                n++;
                XtGetValues(widget, args, n);
            }

            SbViewportRegion vpReg;
            vpReg.setWindowSize(sz);

            vpReg = sv->currentViewer->getCamera()->getViewportBounds(vpReg);

            SoGetBoundingBoxAction bBoxAct(vpReg);
            bBoxAct.apply(selectedObject);
            // 	cerr << "InvCoviseViewer::selectionCallback(..) BBox Xmin: "
            // 	     << bBoxAct.getBoundingBox().getMin()[0]
            // 	     << endl;
            // 	cerr << "InvCoviseViewer::selectionCallback(..) BBox Xmax: "
            // 	     << bBoxAct.getBoundingBox().getMax()[0]
            // 	     << endl;
            // 	cerr << "InvCoviseViewer::selectionCallback(..) BBox Ymin: "
            // 	     << bBoxAct.getBoundingBox().getMin()[1]
            // 	     << endl;
            // 	cerr << "InvCoviseViewer::selectionCallback(..) BBox Ymax: "
            // 	     << bBoxAct.getBoundingBox().getMax()[1]
            // 	     << endl;
            // 	cerr << "InvCoviseViewer::selectionCallback(..) BBox Zmin: "
            // 	     << bBoxAct.getBoundingBox().getMin()[2]
            // 	     << endl;
            // 	cerr << "InvCoviseViewer::selectionCallback(..) BBox Zmax: "
            // 	     << bBoxAct.getBoundingBox().getMax()[2]
            // 	     << endl;

            SbBox3f bb = bBoxAct.getBoundingBox();
            sv->pm_->setSize(bb);
            InvPlaneMover::selectionCB(sv->pm_, selectedObject);
        }

        // now, turn off the sel/desel callbacks.
        // we'll turn them on again after we've adjusted the selection
        sv->selection->removeSelectionCallback(InvCoviseViewer::selectionCallback, (void *)sv);
        sv->selection->removeDeselectionCallback(InvCoviseViewer::deselectionCallback, (void *)sv);
        sv->selectionCallbackInactive = TRUE;

        // attach the manip
        sv->attachManip(sv->curManip, selectedObject);

        //
        // every time an object gets selected we should check if the spacemouse
        // was successfully created at startup time, if not we can do this now
        //

        //
        // If active, attach editors to new selection.
        //
        SoMaterial *mtl = NULL;
        if (sv->materialEditor && sv->materialEditor->isVisible())
        {
            mtl = sv->findMaterialForAttach(selectedObject);
            if (mtl != NULL)
                sv->materialEditor->attach(mtl);
        }

        if (sv->colorEditor && sv->colorEditor->isVisible())
        {
            if (mtl == NULL)
                mtl = sv->findMaterialForAttach(selectedObject);
            if (mtl != NULL)
                sv->colorEditor->attach(&(mtl->diffuseColor), 0, mtl);
        }

        if (sv->transformSliderSet && sv->transformSliderSet->isVisible())
        {
            SoPath *editTransformPath;
            editTransformPath = sv->findTransformForAttach(selectedObject);
            if (editTransformPath == NULL)
            {
                sv->transformSliderSet->setNode(NULL);
            }
            else
            {
                editTransformPath->ref();
                sv->transformSliderSet->setNode(((SoFullPath *)editTransformPath)->getTail());
                editTransformPath->unref();
            }
        }

        // get the object-names from the selected objects
        num = sv->selection->getNumSelected();
        selectedObjectsList = sv->selection->getList();

        nodeList = new SoNodeList(num);
        pathList = new SoPathList(num);

        int i;
        for (i = 0; i < num; i++)
        {

            sv->findObjectName(&objName[0], (*selectedObjectsList)[i]);

            if (objName != NULL)
            {

                // find the transform nodes and attach the sensors:
                // The last selected object is the current selected object
                editTransformPath = sv->findTransformForAttach((*selectedObjectsList)[i]);
                if (editTransformPath != NULL)
                {
                    if (i < num - 1)
                    {
                        nodeList->append((SoTransform *)(editTransformPath->getTail()));
                        pathList->append(editTransformPath);
                    }
                    else
                    {
                        sv->currTransformPath = editTransformPath;
                        sv->currTransformNode = (SoTransform *)(editTransformPath->getTail());
                        sv->transformSensor->attach(sv->currTransformNode);
                    }
                }
                else
                    print_comment(__LINE__, __FILE__, "ERROR: no object for selection found in sectionCB");
            }
        }
        sv->transformNode = nodeList;
        sv->transformPath = pathList;

        // enable/disable cmd key shortcuts and menu items
        sv->updateCommandAvailability();

        // update the object list
        sv->updateObjectList(selectedObject, (SbBool)TRUE);

        // send selection info to other renderer's for all selected objects
        sv->sendSelection((char *)"DESELECT");
        for (i = 0; i < num; i++)
        {
            string = (*selectedObjectsList)[i]->getTail()->getName();
            strcpy(objName, string.getString());
            // cerr << "InvCoviseViewer::selectionCallback: Selected object is : " << objName << endl;
            sv->sendSelection(objName);
        }

        // turn on callbacks again
        sv->selection->addSelectionCallback(InvCoviseViewer::selectionCallback, (void *)sv);
        sv->selection->addDeselectionCallback(InvCoviseViewer::deselectionCallback, (void *)sv);
        sv->selectionCallbackInactive = FALSE;
    }

    //===================================================================
    //
    //  Remove selected objects from the scene graph.
    //  In this demo, we don't really know how the graphs are set up,
    //  so act conservatively, and simply remove the node which is the
    //  tail of the path from its parent. Note if the node is instanced,
    //  all instances will be destroyed. Then travel up the path to a
    //  parent separator. If there are no other shapes under the separator,
    //  destroy it too.
    //
    //  Other applications might delete selected objects a different way,
    //  depending on how the data is organized in the scene graph.
    //
    //  Use: protected
    //
    void
    InvCoviseViewer::destroySelectedObjects()
    //
    //===================================================================
    {
        for (int i = selection->getNumSelected() - 1; i >= 0; i--)
        {
            SoPath *p = (*selection)[i];
            p->ref();

            // Deselect this path
            selection->deselect(i);

            // Remove the tail node from the graph
            SoGroup *g = (SoGroup *)p->getNodeFromTail(1);
            g->removeChild(p->getTail());

            // Travel up the path to separators, and see if this was
            // the only shape node under the sep. If so, delete the sep too.
            // (Don't go all the way up to the selection node).
            SbBool shapeFound = FALSE;
            int j = 0;
            while ((!shapeFound) && (j < p->getLength() - 1))
            {
                SoNode *n = p->getNodeFromTail(j);
                if (n->isOfType(SoSeparator::getClassTypeId()))
                {
                    // Search for other shape nodes
                    SoSearchAction sa;
                    sa.setFind(SoSearchAction::TYPE);
                    sa.setType(SoShape::getClassTypeId());
                    sa.apply(n);

                    // If no other shapes under this separator, delete it!
                    if (sa.getPath() == NULL)
                    {
                        g = (SoGroup *)p->getNodeFromTail(j + 1);
                        g->removeChild(n);

                        // Reset j since we have a new end of path
                        j = 0;
                    }
                    else
                        shapeFound = TRUE;
                }
                // Else a group with no children?
                else if (n->isOfType(SoGroup::getClassTypeId()) && (((SoGroup *)n)->getNumChildren() == 0))
                {
                    g = (SoGroup *)p->getNodeFromTail(j + 1);
                    g->removeChild(n);

                    // Reset j since we have a new end of path
                    j = 0;
                }
                // Else continue up the path looking for separators
                else
                    j++;
            }

            p->unref();
        }
    }

    //======================================================================
    //
    //  This enables/disables cmd key shortcuts and menu items
    //  based on whether there are any objects, and/or any selected objects
    //  in the scene graph.
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::updateCommandAvailability()
    {
        Arg args[1];

        // enable/disable based on the number of child objects in scene
        if (selection->getNumChildren() == 0)
            XtSetArg(args[0], XmNsensitive, False);
        else
            XtSetArg(args[0], XmNsensitive, True);

        // save (if no children, nothing to save)
        XtSetValues(menuItems[SV_FILE_SAVE].widget, args, 1);
        XtSetValues(menuItems[SV_FILE_SAVE_AS].widget, args, 1);

        // enable/disable based on the number of selected objects
        if (selection->getNumSelected() == 0)
            XtSetArg(args[0], XmNsensitive, False);
        else
            XtSetArg(args[0], XmNsensitive, True);

        // if nothing selected, then cannot pick parent, cut, copy, delete,
        // view selection, bring up editors
        //    XtSetValues(menuItems[SV_EDIT_PICK_PARENT].widget, args, 1);
        //    XtSetValues(menuItems[SV_EDIT_CUT].widget, args, 1);
        //    XtSetValues(menuItems[SV_EDIT_COPY].widget, args, 1);
        //    XtSetValues(menuItems[SV_EDIT_DELETE].widget, args, 1);
        XtSetValues(menuItems[SV_VIEW_SELECTION].widget, args, 1);
        XtSetValues(menuItems[SV_EDITOR_TRANSFORM].widget, args, 1);
        XtSetValues(menuItems[SV_EDITOR_MATERIAL].widget, args, 1);
        XtSetValues(menuItems[SV_EDITOR_COLOR].widget, args, 1);
    }

    //======================================================================
    //
    //  This enables/disables cmd key shortcuts and menu items
    //  based on whether there are any objects, and/or any selected objects
    //  in the scene graph.
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setMasterSlaveMenu(short type)
    {
        Arg args[1];

        if (type == FALSE)
            XtSetArg(args[0], XmNsensitive, False);
        else
            XtSetArg(args[0], XmNsensitive, True);

        if (topbarMenuWidget != NULL)
        {
            XtSetValues(topbarMenuWidget, args, 1);
            updateMasterLabel(master);

            // destroy all slave stuff
            if (type == FALSE && sync_flag > SYNC_LOOSE)
            {
                //
                // we are slave now in a still thight coupled session
                //
                XtSetArg(args[0], XmNsensitive, False);
                XtSetValues(topbarMenuWidget, args, 1);
                XtSetValues(menuItems[SV_FILE].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_VIEW].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_MANIP].widget, args, 1);
                XtSetValues(menuItems[SV_LIGHT].widget, args, 1);
                XtSetValues(menuItems[SV_SYNC].widget, args, 1);
                detachManipFromAll();
                removeEditors();
                updateSyncLabel();
#ifndef _AIRBUS
                updateColormapListSensitivity(type, sync_flag);
#endif
                updateObjectListSensitivity(type, sync_flag);
            }
            if (type == TRUE && sync_flag > SYNC_LOOSE)
            {
                //
                // we are master now in a still thight coupled session
                //
                XtSetArg(args[0], XmNsensitive, True);
                XtSetValues(topbarMenuWidget, args, 1);
                XtSetValues(menuItems[SV_FILE].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_VIEW].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_MANIP].widget, args, 1);
                XtSetValues(menuItems[SV_LIGHT].widget, args, 1);
                XtSetValues(menuItems[SV_SYNC].widget, args, 1);
                updateSyncLabel();
                updateMasterLabel(master);
                showEditors();
                updateObjectView();
#ifndef _AIRBUS
                updateColormapListSensitivity(type, sync_flag);
#endif
                updateObjectListSensitivity(type, sync_flag);
            }
            if (type == TRUE && sync_flag == SYNC_LOOSE)
            {
                //
                // we are master now in a still loosely coupled session
                //
                XtSetArg(args[0], XmNsensitive, True);
                XtSetValues(topbarMenuWidget, args, 1);
                XtSetValues(menuItems[SV_FILE].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_VIEW].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_MANIP].widget, args, 1);
                XtSetValues(menuItems[SV_LIGHT].widget, args, 1);
                XtSetValues(menuItems[SV_SYNC].widget, args, 1);
                updateSyncLabel();
                updateMasterLabel(master);
                showEditors();
                updateObjectView();
#ifndef _AIRBUS
                updateColormapListSensitivity(type, sync_flag);
#endif
                updateObjectListSensitivity(type, sync_flag);
            }
            if (type == FALSE && sync_flag == SYNC_LOOSE)
            {
                //
                // we are slave now in a still loosely coupled session
                //
                XtSetArg(args[0], XmNsensitive, True);
                XtSetValues(topbarMenuWidget, args, 1);
                XtSetValues(menuItems[SV_FILE].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_VIEW].widget, args, 1);
                XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
                XtSetValues(menuItems[SV_MANIP].widget, args, 1);
                XtSetValues(menuItems[SV_LIGHT].widget, args, 1);
                XtSetArg(args[0], XmNsensitive, False);
                XtSetValues(menuItems[SV_SYNC].widget, args, 1);
                updateSyncLabel();
                updateMasterLabel(master);
#ifndef _AIRBUS
                updateColormapListSensitivity(type, sync_flag);
#endif
                updateObjectListSensitivity(type, sync_flag);
                // still shown because we were master before
                //showEditors();
                //updateObjectView();
            }
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR :setMasterSlaveMenu : no topbarMenuWidget");
    }

    //======================================================================
    //
    //  sets the menu for a loose/tight master/slave affair
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setSyncMode(int flag)
    {
        Arg args[1];

        //
        // the sync mode has changed not the master slave relation
        //

        if (master == FALSE && flag == FALSE)
        {
            XtSetArg(args[0], XmNsensitive, True);
            XtSetValues(topbarMenuWidget, args, 1);
            XtSetValues(menuItems[SV_FILE].widget, args, 1);
            XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
            XtSetValues(menuItems[SV_VIEW].widget, args, 1);
            XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
            XtSetValues(menuItems[SV_MANIP].widget, args, 1);
            XtSetValues(menuItems[SV_LIGHT].widget, args, 1);
            XtSetArg(args[0], XmNsensitive, False);
            XtSetValues(menuItems[SV_SYNC].widget, args, 1);
            showEditors();
            updateSyncLabel();
            updateColormapListSensitivity(master, flag);
            updateObjectListSensitivity(master, flag);
        }
        else if (master == FALSE && flag == TRUE)
        {
            XtSetArg(args[0], XmNsensitive, False);
            XtSetValues(topbarMenuWidget, args, 1);
            XtSetValues(menuItems[SV_FILE].widget, args, 1);
            XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
            XtSetValues(menuItems[SV_VIEW].widget, args, 1);
            XtSetValues(menuItems[SV_EDITOR].widget, args, 1);
            XtSetValues(menuItems[SV_MANIP].widget, args, 1);
            XtSetValues(menuItems[SV_LIGHT].widget, args, 1);
            XtSetValues(menuItems[SV_SYNC].widget, args, 1);
            detachManipFromAll();
            removeEditors();
            updateSyncLabel();
            updateObjectView();
            updateColormapListSensitivity(master, flag);
            updateObjectListSensitivity(master, flag);
        }
        else if (master == TRUE && flag == TRUE)
        {
            // we are the master so update the slaves
            updateSyncLabel();
            updateObjectView();
            updateColormapListSensitivity(master, flag);
            updateObjectListSensitivity(master, flag);
        }
        else
        {
            updateSyncLabel();
        }
    }

    //======================================================================
    //
    //  update colormap list regarding master/slave and coupling
    //
    //  Use: private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::updateColormapListSensitivity(int master_mode, int sync_mode)
    {
        char buffer[255];

        // cerr << "InvCoviseViewer::updateColormapListSensitivity" << endl;

        if (colormaplist != NULL)
        {

            if (master_mode == TRUE) // enable list, we're master
            {
                XtVaSetValues(colormaplist, XmNsensitive, True, NULL);
                // since we're now master we send our state to the slaves if not LOOSE
                if (sync_mode != SYNC_LOOSE)
                {
                    sprintf(buffer, "%s", colormap_manager->currentColormap());
                    sendColormap(buffer);
                }
            }
            else if (sync_mode == SYNC_LOOSE) // not master but loose coupling: enable
            {
                XtVaSetValues(colormaplist, XmNsensitive, True, NULL);
            }
            else // disable
            {
                XtVaSetValues(colormaplist, XmNsensitive, False, NULL);
            }
        }
    }

    //======================================================================
    //
    //  update object list regarding master/slave and coupling
    //
    //  Use: private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::updateObjectListSensitivity(int master_mode, int sync_mode)
    {
        //char buffer[255];

        // cerr << "InvCoviseViewer::updateObjectListSensitivity" << endl;

        if (objectlist != NULL)
        {

            if (master_mode == TRUE) // enable list, we're master
            {
                XtVaSetValues(objectlist, XmNsensitive, True, NULL);
                // since we're now master we send our state to the slaves if not LOOSE
                // if ( sync_mode != SYNC_LOOSE) {
                //   sprintf(buffer,"%s",colormap_manager->currentColormap());
                //   sendColormap(buffer);
                // }
            }
            else if (sync_mode == SYNC_LOOSE) // not master but loose coupling: enable
            {
                XtVaSetValues(objectlist, XmNsensitive, True, NULL);
            }
            else // disable
            {
                XtVaSetValues(objectlist, XmNsensitive, False, NULL);
            }
        }
    }

    //======================================================================
    //
    //  sets the light mode
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setLightMode(int flag)
    {

        // set light model to BASE_COLOR
        if (flag == SoLightModel::BASE_COLOR)
        {
            lightmodel_state = 0;
            if (currentViewer)
            {
                currentViewer->setLightModelState(lightmodel_state);
                if (textureList)
                {
                    SoTexture2 *tex;
                    int num = textureList->getLength();
                    for (int n = 0; n < num; n++)
                    {
                        tex = (SoTexture2 *)(*(textureList))[n];
                        tex->model.setValue(SoTexture2::DECAL);
                    }
                }
            }
            lightmodel->model = SoLightModel::BASE_COLOR;
            TOGGLE_ON(menuItems[SV_LIGHT_MODEL].widget);
        }
        else
        {
            lightmodel_state = 1;
            if (currentViewer)
            {
                currentViewer->setLightModelState(lightmodel_state);
                if (textureList)
                {
                    SoTexture2 *tex;
                    int num = textureList->getLength();
                    for (int n = 0; n < num; n++)
                    {
                        tex = (SoTexture2 *)(*(textureList))[n];
                        tex->model.setValue(SoTexture2::MODULATE);
                    }
                }
            }
            lightmodel->model = SoLightModel::PHONG;
            TOGGLE_OFF(menuItems[SV_LIGHT_MODEL].widget);
        }
    }

    //======================================================================
    //
    //  sets the selection
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setSelection(char *name)
    {
        SoSearchAction action;

        if (name != NULL)
        {
            // first, detach manips from all selected objects
            detachManipFromAll();

            // now, turn off the sel/desel callbacks.
            // we'll turn them on again after we've adjusted the selection
            selection->removeSelectionCallback(InvCoviseViewer::selectionCallback, this);
            selection->removeDeselectionCallback(InvCoviseViewer::deselectionCallback, this);
            selectionCallbackInactive = TRUE;

            // deselect all if special message arrives,
            // otherwise build up a selection from the pasted paths
            if (strcmp(name, "DESELECT") == 0)
                selection->deselectAll();
            else
            {

                action.setFind(SoSearchAction::NAME);
                action.setName(SbName(name));
                action.setInterest(SoSearchAction::FIRST);

                action.setSearchingAll(TRUE);
                action.apply(selection);
                //cerr << "I found the node to select. His name is : " << name << endl;

                selection->select(action.getPath());

                // update the object list item
                updateObjectListItem(TRUE, name);

                selection->addSelectionCallback(InvCoviseViewer::selectionCallback, this);
                selection->addDeselectionCallback(InvCoviseViewer::deselectionCallback, this);
                selectionCallbackInactive = FALSE;
            }
        }
        else
            print_comment(__LINE__, __FILE__, "Name of object to select is NULL");
    }

    //======================================================================
    //
    //  sets the deselection
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void
    InvCoviseViewer::setDeselection(char *name)
    {
        SoSearchAction action;

        // first, detach manips from all selected objects
        detachManipFromAll();

        if (name)
        {
            // now, turn off the sel/desel callbacks.
            // we'll turn them on again after we've adjusted the selection
            selection->removeSelectionCallback(InvCoviseViewer::selectionCallback, this);
            selection->removeDeselectionCallback(InvCoviseViewer::deselectionCallback, this);
            selectionCallbackInactive = TRUE;

            //action.setFind(SoSearchAction::NAME);
            action.setName(SbName(name));
            action.setInterest(SoSearchAction::FIRST);
            action.setSearchingAll(TRUE);

            action.apply(selection);
            //cerr << "I found the node to deselect. His name is : " << name << endl;

            selection->deselect(action.getPath());

            // update the object list item
            updateObjectListItem(FALSE, name);

            selection->addSelectionCallback(InvCoviseViewer::selectionCallback, this);
            selection->addDeselectionCallback(InvCoviseViewer::deselectionCallback, this);
            selectionCallbackInactive = FALSE;
        }
    }

    //======================================================================
    //
    //  sets the transparency level
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setTransparency(char *message)
    {
#ifdef _COLLAB_VIEWER
        if (strstr(message, "tiff"))
        {
            snap(message);
        }
        else
#endif
        {
            int type;

            if (sscanf(message, "%d", &type) != 1)
            {
                fprintf(stderr, "InvCoviseViewer::setTransparency: sscanf failed\n");
            }

            currentViewer->setTransparencyType((SoGLRenderAction::TransparencyType)type);
        }
    }

    //======================================================================
    //
    //  sets the axis on or off
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setAxis(int onoroff)
    {

        axis_state = onoroff;
        if (onoroff == CO_OFF)
            axis_switch->whichChild.setValue(SO_SWITCH_NONE);
        else
            axis_switch->whichChild.setValue(0);
    }

    //======================================================================
    //
    //  sets clipping on or off
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setClipping(int onoroff)
    {

        clipState = onoroff;
        if (onoroff == CO_OFF)
            clipSwitch->whichChild.setValue(SO_SWITCH_NONE);
        else
            clipSwitch->whichChild.setValue(0);
    }

    //======================================================================
    //
    //  sets the clipping plane equation
    //
    //  Use: static private
    //
    //
    //
    //======================================================================
    void InvCoviseViewer::setClippingPlane(double equation[])
    {
        eqn[0] = equation[0];
        eqn[1] = equation[1];
        eqn[2] = equation[2];
        eqn[3] = equation[3];
        clipSwitch->whichChild.setValue(0);
    }

    //======================================================================
    //
    //  Called by Xt when a menu is about to be displayed.
    //  This gives us a chance to update any items in the menu.
    //
    //  Use: static private
    //
    //
    //======================================================================
    void InvCoviseViewer::menuDisplay(Widget, InvCoviseViewerData * data, XtPointer)
    {
        InvCoviseViewer *sv = data->classPt;
        Arg args[1];
        char str[MAXPATHLEN + 21];
        XmString xmstr;

        switch (data->id)
        {
        case SV_FILE:
            // disable saving if there isn't any geometry
            if (sv->selection->getNumChildren() == 0)
                XtSetArg(args[0], XmNsensitive, False);
            else
                XtSetArg(args[0], XmNsensitive, True);

            XtSetValues(sv->menuItems[SV_FILE_SAVE].widget, args, 1);
            XtSetValues(sv->menuItems[SV_FILE_SAVE_AS].widget, args, 1);

            // disable copy if there is no selection
            if (sv->selection->getNumSelected() == 0)
                XtSetArg(args[0], XmNsensitive, False);
            else
                XtSetArg(args[0], XmNsensitive, True);
            XtSetValues(sv->menuItems[SV_FILE_COPY].widget, args, 1);

            // update the "Save" menu entry to reflect the current file name
            strcpy(str, "Save");
            if (sv->fileName != NULL)
            {
                // get the file name withought the entire path
                // last occurance of '/'
                char *pt = strrchr(sv->fileName, '/');
                pt = (pt == NULL) ? sv->fileName : pt + 1;
                strcat(str, " -> ");
                strcat(str, pt);
            }
            xmstr = XmStringCreate(str, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
            XtSetArg(args[0], XmNlabelString, xmstr);
            XtSetValues(sv->menuItems[SV_FILE_SAVE].widget, args, 1);
            //??? can we ever free the xmstr?
            break;

        case SV_VIEW:
            // set pick/edit toggle
            if (sv->isViewing())
                TOGGLE_OFF(sv->menuItems[SV_VIEW_PICK].widget);
            else
                TOGGLE_ON(sv->menuItems[SV_VIEW_PICK].widget);

            // set user pick toggle
            if (sv->userModeFlag)
                TOGGLE_ON(sv->menuItems[SV_VIEW_USER].widget);
            else
                TOGGLE_OFF(sv->menuItems[SV_VIEW_USER].widget);

            // set the correct transparency type
            TOGGLE_OFF(sv->menuItems[SV_VIEW_SCREEN_TRANSPARENCY].widget);
            TOGGLE_OFF(sv->menuItems[SV_VIEW_BLEND_TRANSPARENCY].widget);
            TOGGLE_OFF(sv->menuItems[SV_VIEW_DELAY_BLEND_TRANSPARENCY].widget);
            TOGGLE_OFF(sv->menuItems[SV_VIEW_SORT_BLEND_TRANSPARENCY].widget);
            switch (sv->getTransparencyType())
            {
            case SoGLRenderAction::SCREEN_DOOR:
                TOGGLE_ON(sv->menuItems[SV_VIEW_SCREEN_TRANSPARENCY].widget);
                break;
            case SoGLRenderAction::BLEND:
                TOGGLE_ON(sv->menuItems[SV_VIEW_BLEND_TRANSPARENCY].widget);
                break;
            case SoGLRenderAction::DELAYED_BLEND:
                TOGGLE_ON(sv->menuItems[SV_VIEW_DELAY_BLEND_TRANSPARENCY].widget);
                break;
            case SoGLRenderAction::SORTED_OBJECT_BLEND:
                TOGGLE_ON(sv->menuItems[SV_VIEW_SORT_BLEND_TRANSPARENCY].widget);
                break;
            default:
                fprintf(stderr, "strange value for sv->getTransparencyType()\n");
                break;
            }

            // disable view selection if nothing is selected
            if (sv->selection->getNumSelected() == 0)
                XtSetArg(args[0], XmNsensitive, False);
            else
                XtSetArg(args[0], XmNsensitive, True);
            XtSetValues(sv->menuItems[SV_VIEW_SELECTION].widget, args, 1);

            // set fog toggle
            if (sv->fogFlag)
                TOGGLE_ON(sv->menuItems[SV_VIEW_FOG].widget);
            else
                TOGGLE_OFF(sv->menuItems[SV_VIEW_FOG].widget);

            // set antialiasing toggle
            if (sv->antialiasingFlag)
                TOGGLE_ON(sv->menuItems[SV_VIEW_ANTIALIASING].widget);
            else
                TOGGLE_OFF(sv->menuItems[SV_VIEW_ANTIALIASING].widget);

            // show the axis
            if (sv->axis_state == CO_ON)
                TOGGLE_OFF(sv->menuItems[SV_VIEW_AXIS].widget);
            else
                TOGGLE_ON(sv->menuItems[SV_VIEW_AXIS].widget);

            break;

        case SV_EDITOR:
            // disable items if there is no selection
            if (sv->selection->getNumSelected() == 0)
                XtSetArg(args[0], XmNsensitive, False);
            else
                XtSetArg(args[0], XmNsensitive, True);

            XtSetValues(sv->menuItems[SV_EDITOR_TRANSFORM].widget, args, 1);
            XtSetValues(sv->menuItems[SV_EDITOR_MATERIAL].widget, args, 1);
            XtSetValues(sv->menuItems[SV_EDITOR_COLOR].widget, args, 1);

            // part editor: disable items if not being master
            if (sv->isMaster())
                XtSetArg(args[0], XmNsensitive, True);
            else
                XtSetArg(args[0], XmNsensitive, False);

            XtSetValues(sv->menuItems[SV_EDITOR_PARTS].widget, args, 1);

            if (sv->handleState_)
            {
                XtSetArg(args[0], XmNsensitive, False);
            }
            else
            {
                XtSetArg(args[0], XmNsensitive, True);
            }

            XtSetValues(sv->menuItems[SV_EDITOR_SNAPH].widget, args, 1);

            if (sv->handleState_)
            {
                XtSetArg(args[0], XmNsensitive, True);
            }
            else
            {
                XtSetArg(args[0], XmNsensitive, False);
            }

            XtSetValues(sv->menuItems[SV_EDITOR_FREEH].widget, args, 1);

            break;

        case SV_MANIP:

            // First, the section with the different types of manipulators.
            TOGGLE_OFF(sv->menuItems[SV_MANIP_HANDLEBOX].widget);
            TOGGLE_OFF(sv->menuItems[SV_MANIP_TRACKBALL].widget);
            TOGGLE_OFF(sv->menuItems[SV_MANIP_JACK].widget);
            TOGGLE_OFF(sv->menuItems[SV_MANIP_CENTERBALL].widget);
            TOGGLE_OFF(sv->menuItems[SV_MANIP_XFBOX].widget);
            TOGGLE_OFF(sv->menuItems[SV_MANIP_TABBOX].widget);
            TOGGLE_OFF(sv->menuItems[SV_MANIP_NONE].widget);

            // Turn appropriate radio button on
            if (sv->curManip == SV_HANDLEBOX)
                TOGGLE_ON(sv->menuItems[SV_MANIP_HANDLEBOX].widget);
            else if (sv->curManip == SV_TRACKBALL)
                TOGGLE_ON(sv->menuItems[SV_MANIP_TRACKBALL].widget);
            else if (sv->curManip == SV_JACK)
                TOGGLE_ON(sv->menuItems[SV_MANIP_JACK].widget);
            else if (sv->curManip == SV_CENTERBALL)
                TOGGLE_ON(sv->menuItems[SV_MANIP_CENTERBALL].widget);
            else if (sv->curManip == SV_XFBOX)
                TOGGLE_ON(sv->menuItems[SV_MANIP_XFBOX].widget);
            else if (sv->curManip == SV_TABBOX)
                TOGGLE_ON(sv->menuItems[SV_MANIP_TABBOX].widget);
            else
                TOGGLE_ON(sv->menuItems[SV_MANIP_NONE].widget);

            // Next, the toggle that says whether we replace current
            // manipulators every time we change the type given in the menu.
            if (sv->curManipReplaces == TRUE)
                TOGGLE_ON(sv->menuItems[SV_MANIP_REPLACE_ALL].widget);
            else
                TOGGLE_OFF(sv->menuItems[SV_MANIP_REPLACE_ALL].widget);

            break;

        case SV_LIGHT:
            // set lightmodel to BASE_COLOR
            if (sv->lightmodel_state == 0)
                TOGGLE_ON(sv->menuItems[SV_LIGHT_MODEL].widget);
            else
                TOGGLE_OFF(sv->menuItems[SV_LIGHT_MODEL].widget);

            // disable the add light entries if we have more than 8 lights
            if (sv->lightDataList.getLength() < 8)
                XtSetArg(args[0], XmNsensitive, True);
            else
                XtSetArg(args[0], XmNsensitive, False);

            XtSetValues(sv->menuItems[SV_LIGHT_ADD_DIRECT].widget, args, 1);
            XtSetValues(sv->menuItems[SV_LIGHT_ADD_POINT].widget, args, 1);
            XtSetValues(sv->menuItems[SV_LIGHT_ADD_SPOT].widget, args, 1);

            // update the headlight label (show on/off with '*')
            sv->isHeadlight() ? strcpy(str, "* ") : strcpy(str, "  ");
            strcat(str, sv->headlightData->name);
            xmstr = XmStringCreate(str, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
            XtSetArg(args[0], XmNlabelString, xmstr);
            XtSetValues(sv->headlightData->cascadeWidget, args, 1);
            XmStringFree(xmstr);

            // update the lights label (show on/off with '*')
            {
                for (int i = 0; i < sv->lightDataList.getLength(); i++)
                {
                    InvLightData *data = (InvLightData *)sv->lightDataList[i];
                    IS_LIGHT_ON(data->lightSwitch) ? strcpy(str, "* ") : strcpy(str, "  ");
                    strcat(str, data->name);
                    xmstr = XmStringCreate(str, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
                    XtSetArg(args[0], XmNlabelString, xmstr);
                    XtSetValues(data->cascadeWidget, args, 1);
                    XmStringFree(xmstr);
                }
            }
            break;
        /*
                  case SV_COLORMAP:
                        {

                     // First, the section with the different types of colormap positions.
                     TOGGLE_OFF( sv->menuItems[SV_COLORMAP_BOTTOM_LEFT].widget );
                     TOGGLE_OFF( sv->menuItems[SV_COLORMAP_TOP_LEFT].widget );
                     TOGGLE_OFF( sv->menuItems[SV_COLORMAP_TOP_RIGHT].widget );
                     TOGGLE_OFF( sv->menuItems[SV_COLORMAP_BOTTOM_RIGHT].widget );

            // Turn appropriate radio button on
            if (sv->cmapPosition == COLORMAP_BOTTOM_LEFT)
            TOGGLE_ON(sv->menuItems[SV_COLORMAP_BOTTOM_LEFT].widget);
            else if (sv->cmapPosition == COLORMAP_TOP_LEFT)
            TOGGLE_ON(sv->menuItems[SV_COLORMAP_TOP_LEFT].widget);
            else if (sv->cmapPosition == COLORMAP_TOP_RIGHT)
            TOGGLE_ON(sv->menuItems[SV_COLORMAP_TOP_RIGHT].widget);
            else if (sv->cmapPosition == COLORMAP_BOTTOM_RIGHT)
            TOGGLE_ON(sv->menuItems[SV_COLORMAP_BOTTOM_RIGHT].widget);

            // update the colormap buttons
            InvObjectList *colList = sv->colormap_manager->getColormapList();
            colList->resetToFirst();
            for (int k=0; k < sv->colormap_manager->getNumColormaps(); k++) {

            InvObject *lobj = colList->getNextObject();
            if (strcmp(lobj->getName(),sv->colormap_manager->currentColormap())==NULL) {
            // toggle on
            //cerr << "Toggling on " << lobj->getName() << endl;
            }
            else {
            // toggle off
            //cerr << "Toggling off " << lobj->getName() << endl;
            }

            }
            }
            break;
            */
        case SV_SYNC:
            // mirror the selection policy
            TOGGLE_OFF(sv->menuItems[SV_SYNC_LOOSE].widget);
            TOGGLE_OFF(sv->menuItems[SV_SYNC_MA_SL].widget);
            TOGGLE_OFF(sv->menuItems[SV_SYNC_TIGHT].widget);

            if (sv->sync_flag == SYNC_LOOSE)
                TOGGLE_ON(sv->menuItems[SV_SYNC_LOOSE].widget);
            else if (sv->sync_flag == SYNC_SYNC)
                TOGGLE_ON(sv->menuItems[SV_SYNC_MA_SL].widget);
            else
                TOGGLE_ON(sv->menuItems[SV_SYNC_TIGHT].widget);
            break;

        default:
            break;
        }
    }

    //======================================================================
    //
    // Description:
    //    Determines whether a given node is affected by a transform.
    //
    // Use: static, public
    //
    //
    //======================================================================
    SbBool InvCoviseViewer::isAffectedByTransform(
        SoNode * theNode) // node to be affected?
    {
        if (theNode->isOfType(SoGroup::getClassTypeId())
            || theNode->isOfType(SoShape::getClassTypeId())
                   //!!	    || theNode->isOfType( SoCamera::getClassTypeId() )
            || theNode->isOfType(SoPerspectiveCamera::getClassTypeId())
            || theNode->isOfType(SoLight::getClassTypeId()))
        {
            return TRUE;
        }
        return FALSE;
    }

    //======================================================================
    //
    // Description:
    //    Determines whether a given node is affected by material node.
    //
    // Use: static, public
    //
    //
    //======================================================================
    SbBool InvCoviseViewer::isAffectedByMaterial(
        SoNode * theNode) // node to be affected?
    {
        if (theNode->isOfType(SoGroup::getClassTypeId())
            || theNode->isOfType(SoShape::getClassTypeId()))
        {
            return TRUE;
        }
        return FALSE;
    }

    //======================================================================
    //
    // Description:
    //	Create the lights and camera environment structure.
    //
    // Use: private
    //
    //
    //======================================================================
    void InvCoviseViewer::createLightsCameraEnvironment()
    {
        // Group {
        //	  Label { "InvCoviseViewer Environment v2.1" }
        //    Camera {}
        //    Environment {}
        //    Group {
        //        Switch { Light 1 }    	# switch is child 0, light is child 0
        //        Switch { Light 2 }    	# switch is child 1, light is child 0
        //        ...
        //    }
        // }
        //
        // NOTE: since the camera may be switched by the viewer (ortho/perspective toggle)
        // make sure to get the camera from the viewer (and not cache the camera).

        lightsCameraEnvironment = new SoGroup;
        camera = new SoPerspectiveCamera;
        environment = new SoEnvironment;
        lightGroup = new SoGroup;
        envLabel = new SoLabel;

        envLabel->label.setValue(SV_ENV_LABEL);
        lightsCameraEnvironment->addChild(envLabel);
        lightsCameraEnvironment->addChild(camera);
        lightsCameraEnvironment->addChild(environment);
        lightsCameraEnvironment->addChild(lightGroup);

        //
        // add the data sensor for the camera change reporting
        //
        cameraSensor = new SoNodeSensor(
            InvCoviseViewer::cameraCallback, this);
        cameraSensor->attach(camera);
    }

    //======================================================================
    //
    // Description:
    //	get ans set a new transformation in a transform node
    //
    // Use: private, friend
    //
    //======================================================================
    void InvCoviseViewer::receiveTransformation(char *message)
    {

        if (master == FALSE)
        {
            char name[255];
            float scale[3];
            float trans[3];
            float cen[3];
            float rot[4];

            int ret = sscanf(message, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f", &name[0],
                             &scale[0], &scale[1], &scale[2],
                             &trans[0], &trans[1], &trans[2],
                             &cen[0], &cen[1], &cen[2],
                             &rot[0], &rot[1], &rot[2], &rot[3]);
            if (ret != 14)
            {
                fprintf(stderr, "InvCoviseViewer::receiveTransformation: sscanf failed\n");
            }

            // set the transformation in the correct node
            if (strcmp(name, "") != 0)
            {
                SoTransform *transform = findTransformNode(&name[0]);

                if (transform != NULL)
                {
                    transform->translation.setValue(trans[0], trans[1], trans[2]);
                    transform->center.setValue(cen[0], cen[1], cen[2]);
                    transform->scaleFactor.setValue(scale[0], scale[1], scale[2]);
                    transform->rotation.setValue(rot[0], rot[1], rot[2], rot[3]);
                }
            }
            else
                print_comment(__LINE__, __FILE__, "ERROR :got garbage object name !");
        }

#ifdef TIMING
        time_str = new char[100];
        sprintf(time_str, "%s: ...TRANSFORM[%d] drawn", ap->get_name(), transform_receive_ctr - 1);
        covise_time->mark(__LINE__, time_str);
#endif
    }

    //======================================================================
    //
    // Description:
    //	Collect transformation change stuff
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::sendTransformation(const char *name, SoTransform *transform)
    {

        if (master == TRUE && sync_flag > SYNC_LOOSE)
        {

            if (transform != NULL)
            { // do not change this routine and NOT the line above

                char message[255];
                SbVec3f scaleFactor;
                float scale[3];
                SbVec3f translation;
                float trans[3];
                SbVec3f center;
                float cen[3];
                SbRotation rotation;
                float rot[4];

                translation = transform->translation.getValue();
                center = transform->center.getValue();
                scaleFactor = transform->scaleFactor.getValue();
                rotation = transform->rotation.getValue();

                scaleFactor.getValue(scale[0], scale[1], scale[2]);
                translation.getValue(trans[0], trans[1], trans[2]);
                center.getValue(cen[0], cen[1], cen[2]);
                rotation.getValue(rot[0], rot[1], rot[2], rot[3]);

                //
                // pack into character string

                sprintf(message, "%s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f",
                        name, scale[0], scale[1], scale[2], trans[0], trans[1], trans[2], cen[0], cen[1], cen[2], rot[0], rot[1], rot[2], rot[3]);

                rm_sendTransformation(message);
            }
        }
    }

    //======================================================================
    //
    // Description:
    //	Collect transform change stuff
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::transformCallback(void *data, SoSensor *)
    {
        InvCoviseViewer *r = (InvCoviseViewer *)data;
        char objName[255];
        int i;
        int num = r->selection->getNumSelected();
        SoTransform *transform = NULL;
        const SoPathList *selectedObjectsList = r->selection->getList();

        // transform all selected objects
        for (i = 0; i < num - 1; i++)
        {
            transform = (SoTransform *)(*(r->transformNode))[i];
            transform->translation.setValue(r->currTransformNode->translation.getValue());
            transform->center.setValue(r->currTransformNode->center.getValue());
            transform->scaleFactor.setValue(r->currTransformNode->scaleFactor.getValue());
            transform->rotation.setValue(r->currTransformNode->rotation.getValue());
        }

        if (r->master == TRUE)
        {
            // print_comment(__LINE__,__FILE__,"RENDERER: transformCallback transform has changed");
            // get the object names from the selected objects
            if (r->currTransformPath != NULL && r->currTransformNode != NULL)
            {
                for (i = 0; i < num; i++)
                {

                    r->findObjectName(&objName[0], (*selectedObjectsList)[i]);
                    if (objName != NULL)
                    {
                        r->sendTransformation(&objName[0], r->currTransformNode);
                    }
                    else
                        print_comment(__LINE__, __FILE__, "ERROR: transformCallback no object found with this name");
                }
            }
        }
    }

    //======================================================================
    //
    // Description:
    //	Collect camera change stuff
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::cameraCallback(void *data, SoSensor *)
    {
        InvCoviseViewer *r = (InvCoviseViewer *)data;

        if (r->master == TRUE && r->sync_flag > SYNC_LOOSE)
        {
            // print_comment(__LINE__,__FILE__,"cameraCallback : camera has changed");

            if (r->viewer_edit_state == CO_OFF) // zoom slider or home button...
            {
                //
                //
                // get the current camera and pass the values to the communication
                // manager
                //
                //      char message[100];
                // CAREFULL: this solves the problem only temporary!!!
                char message[255];
                float pos[3];
                float ori[4];
                int view;
                float aspect;
                float near;
                float far;
                float focal;
                float angleORheightangle; // depends on camera type !

                r->getTransformation(pos, ori, &view, &aspect, &near, &far, &focal, &angleORheightangle);

                //
                // pack into character string

                sprintf(message, "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %5d %7.3f %7.3f %7.3f %7.3f %7.3f",
                        pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3], view, aspect, near, far, focal, angleORheightangle);
                rm_sendCamera(message);
            }
        }
    }

    //======================================================================
    //
    // Description:
    //	set new transformation
    //
    // Use: public, virtual
    //
    //======================================================================
    void InvCoviseViewer::setTransformation(float pos[3], float ori[4], int view,
                                            float aspect, float near, float far,
                                            float focal, float angleORheightangle)

    {

        SoPerspectiveCamera *myPerspCamera = (SoPerspectiveCamera *)currentViewer->getCamera();
        SoOrthographicCamera *myOrthoCamera = (SoOrthographicCamera *)currentViewer->getCamera();
#ifndef _COLLAB_VIEWER
        if (master == FALSE)
#endif
        {

            if (currentViewer->getCameraType() == SoPerspectiveCamera::getClassTypeId())
            {

                float oLen = myPerspCamera->position.getValue().length();
                SbVec3f nPos(pos[0], pos[1], pos[2]);
                float nLen = nPos.length();

                if (oLen != 0.0)
                {
                    Annotations->reScale(nLen / oLen);
                }

                myPerspCamera->heightAngle = angleORheightangle;
                myPerspCamera->position = nPos;
                myPerspCamera->orientation = SbRotation(ori[0], ori[1], ori[2], ori[3]);
                myPerspCamera->viewportMapping = view;
                myPerspCamera->aspectRatio = aspect;
                myPerspCamera->nearDistance = near;
                myPerspCamera->farDistance = far;
                myPerspCamera->focalDistance = focal;
            }
            else
            {
                myOrthoCamera->position = SbVec3f(pos[0], pos[1], pos[2]);
                myOrthoCamera->orientation = SbRotation(ori[0], ori[1], ori[2], ori[3]);
                myOrthoCamera->viewportMapping = view;
                myOrthoCamera->aspectRatio = aspect;
                myOrthoCamera->nearDistance = near;
                myOrthoCamera->farDistance = far;
                myOrthoCamera->focalDistance = focal;
                myOrthoCamera->height = angleORheightangle;
            }
        }

#ifdef TIMING
        time_str = new char[100];
        sprintf(time_str, "%s: ...TRANSFORM[%d] drawn", ap->get_name(), transform_receive_ctr - 1);
        covise_time->mark(__LINE__, time_str);
#endif
    }

    //======================================================================
    //
    // Description:
    //	get actual transformation
    //
    // Use: private
    //
    //======================================================================
    void InvCoviseViewer::getTransformation(float pos[3], float ori[4], int *view,
                                            float *aspect, float *near, float *far,
                                            float *focal, float *angleORheightangle)

    {
        SbVec3f p;
        SbRotation r;

        p = currentViewer->getCamera()->position.getValue();
        r = currentViewer->getCamera()->orientation.getValue();
        *view = currentViewer->getCamera()->viewportMapping.getValue();
        *aspect = currentViewer->getCamera()->aspectRatio.getValue();
        *near = currentViewer->getCamera()->nearDistance.getValue();
        *far = currentViewer->getCamera()->farDistance.getValue();
        *focal = currentViewer->getCamera()->focalDistance.getValue();

        if (currentViewer->getCameraType() == SoPerspectiveCamera::getClassTypeId())
            *angleORheightangle = ((SoPerspectiveCamera *)currentViewer->getCamera())->heightAngle.getValue();
        else
            *angleORheightangle = ((SoOrthographicCamera *)currentViewer->getCamera())->height.getValue();

        p.getValue(pos[0], pos[1], pos[2]);
        r.getValue(ori[0], ori[1], ori[2], ori[3]);
    }

    //
    void InvCoviseViewer::changeCamera(SoCamera * newCamera)
    {
        lightsCameraEnvironment->insertChild(newCamera, 0);
        setCamera(newCamera);
        lightsCameraEnvironment->removeChild(camera);
        camera = newCamera;
    }

    void
    InvCoviseViewer::depthTestCB(void *userData, SoAction *action)
    {
        (void)userData;
        (void)action;
        cerr << "InvCoviseViewer::depthTestCB(..) called" << endl;
        if (Annotations->kbIsActive())
        {
            glPushMatrix();
            glDisable(GL_DEPTH_TEST);
            glPopMatrix();
        }
        else
        {
            glPushMatrix();
            glEnable(GL_DEPTH_TEST);
            glPopMatrix();
        }
    }

    // clipping callback routine
    void
    InvCoviseViewer::clippingCB(void *userData, SoAction *action)
    {
        (void)action;
        InvCoviseViewer *sv = (InvCoviseViewer *)userData;

        glPushMatrix();
        glClipPlane(GL_CLIP_PLANE0, sv->eqn);
        glEnable(GL_CLIP_PLANE0);
        glPopMatrix();
    }

    void InvCoviseViewer::finishClippingCB(void *userData, SoAction *action)
    {
        (void)action;
        InvCoviseViewer *sv = (InvCoviseViewer *)userData;
        if (sv->clipState == CO_ON)
            glDisable(GL_CLIP_PLANE0);
    }

    //======================================================================
    //
    // Description:
    //      Invokes editor to set a clipping plane.
    //
    //======================================================================
    void InvCoviseViewer::editClippingPlane()
    {
        if (clippingPlaneEditor == NULL)
        {
            clippingPlaneEditor = new InvClipPlaneEditor;
            clippingPlaneEditor->setTitle("Clipping Plane");
        }
        clippingPlaneEditor->setViewer(this);

        clippingPlaneEditor->show();
    }

    //======================================================================
    //
    // Description:
    //      Set the clipping plane equation.
    //
    //======================================================================
    void InvCoviseViewer::setClipPlaneEquation(double point[], double normal[])
    {
        eqn[0] = (GLdouble)normal[0];
        eqn[1] = (GLdouble)normal[1];
        eqn[2] = (GLdouble)normal[2];
        eqn[3] = (GLdouble)((-1.0) * (point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2]));
        clipSwitch->whichChild.setValue(0); // only to invoke the necessary traversal of the scene graph

        sendClippingPlane(clipState, eqn);
    }

    //
    // define those generic virtual functions
    //
    const char *
    InvCoviseViewer::getDefaultWidgetName() const
    {
        return getDefaultTitle();
    }

    const char *
    InvCoviseViewer::getDefaultTitle() const
    {
#ifndef _AIRBUS
        return "COVISE Renderer";
#else
    return "NS3D Renderer";
#endif
    }

    const char *
    InvCoviseViewer::getDefaultIconTitle() const
    {
        return getDefaultTitle();
    }

    void
    InvCoviseViewer::unmanageObjs()
    {
        if (mySequencer)
        {
#ifndef _COLLAB_VIEWER
            mySequencer->hide();
#endif
        }

        showMenu(FALSE);
    }

    void
    InvCoviseViewer::manageObjs()
    {

        int rem;
        if (mySequencer)
        {
            rem = mySequencer->getSeqAct();
            mySequencer->activate();
            mySequencer->show();
            mySequencer->setSeqAct(rem);
        }

        showMenu(TRUE);
    }

    int
    InvCoviseViewer::toggleHandleState()
    {
        if (handleState_)
        {
            handleState_ = 0;
            if (pm_)
                pm_->setFreeMotion();
        }
        else
        {
            handleState_ = 1;
            if (pm_)
                pm_->setSnapToAxis();
        }
        return handleState_;
    }

#ifndef _WIN32
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    void x11SnapTIFF(Widget wid, const char *filename)
    {
        // +++++ 1st part: X stuff - get the Widget into an XImage
        //int x=0,y=0;

        Display *dpy = wid ? XtDisplay(wid) : NULL;
        Window window = wid ? XtWindow(wid) : 0;

        XWindowAttributes window_attr;
        if (!XGetWindowAttributes(dpy, window, &window_attr))
        {
            cerr << "Snap to TIFF failed: could not get Window attributes" << endl;
            return;
        }

        int w = window_attr.width - 4; // remove borders
        int h = window_attr.height - 4;

        //int count;
        //XPixmapFormatValues *pmForm = XListPixmapFormats(dpy,&count);

        XImage *ximg = XGetImage(dpy, window, 2, 2, w, h, 0xffffffff, ZPixmap);

        // +++++ 2nd part: get the XImage into a buffer in correct RGB order

        // bytes per Pixel
        if ((ximg->bits_per_pixel & 0x7) || (ximg->bits_per_pixel > 32))
        {
            cerr << "Cannot do snapshots of visuals with "
                 << ximg->bits_per_pixel << " bits per pixel" << endl;
            return;
        }
        int bytesPerPixel = ximg->bits_per_pixel >> 3;

        // if we have MSB byte-order we flip everything: store it in bytesPerPixel
        if (ximg->bitmap_bit_order == MSBFirst)
            bytesPerPixel += 4; // later we use a CASE statement on bytesPerPixel

        // get the masks for R/G/B : we have to AND the pixel with xMask
        //                           to get x color component
        unsigned long rMask = ximg->red_mask;
        unsigned long gMask = ximg->green_mask;
        unsigned long bMask = ximg->blue_mask;

        // after AND operation shift result by xShift bits to right
        int rShift = 0;
        while (((rMask >> rShift) & 1) == 0)
            rShift++;

        int gShift = 0;
        while (((gMask >> gShift) & 1) == 0)
            gShift++;

        int bShift = 0;
        while (((bMask >> bShift) & 1) == 0)
            bShift++;

        // if we use <8 bits per color we shift the result to upper bits

        unsigned int i;
        int countBits = 0;
        for (i = 0; i < 8 * sizeof(unsigned long); i++)
            if (rMask & (1 << i))
                countBits++;
        int rShiftRes = 8 - countBits;

        countBits = 0;
        for (i = 0; i < 8 * sizeof(unsigned long); i++)
            if (gMask & (1 << i))
                countBits++;
        int gShiftRes = 8 - countBits;

        countBits = 0;
        for (i = 0; i < 8 * sizeof(unsigned long); i++)
            if (bMask & (1 << i))
                countBits++;
        int bShiftRes = 8 - countBits;

        // pixel buffer in 8-bit
        unsigned char *buffer = new unsigned char[3 * w * h];

        int l;
        for (l = 0; l < h; l++)
        {
            unsigned char *base = (unsigned char *)ximg->data + l * (ximg->bytes_per_line);
            unsigned long dataL = 0;
            unsigned char *dataB = (unsigned char *)&dataL;

            for (int i = 0; i < w; i++)
            {
                switch (bytesPerPixel)
                {
                case 1:
                    dataB[0] = *base;
                    ++base;
                    break;
                case 2:
                    dataB[0] = *base;
                    ++base;
                    dataB[1] = *base;
                    ++base;
                    break;
                case 3:
                    dataB[0] = *base;
                    ++base;
                    dataB[1] = *base;
                    ++base;
                    dataB[2] = *base;
                    ++base;
                    break;
                case 4:
                    dataB[0] = *base;
                    ++base;
                    dataB[1] = *base;
                    ++base;
                    dataB[2] = *base;
                    ++base;
                    /* 1 Byte skipped */ ++base;
                    break;

                case 5:
                    dataB[0] = *base;
                    ++base;
                    break;
                case 6:
                    dataB[1] = *base;
                    ++base;
                    dataB[0] = *base;
                    ++base;
                    break;
                case 7:
                    dataB[2] = *base;
                    ++base;
                    dataB[1] = *base;
                    ++base;
                    dataB[0] = *base;
                    ++base;
                    break;
                case 8:
                    /* 1 Byte skipped */ ++base;
                    dataB[2] = *base;
                    ++base;
                    dataB[1] = *base;
                    ++base;
                    dataB[0] = *base;
                    ++base;
                    break;
                }

                buffer[3 * (i + l * w)] = (dataL & rMask) >> rShift;
                buffer[3 * (i + l * w) + 1] = (dataL & gMask) >> gShift;
                buffer[3 * (i + l * w) + 2] = (dataL & bMask) >> bShift;

                buffer[3 * (i + l * w)] <<= rShiftRes;
                buffer[3 * (i + l * w) + 1] <<= gShiftRes;
                buffer[3 * (i + l * w) + 2] <<= bShiftRes;
            }
        }

        // +++++ 3rd part - put the RGB buffer into a TIFF image file

        TIFF *image = TIFFOpen(filename, "w");
        if (image)
        {
            TIFFSetField(image, TIFFTAG_IMAGEWIDTH, w);
            TIFFSetField(image, TIFFTAG_IMAGELENGTH, h);
            TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 8);
            TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 3);
            TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, h);

            TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
            TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
            TIFFSetField(image, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
            TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

            TIFFSetField(image, TIFFTAG_XRESOLUTION, 150.0);
            TIFFSetField(image, TIFFTAG_YRESOLUTION, 150.0);
            TIFFSetField(image, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);

            // Write the information to the file
            TIFFWriteEncodedStrip(image, 0, buffer, 3 * w * h);

            // Close the file
            TIFFClose(image);
        }
        else
            cerr << "x11SnapTIFF: Could not create output file: " << filename << endl;
    }
#endif
