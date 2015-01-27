/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <stdio.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>

#include <X11/StringDefs.h>
#include <X11/keysym.h>
#include <Xm/Form.h>
#include <Xm/PushB.h>
#include <Xm/LabelG.h>
#include <Xm/RowColumn.h>
#include <Xm/PushBG.h>
#include <Xm/SelectioB.h>
#include <Xm/Text.h>
#include <Xm/MessageB.h>
#include <Xm/CascadeBG.h>

#include <Inventor/SoInput.h>
#include <Inventor/SoDB.h>
#include <Inventor/SoPath.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/Xt/devices/SoXtInputFocus.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtClipboard.h>
#include <Inventor/Xt/SoXtRenderArea.h>
#include <Inventor/errors/SoDebugError.h>

#include "InvSimpleMaterialEditor.h"
#include "InvMaterialPalette.h"

/*
 * Defines
 */
// Define this to have menus appear in the popup planes
// instead of the normal planes. You lose menu colors,
// but don't have to redraw the scene just to see a menu.
#define MENUS_IN_POPUP
#ifdef __hpux
#define getwd(p) \
    ;            \
    getcwd(p, 1024);
#endif
extern char *STRDUP(const char *s);
// time in msec between consecutive clicks to consider
// the second click being a double click.
#define DOUBLE_CLICK_TIME 300

// name of the directory where custom palettes will be saved
// (under home directory)
#define customPalDir ".materials"

#define OVERLAY_SLOT 1

// those are used to set a flag which will tell use what needs to
// be done in a futur time (like once a dialog disapears).
enum ThingsToDo
{
    SWITCH_PALETTE,
    BRING_NEW_DIALOG,
    CREATE_NEW_PALETTE,
    SAVE_AS_PALETTE
};

enum MenuEntryID
{
    FILE_MENU = 0, // start at 0 since we use an array
    FILE_NEW,
    FILE_SAVE,
    FILE_SAVE_AS,
    FILE_RESET,
    FILE_DELETE,

    EDIT_MENU,
    EDIT_CUT,
    EDIT_COPY,
    EDIT_PASTE,
    EDIT_DELETE,

    // list of needed widget to build the palette menu on the fly
    MENU_BAR,
    PALETTE_BUTTON,
    PALETTE_MENU,

    MAT_LABEL,
    MENU_LENGTH // this must be the last entry
};

struct MaterialNameStruct
{
    char *name;
    char *oldName;
};

struct PaletteStruct
{
    const char *name;
    SbBool user;
    SbBool system;
};

struct MenuButtonItemStruct
{
    const char *name;
    int id;
    const char *accelerator; // e.g. "Alt <Key> p" or "Ctrl <Key> u"
    const char *accelText; // text that appears in the menu item
};

struct MenuStruct
{
    const char *name;
    long id;
    struct MenuButtonItemStruct *subMenu;
    int subItemCount;
};

/*
 * static vars
 */

static MenuButtonItemStruct fileData[] = {
    { "New...", FILE_NEW, "Alt <Key> n", "Alt+n" },
    { "Save", FILE_SAVE, "Alt <Key> s", "Alt+s" },
    { "Save As...", FILE_SAVE_AS, "Alt Shift <Key> s", "Alt+S" },
    { "Reset", FILE_RESET, "Alt <Key> r", "Alt+r" },
    { "Delete", FILE_DELETE, "Alt <Key> d", "Alt+d" },
};

static MenuButtonItemStruct editData[] = {
    { "Cut", EDIT_CUT, "Alt <Key> x", "Alt+x" },
    { "Copy", EDIT_COPY, "Alt <Key> c", "Alt+c" },
    { "Paste", EDIT_PASTE, "Alt <Key> v", "Alt+v" },
    { "Delete", EDIT_DELETE, "<Key> BackSpace", "BckSp" },
};

static MenuStruct pulldownData[] = {
    //  {name, 	    id,	    	    subMenu,    subItemCount}
    { "File", FILE_MENU, fileData, XtNumber(fileData) },
    { "Edit", EDIT_MENU, editData, XtNumber(editData) },
};

static const char *editorTitle = "Material Palette";
static const char *defaultDir = "/usr/share/data/materials";

static const char *geometryBuffer = "\
#Inventor V2.0 ascii\n\
Separator { \
    OrthographicCamera { \
	position 3.5 3.5 5 \
	nearDistance 1.0 \
	farDistance 10.0 \
	height 6.2 \
    } \
    LightModel { model BASE_COLOR } \
    BaseColor { rgb [.4 .4 .4] } \
    Coordinate3 { point [ \
	0 1 0, 7 1 0, 0 2 0, 7 2 0, 0 3 0, 7 3 0, \
	0 4 0, 7 4 0, 0 5 0, 7 5 0, 0 6 0, 7 6 0, \
	1 0 0, 1 7 0, 2 0 0, 2 7 0, 3 0 0, 3 7 0, \
	4 0 0, 4 7 0, 5 0 0, 5 7 0, 6 0 0, 6 7 0] } \
    DrawStyle { lineWidth 2 } \
    LineSet { numVertices [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] } \
    LightModel { model PHONG } \
    DirectionalLight { direction .556 -.551 -.623 intensity .7 } \
    DirectionalLight { direction -.556 -.551 -.623 intensity .7 } \
    Complexity { value .4 } \
    Translation { translation 1 6 0 } \
    Array { \
	numElements1 6 \
	numElements2 6 \
	separation1 1 0 0 \
	separation2 0 -1 0 \
	Switch { \
	    whichChild -2 \
	    Material{} Material{} Material{} Material{} Material{} Material{} \
	    Material{} Material{} Material{} Material{} Material{} Material{} \
	    Material{} Material{} Material{} Material{} Material{} Material{} \
	    Material{} Material{} Material{} Material{} Material{} Material{} \
	    Material{} Material{} Material{} Material{} Material{} Material{} \
	    Material{} Material{} Material{} Material{} Material{} Material{} \
	} \
	Sphere { radius .43 } \
    } \
} ";

static const char *overlayGeometryBuffer = "\
#Inventor V2.0 ascii\n\
Separator { \
    LightModel { model BASE_COLOR } \
    ColorIndex { index 1 } \
    Coordinate3 { point [ -.5 -.5 0, .5 -.5 0, .5 .5 0, -.5 .5 0, -.5 -.5 0 ] } \
    Separator { \
	Translation {} \
	DrawStyle { lineWidth 3 } \
	LineSet { numVertices 5 } \
    } \
    Translation {} \
    LineSet { numVertices 5 } \
} ";

//
//  returns true if the passed file is a subdirectory in the current directory.
//  (call chdir() before calling this).
//
static SbBool
isDirectory(char *file)
{
    struct stat buf;

    if (stat(file, &buf) == 0)
        if ((buf.st_mode & S_IFMT) == S_IFDIR)
            return TRUE;

    return FALSE;
}

//
// return PaletteStruct pointer which contains the given name string
// in the given PbPlist of structs.
//
static PaletteStruct *
findPalette(char *str, SbPList *list)
{
    PaletteStruct *pal = NULL;

    for (int i = 0; i < list->getLength(); i++)
    {
        pal = (PaletteStruct *)(*list)[i];
        if (strcmp(str, pal->name) == 0)
            return pal;
        else
            pal = NULL;
    }
    return pal;
}

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
MyMaterialPalette::MyMaterialPalette(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    const char *dir)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(dir, TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
MyMaterialPalette::MyMaterialPalette(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    const char *dir,
    SbBool buildNow)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(dir, buildNow);
}

////////////////////////////////////////////////////////////////////////
//
// Called by the constructors
//
// private
//
void
MyMaterialPalette::constructorCommon(const char *dir, SbBool buildNow)
//
//////////////////////////////////////////////////////////////////////
{
    int i;

    setClassName("MyMaterialPalette");
    paletteDir = (dir != NULL) ? STRDUP(dir) : STRDUP(defaultDir);
    paletteChanged = FALSE;
    selectedItem = currentItem = -1;
    curPalette = -1;
    prevTime = 0;
    clipboard = NULL;

    // widget vars
    widgetList = new Widget[MENU_LENGTH];
    for (i = 0; i < MENU_LENGTH; i++)
        widgetList[i] = NULL;
    popupWidget = NULL;

    // alocate needed stuff
    matEditor = NULL;

    // allocate the material name list
    mtlNames = new MaterialNameStruct[36];
    for (i = 0; i < 36; i++)
        mtlNames[i].name = mtlNames[i].oldName = NULL;

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
}

////////////////////////////////////////////////////////////////////////
//
//    Destructor.
//
MyMaterialPalette::~MyMaterialPalette()
//
////////////////////////////////////////////////////////////////////////
{
    int i;

    // delete allocated stuff
    delete focus;
    delete ra;
    delete matEditor;
    delete paletteDir;
    delete clipboard;

    // delete palette names
    PaletteStruct *pal;
    for (i = 0; i < paletteList.getLength(); i++)
    {
        pal = (PaletteStruct *)paletteList[i];
        delete[] pal -> name;
        delete pal;
    }
    paletteList.truncate(0);

    // delete material names
    for (i = 0; i < 36; i++)
    {
        if (mtlNames[i].name != NULL)
            delete[] mtlNames[i].name;
        if (mtlNames[i].oldName != NULL)
            delete[] mtlNames[i].oldName;
    }
    delete[] mtlNames;

    // delete widget stuff
    delete[] widgetList;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	deselect the currently selected item in the palette.
//
// Use: public
void
MyMaterialPalette::deselectCurrentItem()
//
////////////////////////////////////////////////////////////////////////
{
    if (selectedItem == -1)
        return;

    // deselect the current item and update things that depend on it
    selectedItem = -1;
    updateOverlayFeedback();
    updateMaterialName();
    updateEditMenu();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets the window title based on the current palette and save status
//
// Use: private
void
MyMaterialPalette::updateWindowTitle()
//
////////////////////////////////////////////////////////////////////////
{
    char str[150];
    sprintf(str, "%s: %s", editorTitle, ((PaletteStruct *)paletteList[curPalette])->name);
    if (paletteChanged)
        strcat(str, "*");
    setTitle(str);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Buils the palette layout (render area + motif buttons).
//
// Use: protected
Widget
MyMaterialPalette::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    // force the window to resize in fixed increments from the
    // window minimum size. This will guarantee that the RA has
    // a 1/1 aspect ratio.
    //
    // ??? this is necessary for the fast window/region picking
    // ??? we are doing during mouse motion.
    //
    if (XtIsShell(parent))
        XtVaSetValues(parent,
                      XmNbaseWidth, 210,
                      XmNbaseHeight, 263,
                      XmNminAspectX, 1,
                      XmNmaxAspectX, 1,
                      XmNminAspectY, 1,
                      XmNmaxAspectY, 1,
                      NULL);

    int n;
    Arg args[12];

    // create a top level form to hold everything together
    Widget form = XmCreateForm(parent, (char *)"matPalForm", NULL, 0);

    //
    // create all the parts
    //

    // Render area
    ra = new SoXtRenderArea(form);
    ra->setSize(SbVec2s(250, 250));
    // spheres are last
    ra->setTransparencyType(SoGLRenderAction::BLEND);
    ra->setEventCallback(MyMaterialPalette::raEventCB, this);
    //    ra->setBackgroundColor(SbColor(.6, .6, .6));
    SbColor col(.9, .2, .2);
    ra->setOverlayColorMap(1, 1, &col);

    Widget raWidget = ra->getWidget();
    createSceneGraph();

#if 0
   // make renderArea single buffered on Starter graphics (less than 24 bits)
   // which is ok since the scene is mostly static.
   long bitnum = getgdesc(GD_BITS_NORM_DBL_RED) + getgdesc(GD_BITS_NORM_DBL_GREEN)
      + getgdesc(GD_BITS_NORM_DBL_BLUE);
   if (bitnum < 12)
      ra->setDoubleBuffer(FALSE);
#endif

    // add leave window events
    focus = new SoXtInputFocus((EventMask)LeaveWindowMask);
    ra->registerDevice(focus);

    // build the menu
    getPaletteNamesAndLoad();
    Widget menu = buildMenu(form);

    widgetList[MAT_LABEL] = XmCreateLabelGadget(form, (char *)"matLabel", NULL, 0);

    //
    // make sure things are updated
    //
    updateMaterialName();
    updateWindowTitle();
    updateFileMenu();
    updateEditMenu();

    //
    // layout !
    //
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(menu, args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 7);
    n++;
    XtSetArg(args[n], XmNalignment, XmALIGNMENT_CENTER);
    n++;
    XtSetValues(widgetList[MAT_LABEL], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, menu);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, widgetList[MAT_LABEL]);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 5);
    n++;
    XtSetValues(raWidget, args, n);

    // manage children
    XtManageChild(menu);
    XtManageChild(widgetList[MAT_LABEL]);
    XtManageChild(raWidget);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// After realization, we can set up the color map for the popup menu windows.
//
// Use: protected
//
void
MyMaterialPalette::afterRealizeHook()
//
////////////////////////////////////////////////////////////////////////
{
    SoXtComponent::afterRealizeHook();

#ifdef MENUS_IN_POPUP
    if (popupWidget)
        SoXt::addColormapToShell(popupWidget, SoXt::getShellWidget(getWidget()));
#endif
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	creates the menu bar
//
// Use: private
Widget
MyMaterialPalette::buildMenu(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    long id;
    Arg args[8];

    // create top bar menu
    Widget menu = XmCreateMenuBar(parent, (char *)"menuBar", NULL, 0);
    widgetList[MENU_BAR] = menu;

    Arg popupargs[4];
    int popupn = 0;
#ifdef MENUS_IN_POPUP
    SoXt::getPopupArgs(XtDisplay(menu), 0, popupargs, &popupn);
#endif

    int itemCount = XtNumber(pulldownData);
    Widget *buttons = new Widget[itemCount + 1]; // for palette

    int i;
    for (i = 0; i < itemCount; i++)
    {
        // Make Topbar menu button
        Widget subMenu = XmCreatePulldownMenu(menu, (char *)"", popupargs, popupn);
        // we only need one widget for loading the proper popup colormap
        if (!popupWidget)
            popupWidget = subMenu;

        widgetList[pulldownData[i].id] = subMenu;

        XtSetArg(args[0], XmNsubMenuId, subMenu);
        buttons[i] = XmCreateCascadeButtonGadget(menu, (char *)pulldownData[i].name, args, 1);

        // make submenu buttons
        int subItemCount = pulldownData[i].subItemCount;
        Widget *subButtons = new Widget[subItemCount];

        for (int j = 0; j < subItemCount; j++)
        {

            n = 0;
            XtSetArg(args[n], XmNuserData, this);
            n++;

            // check for keyboard accelerator
            XmString xmstr = NULL;
            char *accel = (char *)pulldownData[i].subMenu[j].accelerator;
            char *accelText = (char *)pulldownData[i].subMenu[j].accelText;
            if (accel != NULL)
            {
                XtSetArg(args[n], XmNaccelerator, accel);
                n++;

                if (accelText != NULL)
                {
                    xmstr = XmStringCreateSimple(accelText);
                    XtSetArg(args[n], XmNacceleratorText, xmstr);
                    n++;
                }
            }

            id = pulldownData[i].subMenu[j].id;
            widgetList[id] = subButtons[j] = XmCreatePushButtonGadget(subMenu,
                                                                      (char *)pulldownData[i].subMenu[j].name, args, n);

            if (xmstr != NULL)
                XmStringFree(xmstr);

            XtAddCallback(subButtons[j], XmNactivateCallback,
                          (XtCallbackProc)MyMaterialPalette::menuCB,
                          (XtPointer)id);
        }
        XtManageChildren(subButtons, subItemCount);
        delete[] subButtons;
    }

    // create palette menu (the palette popup menu gets created and deleted
    // on the fly so do it separately).
    widgetList[PALETTE_BUTTON] = buttons[i] = XmCreateCascadeButtonGadget(menu,
                                                                          (char *)"Palettes", NULL, 0);
    widgetList[PALETTE_MENU] = NULL;
    buildPaletteSubMenu();

    XtManageChildren(buttons, itemCount + 1); // because of palette
    delete[] buttons;

    return menu;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	adds an entry to the palette menu.
//
// Use: private
Widget
MyMaterialPalette::buildPaletteMenuEntry(long id)
//
////////////////////////////////////////////////////////////////////////
{
    PaletteStruct *pal = (PaletteStruct *)paletteList[id];
    char accel[20];
    char accelText[20];
    XmString xmstr = NULL;
    Arg args[4];
    int n = 0;

    XtSetArg(args[n], XmNuserData, this);
    n++;
    if (id < 10)
    {
        sprintf(accel, "Alt <Key> %ld", id);
        sprintf(accelText, "Alt+%ld", id);
        xmstr = XmStringCreateSimple(accelText);

        XtSetArg(args[n], XmNaccelerator, accel);
        n++;
        XtSetArg(args[n], XmNacceleratorText, xmstr);
        n++;
    }

    Widget w = XmCreatePushButtonGadget(widgetList[PALETTE_MENU], (char *)pal->name, args, n);
    XtAddCallback(w, XmNactivateCallback,
                  (XtCallbackProc)MyMaterialPalette::paletteMenuCB,
                  (XtPointer)id);

    if (xmstr != NULL)
        XmStringFree(xmstr);

    return w;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Buils the palette popup menu.
//
// Use: private
void
MyMaterialPalette::buildPaletteSubMenu()
//
////////////////////////////////////////////////////////////////////////
{
    // rebuild the palette popup
    // ??? we cannot delete the old popup menu or things will brake.
    // ??? Is it automatically deleted for us ?
    Arg args[8];
    int argnum = 0;
#ifdef MENUS_IN_POPUP
    SoXt::getPopupArgs(XtDisplay(widgetList[MENU_BAR]), 0, args, &argnum);
#endif
    widgetList[PALETTE_MENU] = XmCreatePulldownMenu(widgetList[MENU_BAR], (char *)"", args, argnum);

    XtSetArg(args[0], XmNsubMenuId, widgetList[PALETTE_MENU]);
    XtSetValues(widgetList[PALETTE_BUTTON], args, 1);

    // build all of the entries
    Widget *entries = new Widget[paletteList.getLength()];
    for (int i = 0; i < paletteList.getLength(); i++)
        entries[i] = buildPaletteMenuEntry(i);

    XtManageChildren(entries, paletteList.getLength());
    delete[] entries;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Called whenever a menu entry is selected
//
// Use: private
void
MyMaterialPalette::menuCB(Widget w, int id, XmAnyCallbackStruct *cb)
//
////////////////////////////////////////////////////////////////////////
{
    Time eventTime = cb->event->xbutton.time;

    // get the class pointer
    MyMaterialPalette *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    switch (id)
    {
    case FILE_NEW:
        if (p->paletteChanged)
        {
            p->whatToDoNext = BRING_NEW_DIALOG;
            p->createSaveDialog();
        }
        else
        {
            p->whatToDoNext = CREATE_NEW_PALETTE;
            p->createPromptDialog("New Palette Dialog", "New Palette Name:");
        }
        break;

    case FILE_SAVE:
        p->savePalette();
        break;

    case FILE_SAVE_AS:
        p->whatToDoNext = SAVE_AS_PALETTE;
        p->createPromptDialog("Save As Dialog", "Save As:");
        break;

    case FILE_RESET:
        p->createDeleteDialog("Reset Palette Dialog",
                              "Reset to default palette ?", "(all changes will be lost)");
        break;

    case FILE_DELETE:
        p->createDeleteDialog("Delete Palette Dialog",
                              "Delete current palette ?", "(all materials will be lost)");
        break;

    case EDIT_CUT:
    case EDIT_COPY:
        // copy material
        if (p->clipboard == NULL)
            p->clipboard = new SoXtClipboard(p->getWidget());
        p->clipboard->copy(p->itemSwitch->getChild(p->selectedItem), eventTime);

        // now delete material
        if (id == EDIT_CUT)
            p->deleteCurrentMaterial();
        break;

    case EDIT_PASTE:
        if (p->clipboard == NULL)
            p->clipboard = new SoXtClipboard(p->getWidget());
        p->clipboard->paste(eventTime, MyMaterialPalette::pasteDone, p);
        break;

    case EDIT_DELETE:
        p->deleteCurrentMaterial();
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Creates the scene graph which consists of repeated spheres with
//  materials.
//
// Use: private
void
MyMaterialPalette::createSceneGraph()
//
////////////////////////////////////////////////////////////////////////
{
    SoInput in;
    SoNode *node;
    SbBool ok;
    SoSearchAction sa;
    SoFullPath *fullPath;

    //
    // create the regular scene graph
    //

    // read the geometry buffer in
    in.setBuffer((void *)geometryBuffer, (size_t)strlen(geometryBuffer));
    ok = SoDB::read(&in, node);
    if (!ok || node == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MyMaterialPalette::createSceneGraph",
                           "couldn't read geometry");
        exit(1);
#endif
    }
    ra->setSceneGraph(node);

    // search for the switch node which contains all the materials
    sa.setType(SoSwitch::getClassTypeId(), FALSE);
    sa.apply(node);
    if ((fullPath = (SoFullPath *)sa.getPath()) == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MyMaterialPalette::createSceneGraph",
                           "couldn't find switch node");
        exit(1);
#endif
    }
    itemSwitch = (SoSwitch *)fullPath->getTail();

    //
    // now create the overlay scene graph
    //

    // read the overlay geometry buffer
    in.setBuffer((void *)overlayGeometryBuffer, (size_t)strlen(overlayGeometryBuffer));
    ok = SoDB::read(&in, node);
    if (!ok || node == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MyMaterialPalette::createSceneGraph",
                           "couldn't read overlay geometry");
        exit(1);
#endif
    }
    ra->setOverlaySceneGraph(node);

    // search for the camera in the regular scene to also use it for the overlay
    sa.setType(SoOrthographicCamera::getClassTypeId(), FALSE);
    sa.apply(ra->getSceneGraph());
    if ((fullPath = (SoFullPath *)sa.getPath()) == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MyMaterialPalette::createSceneGraph",
                           "couldn't find camera");
        exit(1);
#endif
    }
    ((SoGroup *)ra->getOverlaySceneGraph())->insertChild(fullPath->getTail(), 0);

    // search for the 2 translation nodes for the lineSet feedback
    sa.setInterest(SoSearchAction::ALL);
    sa.setType(SoTranslation::getClassTypeId(), FALSE);
    sa.apply(ra->getOverlaySceneGraph());
    if (sa.getPaths().getLength() != 2)
    {
#ifdef DEBUG
        SoDebugError::post("MyMaterialPalette::createSceneGraph",
                           "couldn't find 2 translations");
        exit(1);
#endif
    }
    overlayTrans1 = (SoTranslation *)((SoFullPath *)sa.getPaths()[0])->getTail();
    overlayTrans2 = (SoTranslation *)((SoFullPath *)sa.getPaths()[1])->getTail();
    updateOverlayFeedback();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Get the list of directories (palettes) from the env var, or the
//  paletteDir variable (constructor). This also searches for palettes
//  saved under the home directory, which would overide the system
//  installed palettes.
//
// Use: private
void
MyMaterialPalette::getPaletteNamesAndLoad()
//
////////////////////////////////////////////////////////////////////////
{
    struct dirent *direntry;
    DIR *dirp;
    char *f;
    char currentDir[MAXPATHLEN + 1];

    // ??? this is only called once at startup, so it doesn't
    // ??? need to free the old paletteList.

    //
    // look for palettes under the default installed place
    //

    // see if SO_MATERIAL_DIR is set, if so use it...
    char *envDir = getenv("SO_MATERIAL_DIR");
    if (envDir != NULL)
    {
        delete paletteDir;
        paletteDir = STRDUP(envDir);
    }
    dirp = opendir(paletteDir);
    if (dirp)
    {
        if (getcwd(currentDir, MAXPATHLEN) == NULL)
        {
            fprintf(stderr, "MyMaterialPalette::getPaletteNamesAndLoad: getcwd1 failed\n");
        }
        if (chdir(paletteDir) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::getPaletteNamesAndLoad: chdir1 failed\n");
        }

        while ((direntry = readdir(dirp)))
        {
            f = direntry->d_name;
            // hide '.' files
            if (f[0] != '.' && isDirectory(f))
            {
                PaletteStruct *pal = new PaletteStruct;
                pal->name = STRDUP(f);
                pal->system = TRUE;
                pal->user = FALSE;
                paletteList.append(pal);
            }
        }
        closedir(dirp);
        // back to our working directory
        if (chdir(currentDir) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::getPaletteNamesAndLoad: chdir2 failed\n");
        }
    }

    //
    // Now look for palettes under the user's home directory, which
    // would overide the installed palettes.
    //

    char customDir[MAXPATHLEN + 1];
    sprintf(customDir, "%s/%s", getenv("HOME"), customPalDir);
    dirp = opendir(customDir);
    if (dirp)
    {
        if (getcwd(currentDir, MAXPATHLEN) == NULL)
        {
            fprintf(stderr, "MyMaterialPalette::getPaletteNamesAndLoad: getcwd2 failed\n");
        }
        if (chdir(customDir) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::getPaletteNamesAndLoad: chdir3 failed\n");
        }

        while ((direntry = readdir(dirp)))
        {
            f = direntry->d_name;
            // hide '.' files
            if (f[0] != '.' && isDirectory(f))
            {

                // check if palette name is already in the list
                // (user overide case), else create a new entry
                //
                PaletteStruct *pal = findPalette(f, &paletteList);
                if (pal != NULL)
                    pal->user = TRUE;
                else
                {
                    pal = new PaletteStruct;
                    pal->name = STRDUP(f);
                    pal->user = TRUE;
                    pal->system = FALSE;
                    paletteList.append(pal);
                }
            }
        }
        closedir(dirp);
        // back to our working directory
        if (chdir(currentDir) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::getPaletteNamesAndLoad: chdir3 failed\n");
        }
    }

    //
    // make sure we have at least one palette, else create and empty default
    // palette for things to work.
    //
    curPalette = 0;
    if (paletteList.getLength() == 0)
    {
#ifdef DEBUG
        SoDebugError::post("MyMaterialPalette::getPaletteNamesAndLoad",
                           "cannot find palettes in directory %s or home directory.  Try setting the environment variable SO_MATERIAL_DIR to a directory which has material files in it.", paletteDir);
#endif

        // create an empty default palette, withough loading anything
        // since we started with a blank palette.
        PaletteStruct *pal = new PaletteStruct;
        pal->name = STRDUP("default");
        pal->user = TRUE;
        pal->system = FALSE;
        paletteList.append(pal);

        // set the material empty name, since we are not loading any
        // palette and this is only done once.
        for (int i = 0; i < 36; i++)
        {
            char str[50];
            sprintf(str, "no_name_%d", i);
            mtlNames[i].name = STRDUP(str);
        }
    }
    else
        // load the first palette
        loadPaletteItems();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Given the current palette, loads all of the materials in the
//  current palette directory.
//
// Use: private
void
MyMaterialPalette::loadPaletteItems()
//
////////////////////////////////////////////////////////////////////////
{
    char palDir[MAXPATHLEN + 1];
    char currentDir[MAXPATHLEN + 1];
    DIR *dirp;
    int i, num = 0;

    // delete the existing name strings of materials
    for (i = 0; i < 36; i++)
    {
        if (mtlNames[i].name != NULL)
        {
            delete[] mtlNames[i].name;
            mtlNames[i].name = NULL;
        }
        if (mtlNames[i].oldName != NULL)
        {
            delete[] mtlNames[i].oldName;
            mtlNames[i].oldName = NULL;
        }
    }

    // go to the palette directory (user saved palettes overide system
    // installed palettes).
    PaletteStruct *pal = (PaletteStruct *)paletteList[curPalette];
    if (pal->user)
        sprintf(palDir, "%s/%s/%s", getenv("HOME"), customPalDir, pal->name);
    else
        sprintf(palDir, "%s/%s", paletteDir, pal->name);

    if ((dirp = opendir(palDir)) != NULL)
    {
        // cd to palette directory
        if (getcwd(currentDir, MAXPATHLEN) == NULL)
        {
            fprintf(stderr, "MyMaterialPalette::loadPaletteItems: getcwd1 failed\n");
        }
        if (chdir(palDir) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::loadPaletteItems: chdir1 failed\n");
        }

        // loop throught the files, and open any material files
        char *f;
        struct dirent *direntry;
        SoNode *mat;

        while ((direntry = readdir(dirp)) && num < 35)
        {
            f = direntry->d_name;
            mat = getMaterialFromFile(f);
            if (mat)
            {
                itemSwitch->replaceChild(num, mat);
                mtlNames[num].name = STRDUP(f);
                num++;
            }
        }

        closedir(dirp);
        // back to our working directory
        if (chdir(currentDir) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::loadPaletteItems: chdir2 failed\n");
        }
    }

    // now clear the reminder materials in the palette (if any)
    for (i = num; i < 36; i++)
    {
        char str[50];
        sprintf(str, "no_name_%d", i);
        itemSwitch->replaceChild(i, new SoMaterial);
        mtlNames[i].name = STRDUP(str);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	returns a material from the given file name (or NULL if the file
//  isn't valid or contain a material). The file will be opened from the cwd
//  (call chdir() before calling this).
//
// Use: private
SoNode *
MyMaterialPalette::getMaterialFromFile(char *file)
//
////////////////////////////////////////////////////////////////////////
{
    // ignore '.' files
    if (file[0] == '.')
        return NULL;

    // ignore directories
    if (isDirectory(file))
        return NULL;

    // open the file and read the material in (assume 1 material per file)
    SoNode *node;
    SoInput in;
    if (!in.openFile(file))
        return NULL;
    if (!SoDB::read(&in, node) || node == NULL)
        return NULL;

    // now check to make sure node is a material
    if (node->isOfType(SoMaterial::getClassTypeId()))
        return node;
    else
    {
        // nuke that node...
        node->ref();
        node->unref();
    }
    return NULL;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Called whenever an event occurs in the render area.
//
// Use: private
SbBool
MyMaterialPalette::handleEvent(XAnyEvent *xe)
//
////////////////////////////////////////////////////////////////////////
{
    switch (xe->type)
    {
    case ButtonPress:
    {
        XButtonEvent *be = (XButtonEvent *)xe;
        if (be->button != Button1)
            break;

        // it is possible to get a mouse down event withought
        // previously receiving a mouse motion event (when the
        // editor window is closed on top of the palette and the
        // mouse doesn't move at all). In that case, find which
        // item we are clicking on...
        if (currentItem == -1)
            findCurrentItem(be->x, be->y);

        //
        // check for double click. If the time difference between
        // successive mouse down is less than 3/10th sec, then
        // consider it a double click.
        // ??? do we need to worry about time warping ?
        //
        if ((be->time - prevTime) < DOUBLE_CLICK_TIME)
        {
            // allocate the editor on the fly....
            if (matEditor == NULL)
            {
                matEditor = new MySimpleMaterialEditor(
                    SoXt::getShellWidget(getWidget()),
                    NULL, FALSE, TRUE);
                matEditor->addCallback(MyMaterialPalette::matEditorCB, this);
                matEditor->setTitle("Material Palette Editor");
                matEditor->setIconTitle("Mat Pal Editor");
            }

            // update the editor if it wasn't already on the screen
            // (would have otherwise been updated by the first click)
            if (!matEditor->isVisible())
            {
                matEditor->setMaterial((SoMaterial *)itemSwitch->getChild(selectedItem));
                matEditor->setMaterialName(mtlNames[selectedItem].name);
            }
            matEditor->show();
        }
        else // single click (or first click of the double click)
        {
            // select the current material
            if (selectedItem != currentItem)
            {

                int oldVal = selectedItem;
                selectedItem = currentItem;
                if (oldVal == -1)
                    updateEditMenu();
                updateOverlayFeedback();

                callbackList.invokeCallbacks(itemSwitch->getChild(selectedItem));
            }

            // update the material editor if it is on the screen
            if (matEditor != NULL && matEditor->isVisible())
            {
                matEditor->setMaterial((SoMaterial *)itemSwitch->getChild(selectedItem));
                matEditor->setMaterialName(mtlNames[selectedItem].name);
            }
        }

        prevTime = be->time;
    }
    break;

    case MotionNotify:
    {
        XMotionEvent *me = (XMotionEvent *)xe;
        findCurrentItem(me->x, me->y);
    }
    break;

    case LeaveNotify:
        currentItem = -1;
        updateMaterialName();
        updateOverlayFeedback();
        break;
#if 0
      case KeyPress:
         if (XLookupKeysym((XKeyEvent *)xe, 0) == XK_Escape)
            exit(0);
#endif
    }

    // always handle the events...
    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	figure out which tile the mouse is over, and make that tile
// the current item (not the seleted item) - this is called when the
// mouse moves over our window.
//
// Use: private
void
MyMaterialPalette::findCurrentItem(int x, int y)
//
////////////////////////////////////////////////////////////////////////
{
    // note: this picking by region will only work
    // if the RA widget is square (which it is since
    // we are forcing the window to resize in even increments).

    SbVec2s raSize = ra->getSize();
    int xpos = int(floorf(6 * x / float(raSize[0])));
    int ypos = int(floorf(6 * y / float(raSize[1])));
    int whichItem = xpos + 6 * ypos;
    if (whichItem != currentItem)
    {
        currentItem = whichItem;
        updateMaterialName();
        updateOverlayFeedback();
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	changes the overlay feedback based on current state (what mat
//  is selected and what mat the mouse is over).
//
// Use: private
void
MyMaterialPalette::updateOverlayFeedback()
//
////////////////////////////////////////////////////////////////////////
{
    int row, col;

    if (selectedItem != -1)
    {
        row = (int)(selectedItem / 6);
        col = selectedItem - 6 * row;
        overlayTrans1->translation.setValue(col + 1, 6 - row, 0);
    }
    else
        overlayTrans1->translation.setValue(-10, -10, 0);

    if (currentItem != -1 && currentItem != selectedItem)
    {
        row = (int)(currentItem / 6);
        col = currentItem - 6 * row;
        overlayTrans2->translation.setValue(col + 1, 6 - row, 0);
    }
    else
        overlayTrans2->translation.setValue(-10, -10, 0);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	updates the material name label widget, based on the current
//  material.
//
// Use: private
void
MyMaterialPalette::updateMaterialName()
//
////////////////////////////////////////////////////////////////////////
{
    char *str = (currentItem < 0) ? ((selectedItem < 0) ? STRDUP(" ") : mtlNames[selectedItem].name) : mtlNames[currentItem].name;
    XmString xmstr = XmStringCreateSimple(str);
    XtVaSetValues(widgetList[MAT_LABEL], XmNlabelString, xmstr, NULL);
    XmStringFree(xmstr);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Called to enable/disable the cut/copy/paste/delete entries
//
// Use: private
void
MyMaterialPalette::updateEditMenu()
//
////////////////////////////////////////////////////////////////////////
{
    if (widgetList[EDIT_MENU] == NULL)
        return;

    Arg args[1];
    XtSetArg(args[0], XmNsensitive, selectedItem >= 0);
    XtSetValues(widgetList[EDIT_CUT], args, 1);
    XtSetValues(widgetList[EDIT_COPY], args, 1);
    XtSetValues(widgetList[EDIT_PASTE], args, 1);
    XtSetValues(widgetList[EDIT_DELETE], args, 1);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Called to enable/disable the "File" menu entries
//
// Use: private
void
MyMaterialPalette::updateFileMenu()
//
////////////////////////////////////////////////////////////////////////
{
    if (widgetList[FILE_MENU] == NULL)
        return;

    PaletteStruct *pal = (PaletteStruct *)paletteList[curPalette];

    XtVaSetValues(widgetList[FILE_RESET], XmNsensitive,
                  pal->user && pal->system, NULL);
    XtVaSetValues(widgetList[FILE_DELETE], XmNsensitive,
                  pal->user && !pal->system, NULL);
}

////////////////////////////////////////////////////////////////////////
//
// show the component
//
// usage: virtual public
//
void
MyMaterialPalette::show()
//
////////////////////////////////////////////////////////////////////////
{
    SoXtComponent::show();

    // now also show the material editor (if it was shown)
    if (matEditor != NULL && matEditor->getWidget() != NULL)
        matEditor->show();
}

////////////////////////////////////////////////////////////////////////
//
// hide the component
//
// usage: virtual public
//
void
MyMaterialPalette::hide()
//
////////////////////////////////////////////////////////////////////////
{
    SoXtComponent::hide();

    // now also hide the material editor
    if (matEditor != NULL)
        matEditor->hide();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates a new palette with the given name (called when the
//  user wants a new palette using the 'new' button).
//
// Use: private
void
MyMaterialPalette::createNewPalette(char *palName)
//
////////////////////////////////////////////////////////////////////////
{
    //
    // create that empty palette directory (under the user's home
    // directory).
    //
    char dirName[MAXPATHLEN];
    struct stat buf;
    sprintf(dirName, "%s/%s/", getenv("HOME"), customPalDir);
    if (stat(dirName, &buf) != 0)
        mkdir(dirName, 0x1ff);
    strcat(dirName, palName);
    if (mkdir(dirName, 0x1ff) != 0)
    {
#ifdef DEBUG
        SoDebugError::post("MyMaterialPalette::createNewPalette",
                           "couldn't create directory %s", dirName);
#endif
        return;
    }

    //
    // add the palette to the popup menu
    //
    PaletteStruct *pal = new PaletteStruct;
    pal->name = STRDUP(palName);
    pal->user = TRUE;
    pal->system = FALSE;
    paletteList.append(pal);
    int id = paletteList.getLength() - 1;
    Widget entry = buildPaletteMenuEntry(id);
    XtManageChild(entry);

    //
    // now make this the current palette (by making it like the
    // user picked it from the popup menu).
    //
    paletteMenuCB(entry, id, NULL);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Creates a dialog to ask the user if he/she wishes to save the
//  current palette (called when a new palette is about to be installed,
//  but the old one hasn't been saved and has changed).
//
// Use: private
void
MyMaterialPalette::createSaveDialog()
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[5];
    XmString xmstr = XmStringCreateSimple((char *)"Warning: current palette was modified.");
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringCreateSimple((char *)"Save changes ?"));

    int n = 0;
    XtSetArg(args[n], XmNautoUnmanage, FALSE);
    n++;
    XtSetArg(args[n], XtNtitle, "Save Palette Dialog");
    n++;
    XtSetArg(args[n], XmNmessageString, xmstr);
    n++;
    Widget dialog = XmCreateWarningDialog(getWidget(), (char *)"SaveDialog", args, n);
    XmStringFree(xmstr);

    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_HELP_BUTTON));
    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_SEPARATOR));

    // register callback to destroy (and not just unmap) the dialog
    XtAddCallback(dialog, XmNokCallback,
                  (XtCallbackProc)MyMaterialPalette::saveDialogCB, (XtPointer) this);
    XtAddCallback(dialog, XmNcancelCallback,
                  (XtCallbackProc)MyMaterialPalette::saveDialogCB, (XtPointer) this);

    XtManageChild(dialog);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	creates a dialog which lets the user type a name.
//
// Use: private
void
MyMaterialPalette::createPromptDialog(const char *title, const char *str)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[5];
    XmString xmstr = XmStringCreateSimple((char *)str);

    int n = 0;
    XtSetArg(args[n], XmNautoUnmanage, FALSE);
    n++;
    XtSetArg(args[n], XtNtitle, title);
    n++;
    XtSetArg(args[n], XmNselectionLabelString, xmstr);
    n++;
    Widget dialog = XmCreatePromptDialog(getWidget(), (char *)"promptDialog", args, n);
    XmStringFree(xmstr);

    XtUnmanageChild(XmSelectionBoxGetChild(dialog, XmDIALOG_HELP_BUTTON));
    XtUnmanageChild(XmSelectionBoxGetChild(dialog, XmDIALOG_SEPARATOR));

    // register callback to destroy (and not just unmap) the dialog
    // and retreive the text (ok push button case).
    XtAddCallback(dialog, XmNokCallback,
                  (XtCallbackProc)MyMaterialPalette::promptDialogCB, (XtPointer) this);
    XtAddCallback(dialog, XmNcancelCallback,
                  (XtCallbackProc)MyMaterialPalette::promptDialogCB, (XtPointer) this);

    XtManageChild(dialog);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	create a dialog which lets the user change it's mind about deleting
//  the current material palette.
//
// Use: private
void
MyMaterialPalette::createDeleteDialog(const char *title, const char *str1, const char *str2)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[5];
    XmString xmstr = XmStringCreateSimple((char *)str1);
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringCreateSimple((char *)str2));

    int n = 0;
    XtSetArg(args[n], XmNautoUnmanage, FALSE);
    n++;
    XtSetArg(args[n], XtNtitle, title);
    n++;
    XtSetArg(args[n], XmNmessageString, xmstr);
    n++;
    Widget dialog = XmCreateWarningDialog(getWidget(), (char *)"DeleteDialog", args, n);
    XmStringFree(xmstr);

    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_HELP_BUTTON));
    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_SEPARATOR));

    // register callback to destroy (and not just unmap) the dialog
    XtAddCallback(dialog, XmNokCallback,
                  (XtCallbackProc)MyMaterialPalette::deleteDialogCB, (XtPointer) this);
    XtAddCallback(dialog, XmNcancelCallback,
                  (XtCallbackProc)MyMaterialPalette::deleteDialogCB, (XtPointer) this);

    XtManageChild(dialog);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Saves the current palette to the given palette name, and switch
//  to it (i.e. create a new palette of the given name withought changing
//  the materials, and write it out).
//
// Use: private
void
MyMaterialPalette::savePaletteAs(char *palName)
//
////////////////////////////////////////////////////////////////////////
{
    //
    // add the palette to the popup menu
    //
    PaletteStruct *pal = new PaletteStruct;
    pal->name = STRDUP(palName);
    pal->user = TRUE;
    pal->system = FALSE;
    paletteList.append(pal);
    int id = paletteList.getLength() - 1;
    XtManageChild(buildPaletteMenuEntry(id));

    //
    // make this the current palette and save it out
    //
    curPalette = id;
    savePalette();
    updateWindowTitle();
    updateFileMenu();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This saves the current palette to the user's home directory.
//
// Use: private
void
MyMaterialPalette::savePalette()
//
////////////////////////////////////////////////////////////////////////
{
    int i;
    char currentDir[MAXPATHLEN + 1];
    if (getcwd(currentDir, MAXPATHLEN) == NULL)
    {
        fprintf(stderr, "MyMaterialPalette::savePalette: getcwd failed\n");
    }

    // go to the materials directory
    if (chdir(getenv("HOME")) == -1)
    {
        fprintf(stderr, "MyMaterialPalette::savePalette: chdir1 failed\n");
    }
    if (chdir(customPalDir) != 0)
    {
        // create that directory and cd to it
        mkdir(customPalDir, 0x1ff);
        if (chdir(customPalDir) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::savePalette: chdir2 failed\n");
        }
    }

    // now go to the palette directory (or create it)
    char *palName = (char *)((PaletteStruct *)paletteList[curPalette])->name;
    if (chdir(palName) != 0)
    {
        mkdir(palName, 0x1ff);
        if (chdir(palName) == -1)
        {
            fprintf(stderr, "MyMaterialPalette::savePalette: chdir3 failed\n");
        }
    }

    // delete any of the old material files if any
    // (material that changed name since last time)
    for (i = 0; i < 36; i++)
    {
        if (mtlNames[i].oldName != NULL)
        {
            unlink(mtlNames[i].oldName);
            delete[] mtlNames[i].oldName;
            mtlNames[i].oldName = NULL;
        }
    }

    // now save all of the materials
    // (even the empty one, this way we remember the order the palette has)
    SoWriteAction writeAct;
    SoOutput *out = writeAct.getOutput();
    for (i = 0; i < 36; i++)
    {
        out->openFile(mtlNames[i].name);
        writeAct.apply(itemSwitch->getChild(i));
        out->closeFile();
    }

    // back to our working directory
    if (chdir(currentDir) == -1)
    {
        fprintf(stderr, "MyMaterialPalette::savePalette: chdir4 failed\n");
    }

    // clear the changed flag
    if (paletteChanged)
    {
        paletteChanged = FALSE;
        updateWindowTitle();
    }
    ((PaletteStruct *)paletteList[curPalette])->user = TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	switches to the new palette (defined by "nextPalette" var). This
//  is called (maybe indirectly) when a new palette is choosen from the
//  palette popup menu.
//
// Use: private
void
MyMaterialPalette::switchPalette()
//
////////////////////////////////////////////////////////////////////////
{
    // assign new palette id
    curPalette = nextPalette;

    // get the new materials
    loadPaletteItems();
    paletteChanged = FALSE;

    // update the material name (nothing selected)
    deselectCurrentItem();
    updateWindowTitle();
    updateFileMenu();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	deletes the current material (reset to default value and name)
//
// Use: private
void
MyMaterialPalette::deleteCurrentMaterial()
//
////////////////////////////////////////////////////////////////////////
{
    // replace material with default empty one
    itemSwitch->replaceChild(selectedItem, new SoMaterial);

    // save original name for file removal and assign new default name
    if (mtlNames[selectedItem].oldName == NULL)
        mtlNames[selectedItem].oldName = mtlNames[selectedItem].name;
    else
        delete[] mtlNames[selectedItem].name;
    char str[50];
    sprintf(str, "no_name_%d", selectedItem);
    mtlNames[selectedItem].name = STRDUP(str);

    // update label and editor
    updateMaterialName();
    if (matEditor != NULL && matEditor->isVisible())
    {
        matEditor->setMaterial((SoMaterial *)itemSwitch->getChild(selectedItem));
        matEditor->setMaterialName(mtlNames[selectedItem].name);
    }

    if (!paletteChanged)
    {
        paletteChanged = TRUE;
        updateWindowTitle();
    }

    callbackList.invokeCallbacks(itemSwitch->getChild(selectedItem));
}

//
// redefine those generic virtual functions
//
const char *
MyMaterialPalette::getDefaultWidgetName() const
{
    return "MyMaterialPalette";
}

const char *
MyMaterialPalette::getDefaultTitle() const
{
    return "Material Palette Gizmo";
}

const char *
MyMaterialPalette::getDefaultIconTitle() const
{
    return "Mat Palette";
}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

SbBool
MyMaterialPalette::raEventCB(void *p, XAnyEvent *xe)
{
    return (((MyMaterialPalette *)p)->handleEvent(xe));
}

//
// called when the save dialog "ok"/"Cancel" buttons gets pressed.
//
void
MyMaterialPalette::saveDialogCB(Widget dialog, MyMaterialPalette *p,
                                XmAnyCallbackStruct *cb)
{
    // save the palette and destroy the dialog
    if (cb->reason == XmCR_OK)
        p->savePalette();

    XtDestroyWidget(dialog);

    // now check what needs to be done, now that the dialog is gone
    switch (p->whatToDoNext)
    {
    case BRING_NEW_DIALOG:
        p->whatToDoNext = CREATE_NEW_PALETTE;
        p->createPromptDialog("New Palette Dialog", "New Palette Name:");
        break;

    case SWITCH_PALETTE:
        p->switchPalette();
        break;
    }
}

//
// called when the delete dialog "ok"/"Cancel" buttons gets pressed. This
// will delete the palette in the user's home directory.
//
void
MyMaterialPalette::deleteDialogCB(Widget dialog, MyMaterialPalette *p,
                                  XmAnyCallbackStruct *cb)
{
    // remove the user's palette
    if (cb->reason == XmCR_OK)
    {

        PaletteStruct *pal = (PaletteStruct *)p->paletteList[p->curPalette];

        // remove the palette directory in user's home
        char palDir[MAXPATHLEN];
        sprintf(palDir, "%s/%s/%s", getenv("HOME"), customPalDir, pal->name);
        unlink(palDir);

        // check if palette was also in the installed place (reset vs delete)
        if (pal->system)
            pal->user = FALSE;
        else
        {
            // remove the palette from the list, making sure we have
            // at least one palette entry
            delete[] pal -> name;
            if (p->paletteList.getLength() == 1)
            {
                // create an empty default palette
                pal->name = STRDUP("default");
            }
            else
            {
                delete pal;
                p->paletteList.remove(p->curPalette);

                // check what palette will be next
                if (p->curPalette == p->paletteList.getLength())
                    p->curPalette--;
            }

            // rebuild the new menu ( ??? have to rebuild, since we can't
            // remove entries)
            p->buildPaletteSubMenu();
        }

        // finaly load the new palette
        p->nextPalette = p->curPalette;
        p->switchPalette();
    }

    XtDestroyWidget(dialog);
}

//
// called whenever the "ok"/"Cancel" buttons within the generic prompt
// dialog get pressed.
//
void
MyMaterialPalette::promptDialogCB(Widget dialog, MyMaterialPalette *p,
                                  XmAnyCallbackStruct *cb)
{
    if (cb->reason == XmCR_OK)
    {

        // retreive text
        Widget field = XmSelectionBoxGetChild(dialog, XmDIALOG_TEXT);
        char *str = XmTextGetString(field);

        // do the right thing
        if (str[0] != '\0')
        {
            switch (p->whatToDoNext)
            {
            case CREATE_NEW_PALETTE:
                p->createNewPalette(str);
                break;

            case SAVE_AS_PALETTE:
                p->savePaletteAs(str);
                break;
            }
        }

        XtFree(str);
    }

    XtDestroyWidget(dialog);
}

//
// called whenever the material editor apply button gets pressed.
// reteive the material and material name, and invoke the
// callbaks.
//
void
MyMaterialPalette::matEditorCB(void *pt, MySimpleMaterialEditor *ed)
{
    MyMaterialPalette *p = (MyMaterialPalette *)pt;

    // no material currently selected
    if (p->selectedItem < 0)
        return;

    // copy the material over
    SoMaterial *mat1 = (SoMaterial *)p->itemSwitch->getChild(p->selectedItem);
    const SoMaterial *mat2 = ed->getMaterial();
    mat1->ambientColor = mat2->ambientColor[0];
    mat1->diffuseColor = mat2->diffuseColor[0];
    mat1->specularColor = mat2->specularColor[0];
    mat1->emissiveColor = mat2->emissiveColor[0];
    mat1->shininess = mat2->shininess[0];
    mat1->transparency = mat2->transparency[0];

    // copy the material name over and update label (if name has changed)
    const char *str = ed->getMaterialName();
    if (str != NULL && strcmp(str, p->mtlNames[p->selectedItem].name) != 0)
    {

        // save the old material name file for later removal
        if (p->mtlNames[p->selectedItem].oldName == NULL)
            p->mtlNames[p->selectedItem].oldName = p->mtlNames[p->selectedItem].name;
        else
            delete[] p -> mtlNames[p->selectedItem].name;

        p->mtlNames[p->selectedItem].name = STRDUP(str);
        p->updateMaterialName();
    }

    if (!p->paletteChanged)
    {
        p->paletteChanged = TRUE;
        p->updateWindowTitle();
    }

    // finally invoke the callbacks
    p->callbackList.invokeCallbacks((void *)mat1);
}

//
//  Called whenever a new item menu is selected from the palette
//  popup menu.
//
void
MyMaterialPalette::paletteMenuCB(Widget w, int num, void *)
{
    // get the class pointer
    MyMaterialPalette *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    // return if the same palette is choosen
    if (p->curPalette == num)
        return;

    // save the new palette id for later uses
    p->nextPalette = num;

    // check to make sure the old palette hasn't changed withought
    // being saved.
    if (p->paletteChanged)
    {
        p->createSaveDialog();
        // this will call switchPalette() once the dialog disapears
        p->whatToDoNext = SWITCH_PALETTE;
    }
    else
        p->switchPalette();
}

//
// called whenever the X server is done doing the paste
//
void
MyMaterialPalette::pasteDone(void *pt, SoPathList *pathList)
{
    MyMaterialPalette *p = (MyMaterialPalette *)pt;

    SoSearchAction sa;
    SoFullPath *fullPath = NULL;

    //
    // search for first material in that pasted scene
    //
    sa.setType(SoMaterial::getClassTypeId());
    for (int i = 0; i < pathList->getLength(); i++)
    {
        sa.apply((*pathList)[i]);
        if ((fullPath = (SoFullPath *)sa.getPath()) != NULL)
        {

            // assign new material, update editor and invoke callback
            SoMaterial *newMat = (SoMaterial *)fullPath->getTail();
            p->itemSwitch->replaceChild(p->selectedItem, newMat);
            if (p->matEditor != NULL && p->matEditor->isVisible())
                p->matEditor->setMaterial(newMat);
            if (!p->paletteChanged)
            {
                p->paletteChanged = TRUE;
                p->updateWindowTitle();
            }
            p->callbackList.invokeCallbacks(newMat);

            break;
        }
    }

    // ??? We delete the callback data when done with it.
    delete pathList;
}
