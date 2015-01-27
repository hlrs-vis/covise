/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvTextureEd.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <covise/covise.h>
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>

#include <X11/StringDefs.h>
#include <Xm/Xm.h>
#include <Xm/Form.h>
#include <Xm/RowColumn.h>
#include <Xm/CascadeBG.h>
#include <Xm/PushBG.h>
#include <Xm/LabelG.h>
#include <Xm/Text.h>
#include <Xm/Scale.h>
#include <Xm/SelectioB.h>
#include <Xm/MessageB.h>
#include <Xm/FileSB.h>
#include "GLwMDrawA.h"

#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/nodes/SoTexture2.h>
#include <Inventor/nodes/SoTexture2Transform.h>
#include <Inventor/nodes/SoTextureCoordinateDefault.h>
#include <Inventor/nodes/SoTextureCoordinateEnvironment.h>
#include <Inventor/nodes/SoTextureCoordinatePlane.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>
#include <Inventor/errors/SoDebugError.h>
#include <GL/gl.h>

#include "InvColorWheel.h"
#include "InvColorSlider.h"
#include "InvThumbWheel.h"
#include "InvTextureEditor.h"

#if defined(__hpux) || defined(__linux__)
#define getwd(p) \
    ;            \
    getcwd(p, 1024);
#endif
extern char *STRDUP(const char *s);
// stuff to read images (similar to gl/image.h)
extern "C" {
#define CM_NORMAL 0
/* file contains rows of values which
    * are either RGB values (zsize == 3)
    * or greyramp values (zsize == 1) */
typedef struct
{
    unsigned short imagic; /* stuff saved on disk . . */
    unsigned short type;
    unsigned short dim;
    unsigned short xsize;
    unsigned short ysize;
    unsigned short zsize;
    unsigned long min;
    unsigned long max;
    unsigned long wastebytes;
    char name[80];
    unsigned long colormap;
} IMAGE;
extern IMAGE *iopen(const char *, const char *);
extern void getrow(IMAGE *, short *, int, int);
extern void iclose(IMAGE *);
#if !defined(__hpux) && !defined(__linux__)
extern void i_seterror(void (*func)(char *));
#endif
};

/*
 * Defines
 */

enum WidgetsID
{
    PALETTE_BUTTON = 0, // must start at 0 since we use array
    PALETTE_MENU,
    MENU_BAR,

    // file menu
    FILE_NEW,
    FILE_RESET,
    FILE_DELETE,

    // text field widgets
    SCALE_X_FIELD,
    SCALE_Y_FIELD,
    TRANS_X_FIELD,
    TRANS_Y_FIELD,
    ROT_FIELD,

    // sliders widgets
    TRANS_X_SLD,
    TRANS_Y_SLD,
    ROT_SLD,
    SCALE_X_LABEL,
    SCALE_Y_LABEL,

    ACCEPT,
    TEXTURE_GLX,
    TEXTURE_NAME,

    // mapping option menu types
    MAPP_PULLDOWN,
    MAP_DEFAULT,
    MAP_ENV,
    MAP_PLANE_XY,
    MAP_PLANE_XZ,
    MAP_PLANE_YZ,
    MAP_UNKNOWN,

    // other option menu types
    OPT_PULLDOWN,
    OPT_REPEAT,
    OPT_CLAMP,
    OPT_UNKNOWN,

    // image dialog widgets
    DIALOG_WINDOW,
    DIALOG_IMAGE,
    DIALOG_NAME,
    DIALOG_INFO,
    DIALOG_BUTTON_0,
    DIALOG_FILE_BROWSER,

    NUM // this must be the last entry
};

// time in msec between consecutive clicks to consider
// the second click being a double click.
#define DOUBLE_CLICK_TIME 300

enum TileDrawStyle
{
    DRAW_NORM,
    DRAW_CURRENT,
    DRAW_SELECTED
};

#define IMAGE_SIZE 38
#define IMAGE_NUM 5
#define IMAGE_TOTAL (IMAGE_NUM * IMAGE_NUM)
#define IMAGE_SPACE 2
#define GLX_SIZE (IMAGE_NUM * IMAGE_SIZE + (IMAGE_NUM + 1) * IMAGE_SPACE)

// ??? doing a GL_LINE_LOOP seems to be missing the top right
// ??? pixel due to subpixel == TRUE in openGL.
#define RECT(x1, y1, x2, y2) \
    glBegin(GL_LINE_STRIP);  \
    glVertex2s(x2, y2);      \
    glVertex2s(x1, y2);      \
    glVertex2s(x1, y1);      \
    glVertex2s(x2, y1);      \
    glVertex2s(x2, y2 + 1);  \
    glEnd();

struct TextureNameStruct
{
    char *name;
    char *fullName;
    int zsize;
    char *iconImage;
};

struct PaletteStruct
{
    char *name;
    SbBool user;
    SbBool system;
};

/*
 * static vars
 */

static const char *customTextureDir = ".textures";
static const char *defaultDir = "/usr/share/data/textures";
static const char *editorTitle = "Texture Editor";
static const char *noFileNameStr = "<empty>";

#define hourglass_width 32
#define hourglass_height 32
#define hourglass_x_hot 16
#define hourglass_y_hot 16
static unsigned char hourglass_bits[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xfe, 0xff, 0x00, 0x00, 0x04, 0x40, 0x00, 0x00, 0x04, 0x40, 0x00,
    0x00, 0x04, 0x40, 0x00, 0x00, 0xe8, 0x2e, 0x00, 0x00, 0xd0, 0x17, 0x00,
    0x00, 0xa0, 0x0b, 0x00, 0x00, 0x40, 0x05, 0x00, 0x00, 0x40, 0x05, 0x00,
    0x00, 0x40, 0x04, 0x00, 0x00, 0x40, 0x04, 0x00, 0x00, 0x40, 0x04, 0x00,
    0x00, 0x20, 0x09, 0x00, 0x00, 0x10, 0x11, 0x00, 0x00, 0x88, 0x23, 0x00,
    0x00, 0xc4, 0x47, 0x00, 0x00, 0xe4, 0x4f, 0x00, 0x00, 0xf4, 0x5f, 0x00,
    0x00, 0xfe, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

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

#if !defined(__hpux) && !defined(__linux__)
static void imageErrorHandler(char *)
{
}
#endif

////////////////////////////////////////////////////////////////////////
//
// constructor
//
MyTextureEditor::MyTextureEditor(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    const char *dir)
    //
    ////////////////////////////////////////////////////////////////////////
    : SoXtComponent(parent, name, buildInsideParent)
{
    int i;

    setClassName("MyTextureEditor");
    paletteDir = (dir != NULL) ? STRDUP(dir) : STRDUP(defaultDir);
    ignoreCallback = FALSE;
    selectedItem = currentItem = -1;
    curPalette = -1;
    loadedPalette = FALSE;

    // widget vars
    widgetList = new Widget[NUM];
    for (i = 0; i < NUM; i++)
        widgetList[i] = NULL;
    oldXThumbVal = oldYThumbVal = 0;
    fieldChanged = FALSE;
    imageDialogCtx = 0;

    // image dialog vars
    dialogImage = NULL;
    dialogImageName = NULL;
    dialogImageInfo = NULL;

    // scene graph vars
    sceneRoot = new SoSeparator;
    texXfNode = new SoTexture2Transform;
    texFuncNode = new SoTextureCoordinateDefault;
    texNode = new SoTexture2;
    sceneRoot->ref();
    sceneRoot->addChild(texFuncNode);
    sceneRoot->addChild(texXfNode);
    sceneRoot->addChild(texNode);
    texXfNode->center.setValue(SbVec2f(.5, .5));
    //???alain
    //    if (getgdesc(GD_TEXTURE))
    if (1)
        userGeometry = new SoCube;
    else
    {
        SoText2 *text = new SoText2;
        text->string.set1Value(0, SbString("Texture not visible"));
        text->string.set1Value(1, SbString("on this machine"));
        text->spacing = 2;
        text->justification = SoText2::CENTER;
        userGeometry = text;
    }
    sceneRoot->addChild(userGeometry);

    // allocate the texture name list
    textureNames = new TextureNameStruct[IMAGE_TOTAL];
    for (i = 0; i < IMAGE_TOTAL; i++)
    {
        textureNames[i].name = textureNames[i].fullName = NULL;
        // for RGB
        textureNames[i].iconImage = new char[IMAGE_SIZE * IMAGE_SIZE * 3];
    }

// set the image lib error handler to prevent iopen() to exit()
// when given a non image fileName.
#if !defined(__hpux) && !defined(__linux__)
    i_seterror(imageErrorHandler);
#endif

    // Build the widget tree, and let SoXtComponent know about our base widget.
    getPaletteNames();
    setBaseWidget(buildWidget(getParentWidget()));
}

////////////////////////////////////////////////////////////////////////
//
//    Destructor.
//
MyTextureEditor::~MyTextureEditor()
//
////////////////////////////////////////////////////////////////////////
{
    int i;

    // delete components
    delete colWheel;
    delete colSlider;
    delete viewer;
    delete scaleXThumb;
    delete scaleYThumb;

    // delete stuff
    delete[] widgetList;
    delete[] paletteDir;

    // delete palette names
    PaletteStruct *pal;
    for (i = 0; i < paletteList.getLength(); i++)
    {
        pal = (PaletteStruct *)paletteList[i];
        delete[] pal -> name;
        delete pal;
    }
    paletteList.truncate(0);

    // delete texture names
    for (i = 0; i < IMAGE_TOTAL; i++)
    {
        deleteTextureEntry(i);
        delete[] textureNames[i].iconImage;
    }
    delete[] textureNames;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets the geometry to use
//
// Use: public
void
MyTextureEditor::setObjectGeometry(SoNode *newGeom)
//
////////////////////////////////////////////////////////////////////////
{
    //???alain
    //    if (! getgdesc(GD_TEXTURE))
    //	return;

    // remove the old geometry, and replace it with supplied one
    sceneRoot->replaceChild(userGeometry, newGeom);
    userGeometry = newGeom;
    viewer->viewAll();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets the SoTexture2 node to use.
//
// Use: public
void
MyTextureEditor::setTextureNode(const SoTexture2 *node)
//
////////////////////////////////////////////////////////////////////////
{
    // copies values over
    texNode->filename = node->filename.getValue();
    texNode->wrapS = node->wrapS.getValue();
    texNode->wrapT = node->wrapT.getValue();
    texNode->model = node->model.getValue();
    texNode->blendColor = node->blendColor.getValue();

    // update the UI and unselect current texture
    updateTexture2UI();
    deselectCurrentItem();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets the SoTexture2 Transform node to use.
//
// Use: public
void
MyTextureEditor::setTransformNode(const SoTexture2Transform *node)
//
////////////////////////////////////////////////////////////////////////
{
    // copies values over
    texXfNode->translation = node->translation.getValue();
    texXfNode->rotation = node->rotation.getValue();
    texXfNode->scaleFactor = node->scaleFactor.getValue();
    texXfNode->center = node->center.getValue();

    //
    // constrain the transform node to our paradigm
    //

    // change the center of rotation/scale to make it look like
    // we are rotation around the center of the image
    texXfNode->center.setValue(
        SbVec2f(.5, .5) - texXfNode->translation.getValue());

    // normalize rotation to [0,2PI]
    float val = texXfNode->rotation.getValue();
    while (val < 0)
        val += 2 * M_PI;
    while (val > 2 * M_PI)
        val -= 2 * M_PI;
    texXfNode->rotation = val;

    // update the UI
    updateTextureXfUI();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets the SoTextureCoordinateFunction node to use.
//
// Use: public
void
MyTextureEditor::setFunctionNode(const SoTextureCoordinateFunction *node)
//
////////////////////////////////////////////////////////////////////////
{
    // make our local func node reflect the given node
    SoNode *newFunc = (node != NULL) ? node->copy() : new SoTextureCoordinateDefault;
    sceneRoot->replaceChild(texFuncNode, newFunc);
    texFuncNode = (SoTextureCoordinateFunction *)newFunc;

    // finally update the UI
    updateTextureFuncUI();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Builds the editor layout
//
// Use: protected
Widget
MyTextureEditor::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[12];

    // create a top level form to hold everything together
    Widget form = XmCreateForm(parent, (char *)"textureForm", NULL, 0);

    //
    // create all the parts
    //

    // create all the comps
    Widget menu = buildMenu(form);

    colWheel = new MyColorWheel(form);
    colWheel->setSize(SbVec2s(110, 110));
    colWheel->setWYSIWYG(TRUE);
    colWheel->addValueChangedCallback(
        MyTextureEditor::colWheelCB, this);
    Widget wheelW = colWheel->getWidget();

    colSlider = new MyColorSlider(form, NULL, TRUE, MyColorSlider::VALUE_SLIDER);
    colSlider->setNumericFieldVisible(FALSE);
    colSlider->setSize(SbVec2s(1, 23));
    colSlider->addValueChangedCallback(
        MyTextureEditor::colSliderCB, this);
    colSlider->setWYSIWYG(TRUE);
    Widget sliderW = colSlider->getWidget();

    viewer = new SoXtExaminerViewer(form, NULL, TRUE, SoXtFullViewer::BUILD_NONE);
    viewer->setSize(SbVec2s(150, 150));
    viewer->setSceneGraph(sceneRoot);
    sceneRoot->unref();
    Widget viewerW = viewer->getWidget();

    Widget glx = buildTexturePaletteWidget(form);
    Widget sldForm = buildSliders(form);
    Widget butForm = buildButtons(form);

    n = 0;
    XtSetArg(args[n], XmNalignment, XmALIGNMENT_CENTER);
    n++;
    Widget textName = XmCreateLabelGadget(form, (char *)"name", args, n);
    widgetList[TEXTURE_NAME] = textName;

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
    XtSetValues(glx, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, menu);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, glx);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetValues(viewerW, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, glx);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightWidget, glx);
    n++;
    XtSetValues(textName, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, butForm);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 5);
    n++;
    XtSetValues(sldForm, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, viewerW);
    n++;
    XtSetArg(args[n], XmNtopOffset, 2);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, viewerW);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetValues(butForm, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, butForm);
    n++;
    XtSetArg(args[n], XmNtopOffset, 10);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, sldForm);
    n++;
    XtSetArg(args[n], XmNleftOffset, 20);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, 15);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 7 + 23);
    n++;
    XtSetValues(wheelW, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, wheelW);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, wheelW);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightWidget, wheelW);
    n++;
    XtSetValues(sliderW, args, n);

    updateTexture2UI();
    updateTextureXfUI();
    updateTextureFuncUI();
    updateTextureName();
    updateWindowTitle();
    updateFileMenu();

    // manage children
    XtManageChild(menu);
    XtManageChild(glx);
    XtManageChild(textName);
    XtManageChild(viewerW);
    XtManageChild(butForm);
    XtManageChild(sldForm);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	creates the menu bar
//
// Use: private
Widget
MyTextureEditor::buildMenu(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[8];
    Widget buttons[10], subButtons[10], subMenu;
    XmString xmstr;
    int n, butNum = 0, subButNum;

    Widget menu = XmCreateMenuBar(parent, (char *)"menuBar", NULL, 0);
    widgetList[MENU_BAR] = menu;

    //
    // create the "File" menu
    //

    subMenu = XmCreatePulldownMenu(menu, (char *)"", NULL, 0);

    XtSetArg(args[0], XmNsubMenuId, subMenu);
    buttons[butNum] = XmCreateCascadeButtonGadget(menu, (char *)"File", args, 1);

    // create the menu entries
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;

#define ADD_ENTRY(NAME, ID, ACC, ACCTEXT)                                                                                          \
    xmstr = XmStringCreateSimple((char *)ACCTEXT);                                                                                 \
    XtSetArg(args[n], XmNaccelerator, (char *)ACC);                                                                                \
    XtSetArg(args[n + 1], XmNacceleratorText, xmstr);                                                                              \
    subButtons[subButNum] = XmCreatePushButtonGadget(subMenu, (char *)NAME, args, n + 2);                                          \
    XtAddCallback(subButtons[subButNum], XmNactivateCallback, (XtCallbackProc)MyTextureEditor::fileMenuCB, (XtPointer)(char *)ID); \
    XmStringFree(xmstr);                                                                                                           \
    widgetList[ID] = subButtons[subButNum++];

    subButNum = 0;
    ADD_ENTRY("New...", FILE_NEW, "Alt <Key> n", "Alt+n")
    ADD_ENTRY("Reset", FILE_RESET, "Alt <Key> r", "Alt+r")
    ADD_ENTRY("Delete", FILE_DELETE, "Alt <Key> d", "Alt+d")
#undef ADD_ENTRY

    XtManageChildren(subButtons, subButNum);
    butNum++;

    //
    // create the "Palette" menu
    //

    widgetList[PALETTE_BUTTON] = buttons[butNum] = XmCreateCascadeButtonGadget(
        menu, (char *)"Palettes", NULL, 0);
    buildPaletteSubMenu();
    butNum++;

    XtManageChildren(buttons, butNum);

    return menu;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	adds an entry to the palette menu.
//
// Use: private
Widget
MyTextureEditor::buildPaletteMenuEntry(long id)
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

    Widget w = XmCreatePushButtonGadget(widgetList[PALETTE_MENU], pal->name, args, n);
    XtAddCallback(w, XmNactivateCallback,
                  (XtCallbackProc)MyTextureEditor::paletteMenuCB,
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
MyTextureEditor::buildPaletteSubMenu()
//
////////////////////////////////////////////////////////////////////////
{
    // rebuild the palette popup
    // ??? we cannot delete the old popup menu or things will brake.
    // ??? Is it automatically deleted for us ?
    widgetList[PALETTE_MENU] = XmCreatePulldownMenu(widgetList[MENU_BAR],
                                                    (char *)"", NULL, 0);

    Arg args[1];
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
//  	Builds the GLX region where the textures will be displayed
//
// Use: private
Widget
MyTextureEditor::buildTexturePaletteWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[12];
    int n = 0;

    XtSetArg(args[n], XtNwidth, GLX_SIZE);
    n++;
    XtSetArg(args[n], XtNheight, GLX_SIZE);
    n++;
    XtSetArg(args[n], GLwNrgba, TRUE);
    n++;
    // This makes sure we get the maximum buffer configuration
    // (by showing interest the number of bits)
    XtSetArg(args[n], GLwNredSize, 1);
    n++;
    XtSetArg(args[n], GLwNgreenSize, 1);
    n++;
    XtSetArg(args[n], GLwNblueSize, 1);
    n++;

    Widget glx = XtCreateWidget("paletteGLX", glwMDrawingAreaWidgetClass,
                                parent, args, n);
    widgetList[TEXTURE_GLX] = glx;

    XtUninstallTranslations(glx);

    XtAddCallback(glx, GLwNginitCallback,
                  (XtCallbackProc)MyTextureEditor::glxInitCB, (XtPointer) this);
    XtAddCallback(glx, GLwNexposeCallback,
                  (XtCallbackProc)MyTextureEditor::glxExposeCB, (XtPointer) this);
    XtAddEventHandler(glx,
                      (PointerMotionMask | ButtonPressMask | ButtonReleaseMask | LeaveWindowMask),
                      FALSE, (XtEventHandler)MyTextureEditor::glxEventCB, (XtPointer) this);

    return glx;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Builds the sliders into a form.
//
// Use: private
Widget
MyTextureEditor::buildSliders(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int i, n, num;
    Arg args[12];
    Widget labels[5], fields[5], sliders[5];

    // create a form to hold everything together
    Widget form = XmCreateForm(parent, (char *)"slidersForm", NULL, 0);

    //
    // create all the parts
    //

    repeatState = TRUE; // build the UI with repeat turned on (default)

    // create all of the labels
    n = 0;
    XtSetArg(args[n], XmNalignment, XmALIGNMENT_END);
    n++;
    labels[0] = XmCreateLabelGadget(form, (char *)"Translate X:", args, n);
    labels[1] = XmCreateLabelGadget(form, (char *)"Translate Y:", args, n);
    labels[2] = XmCreateLabelGadget(form, (char *)"Rotate:", args, n);
    widgetList[SCALE_X_LABEL] = labels[3] = XmCreateLabelGadget(form, (char *)"Repeat X:", args, n);
    widgetList[SCALE_Y_LABEL] = labels[4] = XmCreateLabelGadget(form, (char *)"Repeat Y:", args, n);

    // create all of the text fields
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 1);
    n++;
    XtSetArg(args[n], XmNcolumns, 5);
    n++;

#define CREATE_FIELD(ID)                                                                                                    \
    widgetList[ID] = fields[num] = XmCreateText(form, (char *)"", args, n);                                                 \
    XtAddCallback(fields[num], XmNvalueChangedCallback, (XtCallbackProc)MyTextureEditor::fieldChangedCB, (XtPointer) this); \
    XtAddCallback(fields[num], XmNactivateCallback, (XtCallbackProc)MyTextureEditor::fieldsCB, (XtPointer)ID);              \
    XtAddCallback(fields[num], XmNlosingFocusCallback, (XtCallbackProc)MyTextureEditor::fieldsCB, (XtPointer)ID);           \
    num++;

    num = 0;
    CREATE_FIELD(TRANS_X_FIELD)
    CREATE_FIELD(TRANS_Y_FIELD)
    CREATE_FIELD(ROT_FIELD)
    CREATE_FIELD(SCALE_X_FIELD)
    CREATE_FIELD(SCALE_Y_FIELD)
#undef CREATE_FIELD

    // create all of the sliders
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNorientation, XmHORIZONTAL);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNwidth, 100);
    n++;
    XtSetArg(args[n], XmNheight, 20);
    n++;

#define CREATE_SCALE(ID)                                                                                             \
    widgetList[ID] = sliders[num] = XmCreateScale(form, (char *)"", args, n);                                        \
    XtAddCallback(sliders[num], XmNvalueChangedCallback, (XtCallbackProc)MyTextureEditor::slidersCB, (XtPointer)ID); \
    XtAddCallback(sliders[num++], XmNdragCallback,                                                                   \
                  (XtCallbackProc)MyTextureEditor::slidersCB, (XtPointer)ID);

    num = 0;
    CREATE_SCALE(TRANS_X_SLD)
    CREATE_SCALE(TRANS_Y_SLD)
    CREATE_SCALE(ROT_SLD)

    scaleXThumb = new MyThumbWheel(form);
    scaleXThumb->setSize(SbVec2s(100, 21));
    scaleXThumb->addValueChangedCallback(
        MyTextureEditor::scaleXThumbCB, this);

    scaleYThumb = new MyThumbWheel(form);
    scaleYThumb->setSize(SbVec2s(100, 21));
    scaleYThumb->addValueChangedCallback(
        MyTextureEditor::scaleYThumbCB, this);

    sliders[num++] = scaleXThumb->getWidget();
    sliders[num++] = scaleYThumb->getWidget();
#undef CREATE_SCALE

    //
    // layout !
    //

    // text field first
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(fields[0], args, n);
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopOffset, 1);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    for (i = 1; i < 5; i++)
    {
        XtSetArg(args[n], XmNtopWidget, fields[i - 1]);
        XtSetValues(fields[i], args, n + 1);
    }

    // sliders
    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightOffset, 1);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    for (i = 0; i < 5; i++)
    {
        XtSetArg(args[n], XmNrightWidget, fields[i]);
        XtSetArg(args[n + 1], XmNbottomWidget, fields[i]);
        XtSetArg(args[n + 2], XmNbottomOffset, (i < 3) ? 7 : 4);
        XtSetValues(sliders[i], args, n + 3);
    }

    // labels (centered around text fields)
    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightOffset, 3);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    for (i = 0; i < 5; i++)
    {
        XtSetArg(args[n], XmNrightWidget, sliders[i]);
        XtSetArg(args[n + 1], XmNtopWidget, fields[i]);
        XtSetArg(args[n + 2], XmNbottomWidget, fields[i]);
        XtSetValues(labels[i], args, n + 3);
    }

    // manage children
    XtManageChildren(fields, 5);
    XtManageChildren(sliders, 5);
    XtManageChildren(labels, 5);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Builds the accept + option buttons
//
// Use: private
Widget
MyTextureEditor::buildButtons(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget acceptBut, labels[2], buttons[2];
    Arg args[12];
    int i, n;

    // create a form to hold everything together
    Widget form = XmCreateForm(parent, (char *)"buttonForm", NULL, 0);

    //
    // create all the parts
    //

    // create the accept button
    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    acceptBut = XmCreatePushButtonGadget(form, (char *)"Accept", args, n);
    XtAddCallback(acceptBut, XmNactivateCallback,
                  (XtCallbackProc)MyTextureEditor::acceptCB, (XtPointer) this);

    // create the labels
    n = 0;
    XtSetArg(args[n], XmNalignment, XmALIGNMENT_END);
    n++;
    labels[0] = XmCreateLabelGadget(form, (char *)"Mapping:", args, n);
    labels[1] = XmCreateLabelGadget(form, (char *)"Options:", args, n);
    // make all the labels the same width (for layout)
    short w;
    int width = 0;
    for (i = 0; i < 2; i++)
    {
        XtVaGetValues(labels[i], XtNwidth, &w, NULL);
        if (w > width)
            width = w;
    }
    for (i = 0; i < 2; i++)
        XtVaSetValues(labels[i], XtNwidth, width, NULL);

    // create the mapping button option
    Widget list[15];
    int num;

    widgetList[MAPP_PULLDOWN] = XmCreatePulldownMenu(form, (char *)"", NULL, 0);
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;

#define ADD_ENTRY(NAME, ID)                                                                                  \
    widgetList[ID] = list[num] = XmCreatePushButtonGadget(widgetList[MAPP_PULLDOWN], (char *)NAME, args, n); \
    XtAddCallback(list[num++], XmNactivateCallback,                                                          \
                  (XtCallbackProc)MyTextureEditor::mappingMenuCB, (XtPointer)ID);

    num = 0;
    ADD_ENTRY("default", MAP_DEFAULT)
    ADD_ENTRY("reflection", MAP_ENV)
    ADD_ENTRY("xy plane", MAP_PLANE_XY)
    ADD_ENTRY("yz plane", MAP_PLANE_YZ)
    ADD_ENTRY("xz plane", MAP_PLANE_XZ)
    ADD_ENTRY("unknown", MAP_UNKNOWN)
#undef ADD_ENTRY

    XtVaSetValues(widgetList[MAP_UNKNOWN], XmNsensitive, FALSE, NULL);
    XtManageChildren(list, num);

    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNmarginWidth, 0);
    n++;
    XtSetArg(args[n], XmNmarginHeight, 0);
    n++;
    XtSetArg(args[n], XmNsubMenuId, widgetList[MAPP_PULLDOWN]);
    n++;
    buttons[0] = XmCreateOptionMenu(form, (char *)"optionMenu", args, n);

    // create the options button
    widgetList[OPT_PULLDOWN] = XmCreatePulldownMenu(form, (char *)"", NULL, 0);
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;

#define ADD_ENTRY(NAME, ID)                                                                                 \
    widgetList[ID] = list[num] = XmCreatePushButtonGadget(widgetList[OPT_PULLDOWN], (char *)NAME, args, n); \
    XtAddCallback(list[num++], XmNactivateCallback,                                                         \
                  (XtCallbackProc)MyTextureEditor::optionMenuCB, (XtPointer)ID);

    num = 0;
    ADD_ENTRY("repeat", OPT_REPEAT)
    ADD_ENTRY("clamp", OPT_CLAMP)
    ADD_ENTRY("unknown", OPT_UNKNOWN)
#undef ADD_ENTRY

    XtVaSetValues(widgetList[OPT_UNKNOWN], XmNsensitive, FALSE, NULL);
    XtManageChildren(list, num);

    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNmarginWidth, 0);
    n++;
    XtSetArg(args[n], XmNmarginHeight, 0);
    n++;
    XtSetArg(args[n], XmNsubMenuId, widgetList[OPT_PULLDOWN]);
    n++;
    buttons[1] = XmCreateOptionMenu(form, (char *)"optionMenu", args, n);

    //
    // layout !
    //

    // center the accept button
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNleftPosition, 50);
    n++;
    XtSetArg(args[n], XmNleftOffset, -30);
    n++;
    XtSetValues(acceptBut, args, n);

    // attach option menu buttons to the right
    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;

    XtSetArg(args[n], XmNtopOffset, 5);
    XtSetArg(args[n + 1], XmNtopWidget, acceptBut);
    XtSetValues(buttons[0], args, n + 2);
    XtSetArg(args[n], XmNtopWidget, buttons[0]);
    XtSetValues(buttons[1], args, n + 1);

    // center the labels around the option menu
    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;

    XtSetArg(args[n], XmNrightWidget, buttons[0]);
    XtSetArg(args[n + 1], XmNtopWidget, buttons[0]);
    XtSetArg(args[n + 2], XmNbottomWidget, buttons[0]);
    XtSetValues(labels[0], args, n + 3);
    XtSetArg(args[n], XmNrightWidget, buttons[1]);
    XtSetArg(args[n + 1], XmNtopWidget, buttons[1]);
    XtSetArg(args[n + 2], XmNbottomWidget, buttons[1]);
    XtSetValues(labels[1], args, n + 3);

    // manage children
    XtManageChild(acceptBut);
    XtManageChildren(buttons, 2);
    XtManageChildren(labels, 2);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	redraws the texture palette
//
// Use: private
void
MyTextureEditor::redrawPalette()
//
////////////////////////////////////////////////////////////////////////
{
    if (!isVisible())
        return;

    glXMakeCurrent(getDisplay(), XtWindow(widgetList[TEXTURE_GLX]), paletteCtx);

    for (int i = 0; i < IMAGE_TOTAL; i++)
    {
        if (i != currentItem && i != selectedItem)
            drawTextureTile(i, DRAW_NORM);
    }
    if (currentItem != -1 && currentItem != selectedItem)
        drawTextureTile(currentItem, DRAW_CURRENT);
    if (selectedItem != -1)
        drawTextureTile(selectedItem, DRAW_SELECTED);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	draws the given texture tile
//
// Use: private
void
MyTextureEditor::drawTextureTile(int id, int style)
//
////////////////////////////////////////////////////////////////////////
{
    if (id < 0)
        return;

    int row = int(id / IMAGE_NUM);
    int col = id - IMAGE_NUM * row;
    int s = IMAGE_SPACE + IMAGE_SIZE;
    int x1 = col * s + IMAGE_SPACE;
    int x2 = x1 + IMAGE_SIZE - 1;
    int y1 = (IMAGE_NUM - row - 1) * s + IMAGE_SPACE;
    int y2 = y1 + IMAGE_SIZE - 1;

    // paste the texture image
    if (textureNames[id].name != NULL)
    {
        glRasterPos2i(x1, y1);
        glDrawPixels(IMAGE_SIZE, IMAGE_SIZE, GL_RGB, GL_UNSIGNED_BYTE,
                     textureNames[id].iconImage);
    }
    else
    {
        glColor3ub(150, 150, 150);
        glBegin(GL_POLYGON);
        glVertex2s(x1, y1);
        glVertex2s(x2 + 1, y1);
        glVertex2s(x2 + 1, y2 + 1);
        glVertex2s(x1, y2 + 1);
        glEnd();
    }

    // show the boder highlighting
    switch (style)
    {
    case DRAW_NORM:
        glColor3ub(150, 150, 150);
        RECT(x1 - 2, y1 - 2, x2 + 2, y2 + 2);
        RECT(x1 - 1, y1 - 1, x2 + 1, y2 + 1);
        break;
    case DRAW_CURRENT:
        glColor3ub(230, 50, 50);
        RECT(x1 - 2, y1 - 2, x2 + 2, y2 + 2);
        RECT(x1 - 1, y1 - 1, x2 + 1, y2 + 1);
        break;
    case DRAW_SELECTED:
        glColor3ub(230, 50, 50);
        RECT(x1 - 2, y1 - 2, x2 + 2, y2 + 2);
        RECT(x1 - 1, y1 - 1, x2 + 1, y2 + 1);
        RECT(x1, y1, x2, y2);
        RECT(x1 + 1, y1 + 1, x2 - 1, y2 - 1);
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	update the texture name based of what is selected
//
// Use: private
void
MyTextureEditor::updateTextureName()
//
////////////////////////////////////////////////////////////////////////
{
    char *str = (currentItem < 0) ? ((selectedItem < 0) ? STRDUP(" ") : textureNames[selectedItem].name) : textureNames[currentItem].name;
    if (str == NULL)
        str = (char *)noFileNameStr;
    XmString xmstr = XmStringCreateSimple(str);
    XtVaSetValues(widgetList[TEXTURE_NAME], XmNlabelString, xmstr, NULL);
    XmStringFree(xmstr);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	update the texture node based on currently selected texture
//
// Use: private
void
MyTextureEditor::updateTextureNode()
//
////////////////////////////////////////////////////////////////////////
{
    if (selectedItem < 0)
        return;

    TextureNameStruct *txt = &textureNames[selectedItem];
    if (txt->fullName != NULL)
    {
        texNode->filename.setValue(txt->fullName);

        // check MODULATE vs BLEND
        if (txt->zsize == 3 || txt->zsize == 4)
            texNode->model = SoTexture2::MODULATE;
        else
            texNode->model = SoTexture2::BLEND;
    }
    else // no texture
        texNode->filename.setValue("");
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	updates the UI to reflect the current texture2 node
//
// Use: private
void
MyTextureEditor::updateTexture2UI()
//
////////////////////////////////////////////////////////////////////////
{
    // update color wheel + color slider
    float hsv[3];
    SbColor rgb = texNode->blendColor.getValue();
    rgb.getHSVValue(hsv);
    ignoreCallback = TRUE;
    colWheel->setBaseColor(hsv);
    colSlider->setBaseColor(hsv);
    ignoreCallback = FALSE;

    // update the option menu
    int ID = OPT_UNKNOWN;
    if (texNode->wrapS.getValue() == texNode->wrapT.getValue())
    {
        if (texNode->wrapS.getValue() == SoTexture2::REPEAT)
            ID = OPT_REPEAT;
        else if (texNode->wrapS.getValue() == SoTexture2::CLAMP)
            ID = OPT_CLAMP;
    }
    XtVaSetValues(widgetList[OPT_PULLDOWN], XmNmenuHistory, widgetList[ID], NULL);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	updates the UI to reflect the current texture2Transform node
//
// Use: private
void
MyTextureEditor::updateTextureXfUI()
//
////////////////////////////////////////////////////////////////////////
{
    updateTextureFieldAndSlider(SCALE_X_FIELD);
    updateTextureFieldAndSlider(SCALE_Y_FIELD);
    updateTextureFieldAndSlider(TRANS_X_FIELD);
    updateTextureFieldAndSlider(TRANS_Y_FIELD);
    updateTextureFieldAndSlider(ROT_FIELD);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	updates the UI to reflect the current texture function node
//
// Use: private
void
MyTextureEditor::updateTextureFuncUI()
//
////////////////////////////////////////////////////////////////////////
{
    // update the option menu
    int ID = MAP_UNKNOWN;
    if (texFuncNode->isOfType(SoTextureCoordinateDefault::getClassTypeId()))
        ID = MAP_DEFAULT;
    else if (texFuncNode->isOfType(SoTextureCoordinateEnvironment::getClassTypeId()))
        ID = MAP_ENV;
    else if (texFuncNode->isOfType(SoTextureCoordinatePlane::getClassTypeId()))
    {
        SbVec3f directionS = ((SoTextureCoordinatePlane *)texFuncNode)->directionS.getValue();
        SbVec3f directionT = ((SoTextureCoordinatePlane *)texFuncNode)->directionT.getValue();
        if (directionS == SbVec3f(1, 0, 0) && directionT == SbVec3f(0, 1, 0))
            ID = MAP_PLANE_XY;
        else if (directionS == SbVec3f(1, 0, 0) && directionT == SbVec3f(0, 0, 1))
            ID = MAP_PLANE_XZ;
        else if (directionS == SbVec3f(0, 1, 0) && directionT == SbVec3f(0, 0, 1))
            ID = MAP_PLANE_YZ;
    }
    XtVaSetValues(widgetList[MAPP_PULLDOWN], XmNmenuHistory, widgetList[ID], NULL);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets the repeate state and updates the UI. It returns TRUE if
//  set state has changed.
//
// Use: private
SbBool
MyTextureEditor::setRepeatState(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (repeatState == flag)
        return FALSE;
    repeatState = flag;

    //    XmToggleButtonSetState(widgetList[REPEAT_TOGGLE], repeatState, FALSE);

    // update the scale labels
    XmString xmstr1, xmstr2;
    if (repeatState)
    {
        xmstr1 = XmStringCreateSimple((char *)"Repeat X:");
        xmstr2 = XmStringCreateSimple((char *)"Repeat Y:");
    }
    else
    {
        xmstr1 = XmStringCreateSimple((char *)"Scale X:");
        xmstr2 = XmStringCreateSimple((char *)"Scale Y:");
    }
    XtVaSetValues(widgetList[SCALE_X_LABEL], XmNlabelString, xmstr1, NULL);
    XtVaSetValues(widgetList[SCALE_Y_LABEL], XmNlabelString, xmstr2, NULL);
    XmStringFree(xmstr1);
    XmStringFree(xmstr2);

    // update the scale text fields to new format
    updateTextureFieldAndSlider(SCALE_X_FIELD);
    updateTextureFieldAndSlider(SCALE_Y_FIELD);

    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	updates the given text field and slider to relfect current settings
//
// Use: private
void
MyTextureEditor::updateTextureFieldAndSlider(int fieldID)
//
////////////////////////////////////////////////////////////////////////
{
    float val;
    int intVal;
    char str[10];

    switch (fieldID)
    {
    case TRANS_X_FIELD:
        val = texXfNode->translation.getValue()[0];
        sprintf(str, "%.2f", val);
        intVal = (val > 1.0) ? 100 : ((val < -1) ? 0 : int((val + 1) * 50));
        XmScaleSetValue(widgetList[TRANS_X_SLD], intVal);
        break;

    case TRANS_Y_FIELD:
        val = texXfNode->translation.getValue()[1];
        sprintf(str, "%.2f", val);
        intVal = (val > 1.0) ? 100 : ((val < -1) ? 0 : int((val + 1) * 50));
        XmScaleSetValue(widgetList[TRANS_Y_SLD], intVal);
        break;

    case ROT_FIELD:
        // ??? prevent roundoff
        val = texXfNode->rotation.getValue() * 180.0 / M_PI;
        intVal = int(val);
        sprintf(str, "%d", intVal);
        intVal = int(100 * intVal / 360.0);
        XmScaleSetValue(widgetList[ROT_SLD], intVal);
        break;

    case SCALE_X_FIELD:
        val = texXfNode->scaleFactor.getValue()[0];
        // make it look like we are scaling the image, not the texture coord
        if (repeatState)
            sprintf(str, "%.1f", val);
        else
            sprintf(str, "%.2f", 1 / val);
        break;

    case SCALE_Y_FIELD:
        val = texXfNode->scaleFactor.getValue()[1];
        // make it look like we are scaling the image, not the texture coord
        if (repeatState)
            sprintf(str, "%.1f", val);
        else
            sprintf(str, "%.2f", 1 / val);
        break;
    }

    XmTextSetString(widgetList[fieldID], str);
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
MyTextureEditor::getPaletteNames()
//
////////////////////////////////////////////////////////////////////////
{
    struct dirent *direntry;
    DIR *dirp;
    char *f;
    char currentDir[MAXPATHLEN];

    // this is only called once at startup, so it doesn't
    // need to free the old paletteList.

    //
    // look for palettes under the default installed place
    //

    // see if SO_TEXTURE_DIR is set, if so use it...
    char *envDir = getenv("SO_TEXTURE_DIR");
    if (envDir != NULL)
    {
        delete[] paletteDir;
        paletteDir = STRDUP(envDir);
    }
    dirp = opendir(paletteDir);
    if (dirp)
    {
        if (getcwd(currentDir, sizeof(currentDir)) == NULL)
        {
            cerr << "MyTextureEditor::getPaletteNames: getcwd1 failed" << endl;
        }
        if (chdir(paletteDir) == -1)
        {
            cerr << "MyTextureEditor::getPaletteNames: chdir1 failed" << endl;
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
            cerr << "MyTextureEditor::getPaletteNames: chdir2 failed" << endl;
        }
    }

    //
    // Now look for palettes under the user's home directory, which
    // would overide the installed palettes.
    //

    char customDir[MAXPATHLEN];
    sprintf(customDir, "%s/%s", getenv("HOME"), customTextureDir);
    dirp = opendir(customDir);
    if (dirp)
    {
        if (getcwd(currentDir, sizeof(currentDir)) == NULL)
        {
            cerr << "MyTextureEditor::getPaletteNames: getcwd2 failed" << endl;
        }
        if (chdir(customDir) == -1)
        {
            cerr << "MyTextureEditor::getPaletteNames: chdir3 failed" << endl;
        }
        while ((direntry = readdir(dirp)))
        {
            f = direntry->d_name;
            // hide '.' files
            if (f[0] != '.' && !isDirectory(f))
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
            cerr << "MyTextureEditor::getPaletteNames: chdir4 failed" << endl;
        }
    }

    //
    // make sure we have at least one palette, else create an empty default
    // palette for things to work.
    //
    curPalette = 0;
    if (paletteList.getLength() == 0)
    {
#ifdef DEBUG
        SoDebugError::post("MyTextureEditor::getPaletteNames",
                           "cannot find palettes in directory %s or home directory.  Try setting the environment variable SO_TEXTURE_DIR to a directory which has texture files in it.", paletteDir);
#endif

        // create an empty default palette, withough loading anything
        // since we started with a blank palette.
        PaletteStruct *pal = new PaletteStruct;
        pal->name = STRDUP("default");
        pal->user = TRUE;
        pal->system = FALSE;
        paletteList.append(pal);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	saves the current palette in the user's HOME directory.
//
// Use: private
void
MyTextureEditor::savePalette()
//
////////////////////////////////////////////////////////////////////////
{
    char dirName[MAXPATHLEN];
    char fileName[MAXPATHLEN];
    struct stat buf;
    FILE *file;
    PaletteStruct *pal = (PaletteStruct *)paletteList[curPalette];

    // open the file for writting
    sprintf(dirName, "%s/%s/", getenv("HOME"), customTextureDir);
    if (stat(dirName, &buf) != 0)
        mkdir(dirName, 0x1ff);
    strcpy(fileName, dirName);
    strcat(fileName, pal->name);
    if ((file = fopen(fileName, "w")) == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MyTextureEditor::savePalette",
                           "couldn't create file: %s", fileName);
#endif
        return;
    }

    // write the names of the textures, on line at a time
    for (int i = 0; i < IMAGE_TOTAL; i++)
    {
        if (textureNames[i].fullName != NULL)
            fprintf(file, "%s\n", textureNames[i].fullName);
    }
    fclose(file);

    // finally update things
    if (!pal->user)
    {
        pal->user = TRUE;
        updateFileMenu();
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Given the current palette, loads all of the textures in the
//  current palette directory.
//
// Use: private
void
MyTextureEditor::loadPaletteItems()
//
////////////////////////////////////////////////////////////////////////
{
    if (curPalette == -1)
        return;

    //
    // set the hour glass cursor on....
    //
    Display *display = getDisplay();
    Widget shell = SoXt::getShellWidget(getWidget());
    Window window = XtWindow(shell);
    static Cursor cursor = 0;
    if (cursor == 0)
    {
        Drawable d = DefaultRootWindow(display);
        XColor foreground;
        foreground.red = 65535;
        foreground.green = foreground.blue = 0;

        Pixmap source = XCreateBitmapFromData(display, d,
                                              reinterpret_cast<char *>(hourglass_bits), hourglass_width, hourglass_height);
        cursor = XCreatePixmapCursor(display, source, source,
                                     &foreground, &foreground, hourglass_x_hot, hourglass_y_hot);
        XFreePixmap(display, source);
    }
    XDefineCursor(display, window, cursor);
    XSync(display, FALSE); // better than XFlush() (we get the glClear() to show up)

    // we are going to redraw each tile at a time once it is loaded
    // instead of waiting for all the textures to be read in
    // (feels a lot faster).
    glXMakeCurrent(getDisplay(), XtWindow(widgetList[TEXTURE_GLX]), paletteCtx);
    glClearColor(.6, .6, .6, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    char palDir[MAXPATHLEN];
    char currentDir[MAXPATHLEN];
    DIR *dirp;
    int i, num = 0;

    // go to the palette directory (user saved palettes override system
    // installed palettes).
    PaletteStruct *pal = (PaletteStruct *)paletteList[curPalette];
    if (pal->user)
        sprintf(palDir, "%s/%s", getenv("HOME"), customTextureDir);
    else
        sprintf(palDir, "%s/%s", paletteDir, pal->name);

    if ((dirp = opendir(palDir)) != NULL)
    {
        // cd to palette directory
        if (getcwd(currentDir, sizeof(currentDir)) == NULL)
        {
            cerr << "MyTextureEditor::loadPaletteItems: getcwd failed" << endl;
        }
        if (chdir(palDir) == -1)
        {
            cerr << "MyTextureEditor::loadPaletteItems: chdir1 failed" << endl;
        }

        //
        // get the names of the texture files
        //

        char fullName[MAXPATHLEN];
        if (pal->user)
        {
            // open the custom file and read the content of it (list of textures)
            FILE *file;
            if ((file = fopen(pal->name, "r")) != NULL)
            {

                // read each line entry (full texture path name) and add it to
                // the list of textures
                while (fscanf(file, " %s ", fullName) != EOF && num < (IMAGE_TOTAL - 1))
                {
                    if (addTextureEntry(num, fullName))
                    {
                        drawTextureTile(num, DRAW_NORM);
                        num++;
                    }
                }
                fclose(file);
            }
        }
        // system installed. Look for any image files under the current directory
        else
        {
            struct dirent *direntry;

            while ((direntry = readdir(dirp)) && num < (IMAGE_TOTAL - 1))
            {
                char *file = direntry->d_name;

                // ignore '.' files and directories
                if (file[0] == '.' || isDirectory(file))
                    continue;

                // add the file to the list of textures
                sprintf(fullName, "%s/%s", palDir, file);
                if (addTextureEntry(num, fullName))
                {
                    drawTextureTile(num, DRAW_NORM);
                    num++;
                }
            }
        }

        closedir(dirp);
        // back to our working directory
        if (chdir(currentDir) == -1)
        {
            cerr << "MyTextureEditor::loadPaletteItems: chdir2 failed" << endl;
        }
    }

    // clear the remaining texture names
    for (i = num; i < IMAGE_TOTAL; i++)
        deleteTextureEntry(i);

    // restore the cursor
    XUndefineCursor(display, window);

    loadedPalette = TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	reads the given image file (1-4 channels) in the given buffer of the
//  given size (the image file will be scaled to fit the given size). and
//  returns if succesfull.
//
//  Note: this ignores the alpha channel of the image, and scales the image
//	by skipping/duplicating pixels (VERY fast).
//
// Use: private
SbBool
MyTextureEditor::readScaledImage(
    char *file,
    int xsize, int ysize,
    char *buf,
    int &zsize)
//
////////////////////////////////////////////////////////////////////////
{
    // check to make sure file is a valid image file
    IMAGE *image;
    if (file == NULL || file[0] == '\0')
        return FALSE;
    if ((image = iopen(file, "r")) == NULL)
        return FALSE;
    if (image->colormap != CM_NORMAL)
    {
        iclose(image);
        return FALSE;
    }

    zsize = image->zsize;

    // allocate needed memory
    short *rbuf, *gbuf = NULL, *bbuf = NULL;
    rbuf = new short[image->xsize];
    if (image->zsize > 2)
    {
        gbuf = new short[image->xsize];
        bbuf = new short[image->xsize];
    }

    // read image in, one row at a time
    char *p = buf;
    for (int row = 0; row < ysize; row++)
    {
        // The row we'll read
        int rrow = (row * image->ysize) / ysize;

        if (zsize > 2)
        {
            getrow(image, rbuf, rrow, 0);
            getrow(image, gbuf, rrow, 1);
            getrow(image, bbuf, rrow, 2);
        }
        else
            getrow(image, rbuf, rrow, 0);

        // store these into an unsigned byte RGB format
        for (int i = 0; i < xsize; i++)
        {
            int ri = (i * image->xsize) / xsize;
            if (zsize > 2)
            {
                *p++ = rbuf[ri];
                *p++ = gbuf[ri];
                *p++ = bbuf[ri];
            }
            else
            {
                *p++ = rbuf[ri];
                *p++ = rbuf[-1];
                *p++ = rbuf[-1];
            }
        }
    }

    // delete image buffers
    delete[] rbuf;
    if (zsize > 2)
    {
        delete[] gbuf;
        delete[] bbuf;
    }

    iclose(image);
    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	reads the given image file in (1-4 channels) and return a char *
//  RGB format of the image (withought the alpha channel)
//
// Use: private
char *
MyTextureEditor::readImage(char *file, int &xsize, int &ysize, int &zsize)
//
////////////////////////////////////////////////////////////////////////
{
    // check to make sure file is a valid image file
    IMAGE *image;
    if (file == NULL || file[0] == '\0')
        return NULL;
    if ((image = iopen(file, "r")) == NULL)
        return NULL;
    if (image->colormap != CM_NORMAL)
    {
        iclose(image);
        return NULL;
    }

    xsize = image->xsize;
    ysize = image->ysize;
    zsize = image->zsize;

    // allocate needed memory
    short *rbuf, *gbuf = NULL, *bbuf = NULL;
    char *buf = new char[image->xsize * image->ysize * 3];
    rbuf = new short[image->xsize];
    if (image->zsize > 2)
    {
        gbuf = new short[image->xsize];
        bbuf = new short[image->xsize];
    }

    // read image in, one row at a time
    char *p = buf;
    for (int row = 0; row < image->ysize; row++)
    {

        if (image->zsize > 2)
        {
            getrow(image, rbuf, row, 0);
            getrow(image, gbuf, row, 1);
            getrow(image, bbuf, row, 2);
        }
        else
            getrow(image, rbuf, row, 0);

        // store these into an unsigned byte RGB format
        for (int i = 0; i < image->xsize; i++)
        {
            if (image->zsize > 2)
            {
                *p++ = rbuf[i];
                *p++ = gbuf[i];
                *p++ = bbuf[i];
            }
            else
            {
                *p++ = rbuf[i];
                *p++ = rbuf[i];
                *p++ = rbuf[i];
            }
        }
    }

    // delete image buffers
    delete[] rbuf;
    if (zsize > 2)
    {
        delete[] gbuf;
        delete[] bbuf;
    }

    iclose(image);
    return buf;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	switches to the given palette number
//
// Use: private
void
MyTextureEditor::switchPalette(int id)
//
////////////////////////////////////////////////////////////////////////
{
    curPalette = id;

    // get new textures, update material and selection
    loadPaletteItems();
    deselectCurrentItem(FALSE); // don't redraw palette changes there
    redrawPalette();
    updateWindowTitle();
    updateFileMenu();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	deselect the currently selected item in the palette.
//
// Use: private
void
MyTextureEditor::deselectCurrentItem(SbBool drawHighlight)
//
////////////////////////////////////////////////////////////////////////
{
    if (selectedItem == -1)
        return;

    // deselect the current item and update things that depend on it
    if (drawHighlight)
    {
        glXMakeCurrent(getDisplay(), XtWindow(widgetList[TEXTURE_GLX]), paletteCtx);
        if (selectedItem == currentItem)
            drawTextureTile(selectedItem, DRAW_CURRENT);
        else
        {
            drawTextureTile(selectedItem, DRAW_NORM);
            if (currentItem != -1)
                drawTextureTile(currentItem, DRAW_CURRENT);
        }
    }
    selectedItem = -1;
    updateTextureName();

    // hide color wheel
    colSlider->hide();
    colWheel->hide();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	deletes the names + images of the given texture.
//
//  NOTE: we don't ever delete the iconImage buffer since those
//  will always be IMAGE_SIZE X IMAGE_SIZE
//
// Use: private
void
MyTextureEditor::deleteTextureEntry(int id)
//
////////////////////////////////////////////////////////////////////////
{
    if (textureNames[id].name != NULL)
        delete[] textureNames[id].name;
    if (textureNames[id].fullName != NULL)
        delete[] textureNames[id].fullName;

    textureNames[id].name = textureNames[id].fullName = NULL;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	adds a texture entry (name + fullName + load small size image)
//  at the given index. This returns TRUE if this routine was successfull
//  (valid image file).
//
// Use: private
SbBool
MyTextureEditor::addTextureEntry(int id, char *fullName)
//
////////////////////////////////////////////////////////////////////////
{
    // read the image in iconized form
    if (!readScaledImage(fullName, IMAGE_SIZE, IMAGE_SIZE,
                         textureNames[id].iconImage, textureNames[id].zsize))
        return FALSE;

    // extract the file name (text after the last '/' char)
    deleteTextureEntry(id);
    char *p = strrchr(fullName, '/');
    p = (p != NULL) ? p + 1 : fullName;
    textureNames[id].name = STRDUP(p);
    textureNames[id].fullName = STRDUP(fullName);

    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets the window title based on the current palette and save status
//
// Use: private
void
MyTextureEditor::updateWindowTitle()
//
////////////////////////////////////////////////////////////////////////
{
    char str[150];
    sprintf(str, "%s: %s", editorTitle, ((PaletteStruct *)paletteList[curPalette])->name);
    setTitle(str);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Called whenever an event occurs in the texture palette window.
//
// Use: private
void
MyTextureEditor::handleEvent(XAnyEvent *xe)
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

        //
        // check for double click. If the time difference between
        // successive mouse down is less than 3/10th sec, then
        // consider it a double click.
        // ??? do we need to worry about time warping ?
        //
        if ((be->time - prevTime) < DOUBLE_CLICK_TIME)
        {

            // open the image dialog
            openImageDialog();

            // if we don't have any images there, automatically open
            // the file browser
            // ??? make it look like the "Open..." button was pressed
            //		if (textureNames[selectedItem].fullName == NULL)
            //		    imageDialogOpenCB(NULL, this, NULL);
        }
        else // single click (or first click of the double click)
        {
            // select the current texture (if it is not currently selected)
            if (currentItem != selectedItem)
            {

                // update the feedback
                glXMakeCurrent(getDisplay(), XtWindow(widgetList[TEXTURE_GLX]), paletteCtx);
                drawTextureTile(selectedItem, DRAW_NORM);
                drawTextureTile(currentItem, DRAW_SELECTED);
                selectedItem = currentItem;

                updateTextureNode();

                // ??? update the image dialog (if visible)
                if (widgetList[DIALOG_WINDOW] != NULL)
                    setNewDialogImage(textureNames[selectedItem].fullName);

                // show/hide the color wheel
                if (textureNames[selectedItem].zsize < 3 && textureNames[selectedItem].fullName != NULL)
                {
                    colWheel->show();
                    colSlider->show();
                }
                else
                {
                    colSlider->hide();
                    colWheel->hide();
                }
            }
        }

        prevTime = be->time;
    }
    break;

    case MotionNotify:
    {
        // do a region picking to figure which tile we are above
        XMotionEvent *me = (XMotionEvent *)xe;
        int xpos = int(floorf(IMAGE_NUM * me->x / float(GLX_SIZE)));
        int ypos = int(floorf(IMAGE_NUM * me->y / float(GLX_SIZE)));
        int whichTexture = xpos + IMAGE_NUM * ypos;
        if (whichTexture != currentItem)
        {
            glXMakeCurrent(getDisplay(), XtWindow(widgetList[TEXTURE_GLX]), paletteCtx);
            if (currentItem != selectedItem && currentItem != -1)
                drawTextureTile(currentItem, DRAW_NORM);
            if (whichTexture != selectedItem)
                drawTextureTile(whichTexture, DRAW_CURRENT);
            drawTextureTile(selectedItem, DRAW_SELECTED);
            currentItem = whichTexture;
            updateTextureName();
        }
    }
    break;

    case LeaveNotify:
        if (currentItem != selectedItem)
        {
            glXMakeCurrent(getDisplay(), XtWindow(widgetList[TEXTURE_GLX]), paletteCtx);
            drawTextureTile(currentItem, DRAW_NORM);
            drawTextureTile(selectedItem, DRAW_SELECTED);
            currentItem = -1;
            updateTextureName();
        }
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	creates a dialog which lets the user type a name.
//
// Use: private
void
MyTextureEditor::createNewDialog()
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[5];
    XmString xmstr = XmStringCreateSimple((char *)"New Palette Name:");

    int n = 0;
    XtSetArg(args[n], XmNautoUnmanage, FALSE);
    n++;
    XtSetArg(args[n], XtNtitle, "New Palette Dialog");
    n++;
    XtSetArg(args[n], XmNselectionLabelString, xmstr);
    n++;
    Widget dialog = XmCreatePromptDialog(SoXt::getShellWidget(getWidget()),
                                         (char *)"saveDialog", args, n);
    XmStringFree(xmstr);

    XtUnmanageChild(XmSelectionBoxGetChild(dialog, XmDIALOG_HELP_BUTTON));
    XtUnmanageChild(XmSelectionBoxGetChild(dialog, XmDIALOG_SEPARATOR));

    // register callback to destroy (and not just unmap) the dialog
    // and retreive the text (ok push button case).
    XtAddCallback(dialog, XmNokCallback,
                  (XtCallbackProc)MyTextureEditor::newDialogCB, (XtPointer) this);
    XtAddCallback(dialog, XmNcancelCallback,
                  (XtCallbackProc)MyTextureEditor::newDialogCB, (XtPointer) this);

    XtManageChild(dialog);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	create a dialog which lets the user change it's mind about deleting
//  the current palette.
//
// Use: private
void
MyTextureEditor::createDeleteDialog(const char *title, const char *str1, const char *str2)
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
    Widget dialog = XmCreateWarningDialog(SoXt::getShellWidget(getWidget()),
                                          (char *)"DeleteDialog", args, n);
    XmStringFree(xmstr);

    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_HELP_BUTTON));
    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_SEPARATOR));

    // register callback to destroy (and not just unmap) the dialog
    XtAddCallback(dialog, XmNokCallback,
                  (XtCallbackProc)MyTextureEditor::deleteDialogCB, (XtPointer) this);
    XtAddCallback(dialog, XmNcancelCallback,
                  (XtCallbackProc)MyTextureEditor::deleteDialogCB, (XtPointer) this);

    XtManageChild(dialog);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	updates the "File" menu (grey things out) to reflect current
//  palette.
//
// Use: private
void
MyTextureEditor::updateFileMenu()
//
////////////////////////////////////////////////////////////////////////
{
    PaletteStruct *pal = (PaletteStruct *)paletteList[curPalette];

    XtVaSetValues(widgetList[FILE_RESET], XmNsensitive,
                  pal->user && pal->system, NULL);
    XtVaSetValues(widgetList[FILE_DELETE], XmNsensitive,
                  pal->user && !pal->system, NULL);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates a new palette with the given name (called when the
//  user wants a new palette using the 'new' button).
//
// Use: private
void
MyTextureEditor::createNewPalette(char *palName)
//
////////////////////////////////////////////////////////////////////////
{
    //
    // create that empty palette file (under the user's home
    // directory).
    //
    char dirName[MAXPATHLEN];
    char fileName[MAXPATHLEN];
    struct stat buf;
    FILE *file;
    sprintf(dirName, "%s/%s/", getenv("HOME"), customTextureDir);
    if (stat(dirName, &buf) != 0)
        mkdir(dirName, 0x1ff);
    strcpy(fileName, dirName);
    strcat(fileName, palName);
    if ((file = fopen(fileName, "w")) == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MyTextureEditor::createNewPalette",
                           "couldn't create file: %s", fileName);
#endif
        return;
    }
    fclose(file);

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

    // now switch to new palette
    switchPalette(id);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	updates the image dialog based on the currently selected texture.
//
// Use: private
void
MyTextureEditor::redrawImageDialog()
//
////////////////////////////////////////////////////////////////////////
{
    // make sure window is on the screen
    Widget glx = widgetList[DIALOG_IMAGE];
    if (glx == NULL)
        return;
    Window window = XtWindow(glx);
    if (window == 0)
        return;

    glXMakeCurrent(XtDisplay(glx), window, imageDialogCtx);

    // reset projection
    short w, h;
    XtVaGetValues(glx, XmNwidth, &w, XmNheight, &h, NULL);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);

    // draw image
    if (dialogImage != NULL)
    {
        glRasterPos2i(0, 0);
        glDrawPixels(dialogImageSize[0], dialogImageSize[1], GL_RGB,
                     GL_UNSIGNED_BYTE, dialogImage);
    }
    else
    {
        glClearColor(.6, .6, .6, 0);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	sets which image should be displayed (NULL for clear) in the
//  image dialog.
//
// Use: private
void
MyTextureEditor::setNewDialogImage(char *fileName)
//
////////////////////////////////////////////////////////////////////////
{
    // make sure widget is built
    if (widgetList[DIALOG_WINDOW] == NULL)
        return;

    // delete the existing dialog image and names
    delete dialogImage;
    if (dialogImageName != NULL)
        delete[] dialogImageName;
    if (dialogImageInfo != NULL)
        delete[] dialogImageInfo;
    dialogImage = NULL;
    dialogImageName = NULL;
    dialogImageInfo = NULL;

    if (fileName != NULL)
    {

        // read the full image in and update the picture
        dialogImage = readImage(fileName,
                                dialogImageSize[0], dialogImageSize[1], dialogImageSize[2]);
        if (dialogImage != NULL)
        {

            // resize the glx window
            XtVaSetValues(widgetList[DIALOG_IMAGE], XmNwidth, dialogImageSize[0],
                          XmNheight, dialogImageSize[1], NULL);

            // save the texture name
            dialogImageName = STRDUP(fileName);

            // format the image info
            char str[100];
            sprintf(str, "%d x %d    %d component", dialogImageSize[0],
                    dialogImageSize[1], dialogImageSize[2]);
            if (dialogImageSize[2] > 1)
                strcat(str, "s");
            dialogImageInfo = STRDUP(str);
        }
        else
        {

            // bogus file name was given. print an error dialog
            char str[MAXPATHLEN + 100];
            sprintf(str, "Error opening image file: %s", fileName);
            SoXt::createSimpleErrorDialog(widgetList[DIALOG_WINDOW], (char *)"File Error", str);
        }
    }

    if (dialogImageName == NULL)
        // no images so make the glx tiny size (non zero)
        XtVaSetValues(widgetList[DIALOG_IMAGE], XmNwidth, 1, XmNheight, 1, NULL);

    // update the name label
    XmString xmstr = (dialogImageName != NULL) ? XmStringCreateSimple(dialogImageName) : XmStringCreateSimple((char *)noFileNameStr);
    XtVaSetValues(widgetList[DIALOG_NAME], XmNlabelString, xmstr, NULL);
    XmStringFree(xmstr);

    // update the info label
    xmstr = (dialogImageInfo != NULL) ? XmStringCreateSimple(dialogImageInfo) : XmStringCreateSimple((char *)"");
    XtVaSetValues(widgetList[DIALOG_INFO], XmNlabelString, xmstr, NULL);
    XmStringFree(xmstr);

    //
    // center everything in the window
    //

    // get the max size of all the pieces
    short width[4];
    int maxWidth;
    width[0] = buttonsTotalWidth; // constant size
    XtVaGetValues(widgetList[DIALOG_INFO], XmNwidth, &width[1], NULL);
    XtVaGetValues(widgetList[DIALOG_NAME], XmNwidth, &width[2], NULL);
    XtVaGetValues(widgetList[DIALOG_IMAGE], XmNwidth, &width[3], NULL);
    maxWidth = width[0];
    for (int i = 1; i < 4; i++)
        if (width[i] > maxWidth)
            maxWidth = width[i];

    // now center things
    XtVaSetValues(widgetList[DIALOG_BUTTON_0], XmNleftOffset,
                  (maxWidth - width[0]) / 2, NULL);
    XtVaSetValues(widgetList[DIALOG_INFO], XmNleftOffset,
                  (maxWidth - width[1]) / 2, NULL);
    XtVaSetValues(widgetList[DIALOG_NAME], XmNleftOffset,
                  (maxWidth - width[2]) / 2, NULL);
    XtVaSetValues(widgetList[DIALOG_IMAGE], XmNleftOffset,
                  (maxWidth - width[3]) / 2, NULL);

    // ??? redraw the image now (instead of waiting for an expose event)
    // to minimize the weird display when resizing a window.
    redrawImageDialog();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Builds the image dialog window
//
// Use: private
void
MyTextureEditor::openImageDialog()
//
////////////////////////////////////////////////////////////////////////
{
    // check to see if dialog has already been built
    Widget shell = widgetList[DIALOG_WINDOW];
    if (shell != NULL)
    {
        SoXt::show(shell);
        return;
    }

    int i, n;
    Arg args[12];
    Widget buttons[4], labels[2], glx;

    //
    // create the topLevel Shell window
    //
    n = 0;
    XtSetArg(args[n], XtNtitle, "Texture Image Dialog");
    n++;
    XtSetArg(args[n], XmNiconName, "Image Dialog");
    n++;
    XtSetArg(args[n], XmNallowShellResize, TRUE);
    n++;
    shell = XtCreatePopupShell("SoXtImageDialog", topLevelShellWidgetClass,
                               SoXt::getShellWidget(getWidget()), args, n);

    widgetList[DIALOG_WINDOW] = shell;
    XtAddCallback(shell, XmNdestroyCallback,
                  (XtCallbackProc)MyTextureEditor::imageDialogDestroyCB,
                  (XtPointer) this);

    // create a top level form to hold everything together
    n = 0;
    XtSetArg(args[n], XmNmarginHeight, 10);
    n++;
    XtSetArg(args[n], XmNmarginWidth, 10);
    n++;
    Widget dummyForm = XmCreateForm(shell, (char *)"dummyForm", args, n);
    Widget form = XmCreateForm(dummyForm, (char *)"imageDialogForm", NULL, 0);

    //
    // create all the parts
    //

    // create the push buttons
    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    buttons[0] = XmCreatePushButtonGadget(form, (char *)"Open...", args, n);
    buttons[1] = XmCreatePushButtonGadget(form, (char *)"Clear", args, n);
    buttons[2] = XmCreatePushButtonGadget(form, (char *)"Apply", args, n);
    buttons[3] = XmCreatePushButtonGadget(form, (char *)"CloseWin", args, n);
    widgetList[DIALOG_BUTTON_0] = buttons[0];

    // make them all the same size
    short w;
    int width = 0;
    for (i = 0; i < 4; i++)
    {
        XtVaGetValues(buttons[i], XtNwidth, &w, NULL);
        if (w > width)
            width = w;
    }
    for (i = 0; i < 4; i++)
        XtVaSetValues(buttons[i], XtNwidth, width, NULL);
    buttonsTotalWidth = 4 * width + 3 * 5; // see layout spacing

    XtAddCallback(buttons[0], XmNactivateCallback,
                  (XtCallbackProc)MyTextureEditor::imageDialogOpenCB, (XtPointer) this);
    XtAddCallback(buttons[1], XmNactivateCallback,
                  (XtCallbackProc)MyTextureEditor::imageDialogClearCB, (XtPointer) this);
    XtAddCallback(buttons[2], XmNactivateCallback,
                  (XtCallbackProc)MyTextureEditor::imageDialogApplyCB, (XtPointer) this);
    XtAddCallback(buttons[3], XmNactivateCallback,
                  (XtCallbackProc)MyTextureEditor::imageDialogCloseCB, (XtPointer) this);

    // create the image name label
    widgetList[DIALOG_INFO] = labels[0] = XmCreateLabelGadget(form, (char *)"imageInfo", NULL, 0);
    widgetList[DIALOG_NAME] = labels[1] = XmCreateLabelGadget(form, (char *)"imageName", NULL, 0);

    // create the image glx window
    n = 0;
    XtSetArg(args[n], GLwNrgba, TRUE);
    n++;
    XtSetArg(args[n], GLwNredSize, 1);
    n++;
    XtSetArg(args[n], GLwNgreenSize, 1);
    n++;
    XtSetArg(args[n], GLwNblueSize, 1);
    n++;
    glx = XtCreateWidget("imageDialogGLX", glwMDrawingAreaWidgetClass, form, args, n);
    widgetList[DIALOG_IMAGE] = glx;

    XtUninstallTranslations(glx);
    XtAddCallback(glx, GLwNginitCallback,
                  (XtCallbackProc)MyTextureEditor::imageDialogInitCB, (XtPointer) this);
    XtAddCallback(glx, GLwNexposeCallback,
                  (XtCallbackProc)MyTextureEditor::imageDialogExposeCB, (XtPointer) this);

    //
    // layout !
    //
    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;

    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    XtSetValues(buttons[0], args, n + 1);
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    XtSetArg(args[n + 1], XmNleftOffset, 5);
    for (i = 1; i < 4; i++)
    {
        XtSetArg(args[n + 2], XmNleftWidget, buttons[i - 1]);
        XtSetValues(buttons[i], args, n + 3);
    }

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;

    XtSetArg(args[n], XmNbottomWidget, buttons[0]);
    XtSetArg(args[n + 1], XmNbottomOffset, 10);
    XtSetValues(labels[0], args, n + 2);
    XtSetArg(args[n], XmNbottomWidget, labels[0]);
    XtSetArg(args[n + 1], XmNbottomOffset, 7);
    XtSetValues(labels[1], args, n + 2);

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, labels[1]);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 10);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(glx, args, n);

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(form, args, n);

    setNewDialogImage(textureNames[selectedItem].fullName);

    // manage children
    XtManageChildren(buttons, 4);
    XtManageChildren(labels, 2);
    XtManageChild(glx);
    XtManageChild(form);
    XtManageChild(dummyForm);
    SoXt::show(shell);
}

//
// redefine those generic virtual functions
//
const char *
MyTextureEditor::getDefaultWidgetName() const
{
    return "MyTextureEditor";
}

const char *
MyTextureEditor::getDefaultTitle() const
{
    return "Texture Editor";
}

const char *
MyTextureEditor::getDefaultIconTitle() const
{
    return "Texture Editor";
}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

void
MyTextureEditor::fieldChangedCB(Widget, MyTextureEditor *p, void *)
{
    p->fieldChanged = TRUE;
}

void
MyTextureEditor::glxExposeCB(Widget, MyTextureEditor *p, void *)
{
    if (p->loadedPalette)
        p->redrawPalette();
    else
        p->loadPaletteItems(); // this will also redraw the tiles
}

void
MyTextureEditor::glxInitCB(Widget glx, MyTextureEditor *p, void *)
{
    // create a GLX context
    XVisualInfo *vis;
    XtVaGetValues(glx, GLwNvisualInfo, &vis, NULL);
    p->paletteCtx = glXCreateContext(XtDisplay(glx), vis, NULL, GL_TRUE);

    glXMakeCurrent(XtDisplay(glx), XtWindow(glx), p->paletteCtx);

    // set the projection
    short w, h;
    XtVaGetValues(glx, XmNwidth, &w, XmNheight, &h, NULL);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

void
MyTextureEditor::glxEventCB(Widget, MyTextureEditor *p, XAnyEvent *xe, Boolean *)
{
    p->handleEvent(xe);
}

void
MyTextureEditor::imageDialogExposeCB(Widget, MyTextureEditor *p, void *)
{
    p->redrawImageDialog();
}

void
MyTextureEditor::imageDialogInitCB(Widget glx, MyTextureEditor *p, void *)
{
    // create a GLX context
    XVisualInfo *vis;
    XtVaGetValues(glx, GLwNvisualInfo, &vis, NULL);
    p->imageDialogCtx = glXCreateContext(XtDisplay(glx), vis, NULL, GL_TRUE);

    glXMakeCurrent(XtDisplay(glx), XtWindow(glx), p->imageDialogCtx);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

void
MyTextureEditor::imageDialogClearCB(Widget, MyTextureEditor *p, void *)
{
    p->setNewDialogImage(NULL);
}

void
MyTextureEditor::imageDialogCloseCB(Widget, MyTextureEditor *p, void *)
{
    SoXt::hide(p->widgetList[DIALOG_WINDOW]);
}

void
MyTextureEditor::acceptCB(Widget, MyTextureEditor *p, void *)
{
    p->callbackList.invokeCallbacks(p);
}

//
// called by the color wheel when the color changes
//
void
MyTextureEditor::colWheelCB(void *pt, const float hsv[3])
{
    MyTextureEditor *p = (MyTextureEditor *)pt;
    if (p->ignoreCallback)
        return;

    p->ignoreCallback = TRUE;
    p->colSlider->setBaseColor(hsv);
    p->ignoreCallback = FALSE;

    // update the texture
    SbColor rgb;
    rgb.setHSVValue(hsv);
    p->texNode->blendColor = rgb;
}

//
// called by color slider when the color changes
//
void
MyTextureEditor::colSliderCB(void *pt, float)
{
    MyTextureEditor *p = (MyTextureEditor *)pt;
    if (p->ignoreCallback)
        return;

    const float *hsv = p->colSlider->getBaseColor();
    p->ignoreCallback = TRUE;
    p->colWheel->setBaseColor(hsv);
    p->ignoreCallback = FALSE;

    // update the texture
    SbColor rgb;
    rgb.setHSVValue(hsv);
    p->texNode->blendColor = rgb;
}

//
// called whenever the scale X thumbwheel changes
//
void
MyTextureEditor::scaleXThumbCB(void *pt, float val)
{
    MyTextureEditor *p = (MyTextureEditor *)pt;

    SbVec2f scale = p->texXfNode->scaleFactor.getValue();

    // shorter/grow the scale value
    if (p->repeatState)
        scale[0] *= pow(1.5, (double)val - p->oldXThumbVal);
    else
        scale[0] /= pow(1.5, (double)val - p->oldXThumbVal);
    p->texXfNode->scaleFactor.setValue(scale);
    p->oldXThumbVal = val;

    p->updateTextureFieldAndSlider(SCALE_X_FIELD);
}

//
// called whenever the scale Y thumbwheel changes
//
void
MyTextureEditor::scaleYThumbCB(void *pt, float val)
{
    MyTextureEditor *p = (MyTextureEditor *)pt;

    SbVec2f scale = p->texXfNode->scaleFactor.getValue();

    // shorter/grow the scale value
    if (p->repeatState)
        scale[1] *= pow(1.5, (double)val - p->oldYThumbVal);
    else
        scale[1] /= pow(1.5, (double)val - p->oldYThumbVal);
    p->texXfNode->scaleFactor.setValue(scale);
    p->oldYThumbVal = val;

    p->updateTextureFieldAndSlider(SCALE_Y_FIELD);
}

//
// called whenever the slider's text fields changes
//
void
MyTextureEditor::fieldsCB(Widget w, int id, void *)
{
    // get the class pointer
    MyTextureEditor *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    if (!p->fieldChanged)
        return;
    p->fieldChanged = FALSE;

    // get the text field value and update the texture node
    char *str = XmTextGetString(w);
    float val;
    SbVec2f trans, scale, cent;
    if (sscanf(str, "%f", &val))
    {
        switch (id)
        {
        case SCALE_X_FIELD:
            // make it look like we are scaling the image, not the texture coord
            scale = p->texXfNode->scaleFactor.getValue();
            if (p->repeatState)
                scale[0] = val;
            else
                scale[0] = 1 / val;
            p->texXfNode->scaleFactor.setValue(scale);
            break;

        case SCALE_Y_FIELD:
            // make it look like we are scaling the image, not the texture coord
            scale = p->texXfNode->scaleFactor.getValue();
            if (p->repeatState)
                scale[1] = val;
            else
                scale[1] = 1 / val;
            p->texXfNode->scaleFactor.setValue(scale);
            break;

        case TRANS_X_FIELD:
            trans = p->texXfNode->translation.getValue();
            trans[0] = val;
            p->texXfNode->translation.setValue(trans);

            // update to rotate around center of image
            cent = p->texXfNode->center.getValue();
            cent[0] = 0.5 - trans[0];
            p->texXfNode->center.setValue(cent);
            break;

        case TRANS_Y_FIELD:
            trans = p->texXfNode->translation.getValue();
            trans[1] = val;
            p->texXfNode->translation.setValue(trans);

            // update to rotate around center of image
            cent = p->texXfNode->center.getValue();
            cent[1] = 0.5 - trans[1];
            p->texXfNode->center.setValue(cent);
            break;

        case ROT_FIELD:
            while (val < 0)
                val += 360;
            while (val > 360)
                val -= 360;
            p->texXfNode->rotation = val * M_PI / 180.0;
            break;
        }
    }
    XtFree(str);

    // reformat the text field and update the slider
    p->updateTextureFieldAndSlider(id);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(p->getWidget()), XmTRAVERSE_CURRENT);
}

//
// called whenever the motif sliders changes
//
void
MyTextureEditor::slidersCB(Widget w, int id, void *)
{
    // get the class pointer
    MyTextureEditor *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    // get the slider value
    int intVal;
    XmScaleGetValue(w, &intVal);
    float val = intVal / 100.0;

    char str[10];
    SbVec2f trans, cent;

    switch (id)
    {
    case TRANS_X_SLD:
        val = 1 - 2 * val; // make inverted [-1,1]
        trans = p->texXfNode->translation.getValue();
        trans[0] = val;
        p->texXfNode->translation.setValue(trans);

        // update to rotate around center of image
        cent = p->texXfNode->center.getValue();
        cent[0] = 0.5 - trans[0];
        p->texXfNode->center.setValue(cent);

        sprintf(str, "%.2f", val);
        XmTextSetString(p->widgetList[TRANS_X_FIELD], str);
        break;

    case TRANS_Y_SLD:
        val = 1 - 2 * val; // make inverted [-1,1]
        trans = p->texXfNode->translation.getValue();
        trans[1] = val;
        p->texXfNode->translation.setValue(trans);

        // update to rotate around center of image
        cent = p->texXfNode->center.getValue();
        cent[1] = 0.5 - trans[1];
        p->texXfNode->center.setValue(cent);

        sprintf(str, "%.2f", val);
        XmTextSetString(p->widgetList[TRANS_Y_FIELD], str);
        break;

    case ROT_SLD:
        p->texXfNode->rotation = val * 2 * M_PI;
        intVal = int(val * 360);
        sprintf(str, "%d", intVal);
        XmTextSetString(p->widgetList[ROT_FIELD], str);
        break;
    }
}

//
// called whenever an entry within the mapping menu gets selected
//
void
MyTextureEditor::mappingMenuCB(Widget w, int id, void *)
{
    // get the class pointer
    MyTextureEditor *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    SoTextureCoordinateFunction *newFunc;

    switch (id)
    {
    case MAP_DEFAULT:
        newFunc = new SoTextureCoordinateDefault;
        break;

    case MAP_ENV:
        newFunc = new SoTextureCoordinateEnvironment;
        break;

    case MAP_PLANE_XY:
        newFunc = new SoTextureCoordinatePlane;
        ((SoTextureCoordinatePlane *)newFunc)->directionS = SbVec3f(1, 0, 0);
        ((SoTextureCoordinatePlane *)newFunc)->directionT = SbVec3f(0, 1, 0);
        break;

    case MAP_PLANE_XZ:
        newFunc = new SoTextureCoordinatePlane;
        ((SoTextureCoordinatePlane *)newFunc)->directionS = SbVec3f(1, 0, 0);
        ((SoTextureCoordinatePlane *)newFunc)->directionT = SbVec3f(0, 0, 1);
        break;

    case MAP_PLANE_YZ:
        newFunc = new SoTextureCoordinatePlane;
        ((SoTextureCoordinatePlane *)newFunc)->directionS = SbVec3f(0, 1, 0);
        ((SoTextureCoordinatePlane *)newFunc)->directionT = SbVec3f(0, 0, 1);
        break;

    case MAP_UNKNOWN:
// this should never happen !
#ifdef DEBUG
        SoDebugError::post("MyTextureEditor::mappingMenuCB",
                           "MAP_UNKNOWN selected!");
#endif
        return;
    default:
        newFunc = NULL;
        fprintf(stderr, "MyTextureEditor::mappingMenuCB(): newFunc uninitialized\n");
        break;
    }

    if (newFunc)
    {
        // replace the old func node
        p->sceneRoot->replaceChild(p->texFuncNode, newFunc);
        p->texFuncNode = newFunc;
    }
}

//
// called whenever an entry within the option menu gets selected
//
void
MyTextureEditor::optionMenuCB(Widget w, int id, void *)
{
    // get the class pointer
    MyTextureEditor *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    switch (id)
    {
    case OPT_REPEAT:
        p->texNode->wrapS = SoTexture2::REPEAT;
        p->texNode->wrapT = SoTexture2::REPEAT;
        p->setRepeatState(TRUE);
        break;

    case OPT_CLAMP:
        p->texNode->wrapS = SoTexture2::CLAMP;
        p->texNode->wrapT = SoTexture2::CLAMP;
        p->setRepeatState(FALSE);
        break;

    case OPT_UNKNOWN:
// this should never happen !
#ifdef DEBUG
        SoDebugError::post("MyTextureEditor::optionMenuCB",
                           "OPT_UNKNOWN selected!");
#endif
        break;
    }
}

//
// called when an entry within the "File" menu is selected
//
void
MyTextureEditor::fileMenuCB(Widget w, int id, void *)
{
    // get the class pointer
    MyTextureEditor *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    switch (id)
    {
    case FILE_NEW:
        p->createNewDialog();
        break;

    case FILE_RESET:
        p->createDeleteDialog("Reset Palette Dialog",
                              "Reset to default palette ?", "(all changes will be lost)");
        break;

    case FILE_DELETE:
        p->createDeleteDialog("Delete Palette Dialog",
                              "Delete current palette ?", "(list of texture files will be lost)");
        break;
    }
}

//
//  Called whenever a new item menu is selected from the palette
//  popup menu.
//
void
MyTextureEditor::paletteMenuCB(Widget w, int num, void *)
{
    // get the class pointer
    MyTextureEditor *p;
    XtVaGetValues(w, XmNuserData, &p, NULL);

    // return if the same palette is choosen
    if (p->curPalette == num)
        return;

    p->switchPalette(num);
}

//
// called when the delete dialog "ok"/"Cancel" buttons gets pressed. This
// will delete the palette in the user's home directory.
//
void
MyTextureEditor::deleteDialogCB(Widget dialog, MyTextureEditor *p,
                                XmAnyCallbackStruct *cb)
{
    // remove the user's palette
    if (cb->reason == XmCR_OK)
    {

        PaletteStruct *pal = (PaletteStruct *)p->paletteList[p->curPalette];

        // remove the palette in user's home
        char palDir[MAXPATHLEN];
        sprintf(palDir, "%s/%s/%s", getenv("HOME"), customTextureDir, pal->name);
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
        p->switchPalette(p->curPalette);
    }

    XtDestroyWidget(dialog);
}

//
// called whenever the "ok"/"Cancel" buttons within the "New Palette" dialog
// gets pressed.
//
void
MyTextureEditor::newDialogCB(Widget dialog, MyTextureEditor *p,
                             XmAnyCallbackStruct *cb)
{
    if (cb->reason == XmCR_OK)
    {

        // retreive text and create palette
        Widget field = XmSelectionBoxGetChild(dialog, XmDIALOG_TEXT);
        char *str = XmTextGetString(field);
        if (str[0] != '\0')
            p->createNewPalette(str);
        XtFree(str);
    }

    XtDestroyWidget(dialog);
}

//
// called when image dialog gets destroyed (reset widget pointers).
//
void
MyTextureEditor::imageDialogDestroyCB(Widget, MyTextureEditor *p, void *)
{
    //printf("shell destroyed\n");
    // reset widget pointers
    p->widgetList[DIALOG_WINDOW] = NULL;
    p->widgetList[DIALOG_IMAGE] = NULL;
    p->widgetList[DIALOG_NAME] = NULL;
    p->widgetList[DIALOG_FILE_BROWSER] = NULL;

    // free image data and name
    delete p->dialogImage;
    if (p->dialogImageName != NULL)
        delete[] p -> dialogImageName;
    if (p->dialogImageInfo != NULL)
        delete[] p -> dialogImageInfo;
    p->dialogImage = NULL;
    p->dialogImageName = NULL;
    p->dialogImageInfo = NULL;

    // free the glx contex
    if (p->imageDialogCtx)
        glXDestroyContext(p->getDisplay(), p->imageDialogCtx);
    p->imageDialogCtx = 0;
}

//
// called when image dialog "Open..." button gets pressed
//
void
MyTextureEditor::imageDialogOpenCB(Widget, MyTextureEditor *p, void *)
{
    if (p->widgetList[DIALOG_FILE_BROWSER] == NULL)
    {

        Arg args[5];
        int n = 0;

        // ??? need to find a way to save the motif file browser current directory
        // ??? for next time around (we really shouldn't delete this guy, but its
        // ??? parent gets destroyed when closed).
        // unmanage when ok/cancel are pressed
        XtSetArg(args[n], XmNautoUnmanage, TRUE);
        n++;
        XtSetArg(args[n], XmNtitle, "Image File Browser");
        n++;
        Widget fileDialog = XmCreateFileSelectionDialog(
            p->widgetList[DIALOG_WINDOW], (char *)"fileBrowser", args, n);

        p->widgetList[DIALOG_FILE_BROWSER] = fileDialog;
        XtAddCallback(fileDialog, XmNokCallback,
                      (XtCallbackProc)MyTextureEditor::fileDialogOkCB,
                      (XtPointer)p);
    }

    XtManageChild(p->widgetList[DIALOG_FILE_BROWSER]);
}

//
// called when file dialog "Ok" button gets pressed
//
void
MyTextureEditor::fileDialogOkCB(Widget, MyTextureEditor *p,
                                XmFileSelectionBoxCallbackStruct *data)
{
    // Get the file name
    char *fileName;
    if (!XmStringGetLtoR(data->value, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET, &fileName))
        return;

    p->setNewDialogImage(fileName);
    XtFree(fileName);
}

//
// called when image dialog "Apply" button gets pressed
//
void
MyTextureEditor::imageDialogApplyCB(Widget, MyTextureEditor *p, void *)
{
    if (p->selectedItem == -1)
        return;

    TextureNameStruct *txt = &p->textureNames[p->selectedItem];
    SbBool textureChanged = FALSE;

    // assign new texture
    if (p->dialogImageName != NULL)
    {
        if (txt->fullName == NULL || strcmp(p->dialogImageName, txt->fullName) != 0)
        {
            // add the new texture
            p->addTextureEntry(p->selectedItem, p->dialogImageName);
            textureChanged = TRUE;
        }
    }
    // else clear existing texture
    else
    {
        if (txt->fullName != NULL)
        {
            p->deleteTextureEntry(p->selectedItem);
            textureChanged = TRUE;
        }
    }

    if (textureChanged)
    {
        // update the feedback and texture node
        glXMakeCurrent(p->getDisplay(), XtWindow(p->widgetList[TEXTURE_GLX]),
                       p->paletteCtx);
        p->drawTextureTile(p->selectedItem, DRAW_SELECTED);
        p->updateTextureNode();

        p->savePalette();
    }
}
