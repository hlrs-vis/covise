/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvColEd.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

/*
 * Includes
 */

#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <Xm/Form.h>
#include <Xm/RowColumn.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/CascadeB.h>
#include <Xm/CascadeBG.h>
#include <Xm/BulletinB.h>
#include <Xm/Separator.h>
#include <Xm/SeparatoG.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>

#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/SoLists.h>
#include <Inventor/SoPath.h>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/fields/SoMFColor.h>
#include <Inventor/fields/SoSFColor.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtResource.h>
#include <Inventor/Xt/SoXtClipboard.h>
#include <Inventor/errors/SoDebugError.h>
#include "InvColorPatch.h"
#include "InvColorEditor.h"
#include "InvColorSlider.h"
#include "InvColorWheel.h"

/*
 * Defines
 */

// default window sizes, and layout positions
#define DEFAULT_WIDTH 280
#define DEFAULT_HEIGHT 170
#define TOP_REGION_SIZE 4.1 // size of the upper half in sliders number
#define BUTTONS_FORM_RIGHT_POSITION 50
// pixels offset used for placing things
#define OFFSET 5

// ID list for all parts of the color editor
enum
{
    R_SLIDER_ID = 0, // convenient to have it start a 0
    G_SLIDER_ID,
    B_SLIDER_ID,
    H_SLIDER_ID,
    S_SLIDER_ID,
    V_SLIDER_ID,
    COLOR_WHEEL_ID,
    SAVE_ID,
    SWAP_ID,
    RESTORE_ID,
    ACCEPT_ID,
    CONTINUOUS_ID,
    MANUAL_ID,
    NONE_SLIDER_ID,
    INTENSITY_SLIDER_ID,
    RGB_SLIDERS_ID,
    HSV_SLIDERS_ID,
    RGB_V_SLIDERS_ID,
    RGB_HSV_SLIDERS_ID,
    WYSIWYG_ID,
    COPY_ID,
    PASTE_ID,
    HELP_ID,
    NUM_IDS // this must be last
};

// the menu items which toggle
enum
{
    CONTINUOUS_TOGGLE = 0, // convenient to start at 0
    ACCEPT_TOGGLE,
    WYSIWYG_TOGGLE,
    NONE_TOGGLE,
    INTENSITY_TOGGLE,
    RGB_TOGGLE,
    HSV_TOGGLE,
    RGB_V_TOGGLE,
    RGB_HSV_TOGGLE,
    NUM_TOGGLES // this must be last
};

#define TOGGLE_ON(BUTTON) \
    XmToggleButtonSetState((Widget)BUTTON, TRUE, FALSE)
#define TOGGLE_OFF(BUTTON) \
    XmToggleButtonSetState((Widget)BUTTON, FALSE, FALSE)

//
// struct used for internal callbacks
//
typedef struct ColorEditorCBData
{
    short id;
    class MyColorEditor *classPtr;
} ColorEditorCBData_;

/*
 * Globals vars
 */

// strings used in motif buttons/menus
static const char *slider_labels[] = { "R", "G", "B", "H", "S", "V" };
static const char *button_names[] = { "right", "switch", "left" };
static const char *edit_menu[] = {
    "Continuous", "Manual",
    "sep", "WYSIWYG",
    "sep", "Copy", "Paste",
    "sep", "Help"
};
static const char *slider_menu[] = {
    "None", "Value", "RGB", "HSV",
    "RGB V", "RGB HSV"
};

// arrow pointing to the right
#define right_width 24
#define right_height 12
static unsigned char right_bits[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0xc0, 0x01,
    0x00, 0xc0, 0x07, 0xf0, 0xff, 0x1f, 0xf0, 0xff, 0x1f, 0x00, 0xc0, 0x07,
    0x00, 0xc0, 0x01, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// switch arrow
#define switch_width 24
#define switch_height 12
static unsigned char switch_bits[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x02, 0x70, 0x00, 0x0e,
    0x7c, 0x00, 0x3e, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7c, 0x00, 0x3e,
    0x70, 0x00, 0x0e, 0x40, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// arrow pointing to the left
#define left_width 24
#define left_height 12
static unsigned char left_bits[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x80, 0x03, 0x00,
    0xe0, 0x03, 0x00, 0xf8, 0xff, 0x0f, 0xf8, 0xff, 0x0f, 0xe0, 0x03, 0x00,
    0x80, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

static const char *thisClassName = "MyColorEditor";

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
MyColorEditor::MyColorEditor(
    Widget parent,
    const char *name,
    SbBool buildInsideParent)
    : SoXtComponent(parent, name, buildInsideParent)
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
MyColorEditor::MyColorEditor(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool buildNow)
    : SoXtComponent(parent, name, buildInsideParent)
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(buildNow);
}

////////////////////////////////////////////////////////////////////////
//
// Called by the constructors
//
// private
//
void
MyColorEditor::constructorCommon(SbBool buildNow)
//
//////////////////////////////////////////////////////////////////////
{
    int i;

    // init local vars
    setClassName(thisClassName);
    addVisibilityChangeCallback(visibilityChangeCB, this);
    WYSIWYGmode = FALSE;
    whichSliders = INTENSITY;
    baseRGB[0] = baseRGB[2] = 1.0;
    baseRGB[1] = 0.0;
    baseRGB.getHSVValue(baseHSV);
    acceptButton = slidersForm = NULL;
    mgrWidget = NULL;
    updateFreq = CONTINUOUS;

    // copy/paste support
    clipboard = NULL;

    // default size
    setSize(SbVec2s(DEFAULT_WIDTH, DEFAULT_HEIGHT));

    // color field vars
    attached = FALSE;
    colorSF = NULL;
    colorMF = NULL;
    editNode = NULL;
    colorSensor = new SoNodeSensor(MyColorEditor::fieldChangedCB, this);

    // init callbacks data Ids;
    dataId = new ColorEditorCBData[NUM_IDS];
    for (i = 0; i < NUM_IDS; i++)
    {
        dataId[i].id = i; // since Ids start at 0
        dataId[i].classPtr = this;
    }

    // user callbacks
    callbackList = new SoCallbackList;
    ignoreCallback = FALSE;

    // NULL out UI components. We'll create them in buildWidget().
    wheel = NULL;
    current = NULL;
    previous = NULL;
    for (i = 0; i < 6; i++)
        sliders[i] = NULL;

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

MyColorEditor::~MyColorEditor()
//
////////////////////////////////////////////////////////////////////////
{
    // detaches itself and remove sensor
    if (isAttached())
        detach();

    // delete everything
    delete[] dataId;
    delete clipboard;
    delete callbackList;
    delete wheel;
    delete current;
    delete previous;
    for (int i = 0; i < 6; i++)
        delete sliders[i];
}

////////////////////////////////////////////////////////////////////////
//
//    This routine builds all the widgets, and do the layout using motif.
//
// usage: private
//
Widget
MyColorEditor::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget menubar;
    int n;
    Arg args[12];

    //
    // create a top level form to hold everything together
    //

    SbVec2s size = getSize();
    n = 0;
    if ((size[0] != 0) && (size[1] != 0))
    {
        XtSetArg(args[n], XtNwidth, size[0]);
        n++;
        XtSetArg(args[n], XtNheight, size[1]);
        n++;
    }

    // create the top level widget, then register it with a class name
    mgrWidget = XtCreateWidget(getWidgetName(), xmFormWidgetClass, parent, args, n);
    registerWidget(mgrWidget);

    //
    // build top level components
    //
    menubar = buildPulldownMenu(mgrWidget);
    buttonsForm = buildControls(mgrWidget);

    //
    // allocate color wheel
    //
    wheel = new MyColorWheel(mgrWidget);
    wheel->setBaseColor(baseHSV);
    wheel->addValueChangedCallback(&MyColorEditor::wheelCallback, this);
    wheelForm = wheel->getWidget();

    slidersForm = buildSlidersForm(mgrWidget);

    //
    // layout !
    //
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNrightPosition, BUTTONS_FORM_RIGHT_POSITION);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_NONE);
    n++;
    XtSetValues(menubar, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, menubar);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNrightPosition, BUTTONS_FORM_RIGHT_POSITION);
    n++;
    // Note: bottom attachment changes dynamically
    XtSetValues(buttonsForm, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, buttonsForm);
    n++;
    // Note: bottom attachment changes dynamically
    XtSetValues(wheelForm, args, n);

    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, OFFSET);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, OFFSET);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, OFFSET);
    n++;
    // Note: top attachment changes dynamically
    XtSetValues(slidersForm, args, n);

    doDynamicTopLevelLayout();

    // manage all widgets
    XtManageChild(menubar);
    XtManageChild(buttonsForm);
    XtManageChild(wheelForm);

    //
    // get default values from X resource!
    //
    SoXtResource xr(mgrWidget);
    char *val;
    SbBool b;

    if (xr.getResource((char *)"wysiwyg", (char *)"Wysiwyg", b))
        setWYSIWYG(b);

    //??? we could get the quark for each string, then get the quark for val,
    //??? and compare quarks - it might be faster.
    // strcasecmp is case insensitive
    if (xr.getResource((char *)"colorSliders", (char *)"ColorSliders", val))
    {
        if (strcasecmp(val, "none") == 0)
            setCurrentSliders(NONE);
        else if (strcasecmp(val, "intensity") == 0)
            setCurrentSliders(INTENSITY);
        else if (strcasecmp(val, "rgb") == 0)
            setCurrentSliders(RGB);
        else if (strcasecmp(val, "hsv") == 0)
            setCurrentSliders(HSV);
        else if (strcasecmp(val, "rgb_v") == 0)
            setCurrentSliders(RGB_V);
        else if (strcasecmp(val, "rgb_hsv") == 0)
            setCurrentSliders(RGB_HSV);
    }

    //??? should the base class do the check for continuous and manual?
    if (xr.getResource((char *)"updateFrequency", (char *)"UpdateFrequency", val))
    {
        if (strcasecmp(val, "continuous") == 0)
            setUpdateFrequency(CONTINUOUS);
        else if (strcasecmp(val, "manual") == 0)
            setUpdateFrequency(AFTER_ACCEPT);
    }

    return mgrWidget;
}

////////////////////////////////////////////////////////////////////////
//
//    builds the pulldown menu
//
// usage: private

Widget
MyColorEditor::buildPulldownMenu(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget menuw[2], sub1w[15], sub2w[15];
    int num1 = 0, num2 = 0;
    int i, n;
    Arg args[2];

    //
    // create the pulldown menu
    //
    Widget menubar = XmCreateMenuBar(parent, (char *)"menuBar", NULL, 0);

    // NOTE: menu items must be created in this order!
    menuItems.truncate(0);

    //
    // SUBMENU 1
    //
    Widget sub_menu1 = XmCreatePulldownMenu(menubar, (char *)"", NULL, 0);
    n = 0;
    XtSetArg(args[n], XmNsubMenuId, sub_menu1);
    n++;
    menuw[0] = XtCreateWidget("Edit", xmCascadeButtonGadgetClass,
                              menubar, args, n);

    // CONTINUOUS_ID,
    // MANUAL_ID,
    n = 0;
    XtSetArg(args[n], XmNindicatorType, XmONE_OF_MANY);
    n++;
    for (i = 0; i < 2; i++)
    {
        sub1w[num1] = XtCreateWidget(edit_menu[num1],
                                     xmToggleButtonGadgetClass, sub_menu1, args, n);
        XtAddCallback(sub1w[num1], XmNvalueChangedCallback,
                      (XtCallbackProc)&MyColorEditor::editMenuCallback,
                      (XtPointer)&dataId[CONTINUOUS_ID + i]);
        menuItems.append(sub1w[num1]);
        num1++;
    }

    // SEPARATOR
    sub1w[num1] = XtCreateWidget(edit_menu[num1], xmSeparatorGadgetClass,
                                 sub_menu1, NULL, 0);
    num1++;

    // WYSIWYG_ID
    sub1w[num1] = XtCreateWidget(edit_menu[num1], xmToggleButtonGadgetClass,
                                 sub_menu1, NULL, 0);
    XtAddCallback(sub1w[num1], XmNvalueChangedCallback,
                  (XtCallbackProc)&MyColorEditor::editMenuCallback,
                  (XtPointer)&dataId[WYSIWYG_ID]);
    menuItems.append(sub1w[num1]);
    num1++;

    // SEPARATOR
    sub1w[num1] = XtCreateWidget(edit_menu[num1],
                                 xmSeparatorGadgetClass, sub_menu1, NULL, 0);
    num1++;

    // COPY_ID,
    // PASTE_ID,
    for (i = 0; i < 2; i++)
    {
        sub1w[num1] = XtCreateWidget(edit_menu[num1],
                                     xmPushButtonGadgetClass, sub_menu1, NULL, 0);
        XtAddCallback(sub1w[num1], XmNactivateCallback,
                      (XtCallbackProc)&MyColorEditor::editMenuCallback,
                      (XtPointer)&dataId[COPY_ID + i]);
        num1++;
        // we do not append these to menuItems; since they are push buttons
        // and not toggle buttons, we don't need to save them for future updates
    }

    // SEPARATOR
    sub1w[num1] = XtCreateWidget(edit_menu[num1],
                                 xmSeparatorGadgetClass, sub_menu1, NULL, 0);
    num1++;

    // HELP_ID
    sub1w[num1] = XtCreateWidget(edit_menu[num1],
                                 xmPushButtonGadgetClass, sub_menu1, NULL, 0);
    XtAddCallback(sub1w[num1], XmNactivateCallback,
                  (XtCallbackProc)&MyColorEditor::editMenuCallback,
                  (XtPointer)&dataId[HELP_ID]);
    num1++;

    //
    // SUBMENU 2
    //
    Widget sub_menu2 = XmCreatePulldownMenu(menubar, (char *)"", NULL, 0);
    n = 0;
    XtSetArg(args[n], XmNsubMenuId, sub_menu2);
    n++;
    menuw[1] = XtCreateWidget("Sliders", xmCascadeButtonGadgetClass,
                              menubar, args, n);

    // NONE
    // INTENSITY
    // RGB
    // HSV
    // RGB_V
    // RGB_HSV
    n = 0;
    XtSetArg(args[n], XmNindicatorType, XmONE_OF_MANY);
    n++;
    for (i = 0; i < 6; i++)
    {
        sub2w[num2] = XtCreateWidget(slider_menu[num2],
                                     xmToggleButtonGadgetClass, sub_menu2, args, n);
        XtAddCallback(sub2w[num2], XmNvalueChangedCallback,
                      (XtCallbackProc)&MyColorEditor::sliderMenuCallback,
                      (XtPointer)&dataId[NONE_SLIDER_ID + i]);
        menuItems.append(sub2w[num2]);
        num2++;
    }

    // menu display callback so we can change the menu item
    XtAddCallback(sub_menu1, XmNmapCallback,
                  (XtCallbackProc)MyColorEditor::menuDisplay, (XtPointer) this);
    XtAddCallback(sub_menu2, XmNmapCallback,
                  (XtCallbackProc)MyColorEditor::menuDisplay, (XtPointer) this);

    // manage all children
    XtManageChildren(sub1w, num1);
    XtManageChildren(sub2w, num2);
    XtManageChildren(menuw, 2);

    return menubar;
}

////////////////////////////////////////////////////////////////////////
//
//    builds the color swatches, arrow buttons and accept button.
//
// usage: private

Widget
MyColorEditor::buildControls(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget curW, prevW, buttonw[3], patchButForm;
    int i, n;
    Arg args[12];

    //
    // build the buttons form
    //
    buttonsForm = XtCreateWidget("buttonsForm", xmFormWidgetClass, parent, NULL, 0);

    // build swatches
    current = new MyColorPatch(buttonsForm, "Current");
    current->setColor(baseRGB);
    curW = current->getWidget();
    previous = new MyColorPatch(buttonsForm, "Previous");
    previous->setColor(baseRGB);
    prevW = previous->getWidget();

    //
    // create the arrow buttons (use a form to lay them inside)
    //
    patchButForm = XtCreateWidget("patchButForm", xmFormWidgetClass,
                                  buttonsForm, NULL, 0);
    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
// ??? bug 228368 prevents the pixmap from also highlighting
#ifdef __sgi
    XtSetArg(args[n], SgNpixmapLocateHighlight, True);
    n++;
#endif
    for (i = 0; i < 3; i++)
    {
        buttonw[i] = XtCreateWidget(button_names[i], xmPushButtonGadgetClass,
                                    patchButForm, args, n);
        XtAddCallback(buttonw[i], XmNactivateCallback,
                      (XtCallbackProc)&MyColorEditor::buttonsCallback,
                      (XtPointer)&dataId[SAVE_ID + i]);
    }

    //
    // create the pixmaps for the arrow buttons
    //
    Pixmap pixmaps[3][2];
    Display *display = XtDisplay(parent);
    Drawable d = DefaultRootWindow(display);
    Pixel fg, bg, hbg;

    // get the color of the push buttons
    XtVaGetValues(buttonw[0], XmNforeground, &fg, XmNbackground, &bg, NULL);
    //#if defined(__hpux) || defined (__linux__)
    hbg = bg;
    //#else
    //   hbg = SgGetLocatePixel(buttonw[0], bg);
    //#endif

    // create the pixmaps from the bitmap data (depth 1).
    // Two sets of pixmaps are created for when the button is
    // up and down.
    int depth = XDefaultDepthOfScreen(XtScreen(parent));
    pixmaps[0][0] = XCreatePixmapFromBitmapData(display, d,
                                                reinterpret_cast<char *>(right_bits), right_width, right_height, fg, bg, depth);
    pixmaps[0][1] = XCreatePixmapFromBitmapData(display, d,
                                                reinterpret_cast<char *>(right_bits), right_width, right_height, fg, hbg, depth);
    pixmaps[1][0] = XCreatePixmapFromBitmapData(display, d,
                                                reinterpret_cast<char *>(switch_bits), switch_width, switch_height, fg, bg, depth);
    pixmaps[1][1] = XCreatePixmapFromBitmapData(display, d,
                                                reinterpret_cast<char *>(switch_bits), switch_width, switch_height, fg, hbg, depth);
    pixmaps[2][0] = XCreatePixmapFromBitmapData(display, d,
                                                reinterpret_cast<char *>(left_bits), left_width, left_height, fg, bg, depth);
    pixmaps[2][1] = XCreatePixmapFromBitmapData(display, d,
                                                reinterpret_cast<char *>(left_bits), left_width, left_height, fg, hbg, depth);

    // assign the pixmaps to the push buttons
    XtSetArg(args[0], XmNlabelType, XmPIXMAP);
    for (i = 0; i < 3; i++)
    {
        XtSetArg(args[1], XmNlabelPixmap, pixmaps[i][0]);
#ifdef __sgi
        XtSetArg(args[2], SgNlocatePixmap, pixmaps[i][1]);
#endif
        XtSetValues(buttonw[i], args, 3);
    }

    //
    // build the accept button
    //
    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    acceptButton = XtCreateWidget("Accept", xmPushButtonGadgetClass,
                                  buttonsForm, args, n);
    XtAddCallback(acceptButton, XmNactivateCallback,
                  (XtCallbackProc)&MyColorEditor::buttonsCallback,
                  (XtPointer)&dataId[ACCEPT_ID]);

    //
    // layout !
    //
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNleftPosition, 10);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNrightPosition, 49);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNtopPosition, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNbottomPosition, 45);
    n++;
    XtSetValues(curW, args, n);
    XtSetArg(args[1], XmNleftPosition, 51);
    XtSetArg(args[3], XmNrightPosition, 90);
    XtSetValues(prevW, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[3], XmNleftPosition, 0);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[5], XmNrightPosition, 30);
    n++;
    XtSetValues(buttonw[0], args, n);
    XtSetArg(args[3], XmNleftPosition, 31);
    XtSetArg(args[5], XmNrightPosition, 69);
    XtSetValues(buttonw[1], args, n);
    XtSetArg(args[3], XmNleftPosition, 70);
    XtSetArg(args[5], XmNrightPosition, 100);
    XtSetValues(buttonw[2], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, curW);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, curW);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightWidget, prevW);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_NONE);
    n++;
    XtSetValues(patchButForm, args, n);

    int offset = (whichSliders == NONE) ? 0 : OFFSET;
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNleftPosition, 30);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNrightPosition, 70);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_NONE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, offset);
    n++;
    XtSetValues(acceptButton, args, n);

    // manage all children
    XtManageChild(curW);
    XtManageChild(prevW);
    XtManageChildren(buttonw, 3);
    XtManageChild(patchButForm);
    if (updateFreq == AFTER_ACCEPT)
        XtManageChild(acceptButton);

    return buttonsForm;
}

////////////////////////////////////////////////////////////////////////
//
//    builds the sliders form
//
// usage: private

Widget
MyColorEditor::buildSlidersForm(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[12];

    //
    // build the sliders form
    //
    n = 0;
    XtSetArg(args[n], XmNfractionBase, 1000);
    n++;
    slidersForm = XtCreateWidget("slidersForm", xmFormWidgetClass,
                                 parent, args, n);

    // build sliders
    sliders[0] = new MyColorSlider(slidersForm, NULL, TRUE,
                                   MyColorSlider::RED_SLIDER);
    sliders[1] = new MyColorSlider(slidersForm, NULL, TRUE,
                                   MyColorSlider::GREEN_SLIDER);
    sliders[2] = new MyColorSlider(slidersForm, NULL, TRUE,
                                   MyColorSlider::BLUE_SLIDER);
    sliders[3] = new MyColorSlider(slidersForm, NULL, TRUE,
                                   MyColorSlider::HUE_SLIDER);
    sliders[4] = new MyColorSlider(slidersForm, NULL, TRUE,
                                   MyColorSlider::SATURATION_SLIDER);
    sliders[5] = new MyColorSlider(slidersForm, NULL, TRUE,
                                   MyColorSlider::VALUE_SLIDER);

    int i;
    for (i = 0; i < 3; i++)
        sliders[i]->setBaseColor(baseRGB.getValue());
    for (i = 3; i < 6; i++)
        sliders[i]->setBaseColor(baseHSV);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    for (i = 0; i < 6; i++)
    {
        sliders[i]->setLabel(slider_labels[i]);
        sliders[i]->addValueChangedCallback(
            &MyColorEditor::sliderCallback,
            &dataId[R_SLIDER_ID + i]);
        XtSetValues(sliders[i]->getWidget(), args, n);
    }

    //
    // layout !
    //
    doSliderLayout();

    return slidersForm;
}

////////////////////////////////////////////////////////////////////////
//
//    This routine attaches itself to a single color field.
//
// usage: public

void
MyColorEditor::attach(SoSFColor *sf, SoBase *node)
//
////////////////////////////////////////////////////////////////////////
{
    if (isAttached())
        detach();

    if (sf != NULL && node != NULL)
    {
        setColor(sf->getValue());
        colorSF = sf;
        editNode = node;
        editNode->ref();
        colorSensor->attach((SoNode *)editNode);
        attached = TRUE;
    }
}

////////////////////////////////////////////////////////////////////////
//
//    This routine attaches itself to a multiple value color field.
//
// usage: public

void
MyColorEditor::attach(SoMFColor *mf, int ind, SoBase *node)
//
////////////////////////////////////////////////////////////////////////
{
    if (isAttached())
        detach();

    if (mf != NULL && ind >= 0 && node != NULL)
    {
        setColor((*mf)[ind]);
        colorMF = mf;
        index = ind;
        editNode = node;
        editNode->ref();
        colorSensor->attach((SoNode *)editNode);
        attached = TRUE;
    }
}

////////////////////////////////////////////////////////////////////////
//
//    This routine detaches itself from the color field.
//
// usage: public

void
MyColorEditor::detach()
//
////////////////////////////////////////////////////////////////////////
{
    if (!isAttached())
        return;

    colorSensor->detach();
    editNode->unref();
    editNode = NULL;
    colorSF = NULL;
    colorMF = NULL;
    attached = FALSE;
}

////////////////////////////////////////////////////////////////////////
//
//    This routine sets the current color.
//
// usage: public
//
void
MyColorEditor::setColor(const SbColor &color)
//
////////////////////////////////////////////////////////////////////////
{
    if (color == baseRGB)
        return;

    // save color
    baseRGB = color;
    baseRGB.getHSVValue(baseHSV);

    ignoreCallback = TRUE;

    // now send the colors to the sliders/color wheel
    int i;
    for (i = 0; i < 3; i++)
        sliders[i]->setBaseColor(baseRGB.getValue());
    for (i = 3; i < 6; i++)
        sliders[i]->setBaseColor(baseHSV);
    wheel->setBaseColor(baseHSV);
    current->setColor(baseRGB);

    ignoreCallback = FALSE;

    if (updateFreq == CONTINUOUS)
        doUpdates();
}

////////////////////////////////////////////////////////////////////////
//
//    This routine sets the WYSIWYG mode.
//
// usage: public

void
MyColorEditor::setWYSIWYG(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (WYSIWYGmode == flag)
        return;

    WYSIWYGmode = flag;

    // now update the sliders and color wheel
    for (int i = 0; i < 6; i++)
        sliders[i]->setWYSIWYG(WYSIWYGmode);
    wheel->setWYSIWYG(WYSIWYGmode);
}

////////////////////////////////////////////////////////////////////////
//
//    This routine sets the update frequency.
//
// usage: virtual public
//
void
MyColorEditor::setUpdateFrequency(MyColorEditor::UpdateFrequency freq)
//
////////////////////////////////////////////////////////////////////////
{
    if (updateFreq == freq)
        return;

    updateFreq = freq;

    // show/hide the accept button
    if (acceptButton != NULL)
    {
        if (updateFreq == CONTINUOUS)
            XtUnmanageChild(acceptButton);
        else
            XtManageChild(acceptButton);
    }

    // update the attached node if we switch to continous
    if (updateFreq == CONTINUOUS)
        doUpdates();
}

////////////////////////////////////////////////////////////////////////
//
//    This routine specifies which sliders are being displayed.
//
// usage: public

void
MyColorEditor::setCurrentSliders(MyColorEditor::Sliders id)
//
////////////////////////////////////////////////////////////////////////
{
    int i, prevNum, curNum;

    if (whichSliders == id)
        return;

    prevNum = numberOfSliders(whichSliders);
    curNum = numberOfSliders(id);

    // check to make sure widget has been built, otherwise just change the
    // default window size.
    if (mgrWidget == NULL)
    {
        // set new height
        SbVec2s size = getSize();
        float r = (TOP_REGION_SIZE + curNum) / float(TOP_REGION_SIZE + prevNum);
        size[1] = short(size[1] * r);
        setSize(size);
        whichSliders = id;
        return;
    }

    //
    // hide the current set of sliders
    //
    switch (whichSliders)
    {
    case NONE:
        break;
    case INTENSITY:
        sliders[5]->hide();
        break;
    case RGB:
        for (i = 0; i < 3; i++)
            sliders[i]->hide();
        break;
    case HSV:
        for (i = 3; i < 6; i++)
            sliders[i]->hide();
        break;
    case RGB_V:
        for (i = 0; i < 3; i++)
            sliders[i]->hide();
        sliders[5]->hide();
        break;
    case RGB_HSV:
        for (i = 0; i < 6; i++)
            sliders[i]->hide();
        break;
    }

    //
    // check if window needs to be resized
    //
    Widget parent = XtParent(mgrWidget);
    if (XtIsShell(parent) && curNum != prevNum)
    {

        // get current window height and find new height
        SbVec2s size = getSize();
        float r = (TOP_REGION_SIZE + curNum) / float(TOP_REGION_SIZE + prevNum);
        size[1] = short(size[1] * r);
        SoXt::setWidgetSize(parent, size);
    }

    // finally do new layout
    whichSliders = id;
    doDynamicTopLevelLayout();
    doSliderLayout();
}

////////////////////////////////////////////////////////////////////////
//
//    routine to lay sliders out within the slider form.
//
// usage: protected

void
MyColorEditor::doSliderLayout()
//
////////////////////////////////////////////////////////////////////////
{
    int i, n;
    Arg args[4];

    ignoreCallback = TRUE;

    switch (whichSliders)
    {
    case NONE:
        break;

    case INTENSITY:
        n = 0;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_POSITION);
        n++;
        XtSetArg(args[n], XmNbottomPosition, 990);
        n++;
        XtSetValues(sliders[5]->getWidget(), args, n);
        sliders[5]->setBaseColor(baseHSV);
        sliders[5]->show();
        break;

    case RGB:
        for (i = 0; i < 3; i++)
        {
            n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNtopPosition, int((i * 1000) / 3.0));
            n++;
            XtSetArg(args[n], XmNbottomAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNbottomPosition, int(((i + 1) * 1000) / 3.0) - 10);
            n++;
            XtSetValues(sliders[i]->getWidget(), args, n);
            sliders[i]->setBaseColor(baseRGB.getValue());
            sliders[i]->show();
        }
        break;

    case HSV:
        for (i = 3; i < 6; i++)
        {
            n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNtopPosition, int(((i - 3) * 1000) / 3.0));
            n++;
            XtSetArg(args[n], XmNbottomAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNbottomPosition, int(((i - 2) * 1000) / 3.0) - 10);
            n++;
            XtSetValues(sliders[i]->getWidget(), args, n);
            sliders[i]->setBaseColor(baseHSV);
            sliders[i]->show();
        }
        break;

    case RGB_V:
        for (i = 0; i < 4; i++)
        {
            n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNtopPosition, i * 250);
            n++;
            XtSetArg(args[n], XmNbottomAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNbottomPosition, (i + 1) * 250 - 10);
            n++;
            if (i == 3)
            {
                XtSetValues(sliders[5]->getWidget(), args, n);
                sliders[5]->setBaseColor(baseHSV);
                sliders[5]->show();
            }
            else
            {
                XtSetValues(sliders[i]->getWidget(), args, n);
                sliders[i]->setBaseColor(baseRGB.getValue());
                sliders[i]->show();
            }
        }
        break;

    case RGB_HSV:
        for (i = 0; i < 6; i++)
        {
            n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNtopPosition, int((i * 1000) / 6.0));
            n++;
            XtSetArg(args[n], XmNbottomAttachment, XmATTACH_POSITION);
            n++;
            XtSetArg(args[n], XmNbottomPosition, int(((i + 1) * 1000) / 6.0) - 10);
            n++;
            XtSetValues(sliders[i]->getWidget(), args, n);
            if (i > 2)
                sliders[i]->setBaseColor(baseHSV);
            else
                sliders[i]->setBaseColor(baseRGB.getValue());
            sliders[i]->show();
        }
        break;
    }

    ignoreCallback = FALSE;
}

////////////////////////////////////////////////////////////////////////
//
// routine which does the top level forms layout (slider form, button form
// and colorWheel) which changes dynamically as the number of sliders changes.
//
// usage: private

void
MyColorEditor::doDynamicTopLevelLayout()
//
////////////////////////////////////////////////////////////////////////
{
    int n, num = numberOfSliders(whichSliders);
    Arg args[4];

    if (num)
    {
        // calculate the sliders form top position based on how many sliders there are
        float top = 100 * TOP_REGION_SIZE / float(TOP_REGION_SIZE + num);
        n = 0;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_POSITION);
        n++;
        XtSetArg(args[n], XmNtopPosition, int(top));
        n++;
        XtSetValues(slidersForm, args, n);

        if (!XtIsManaged(slidersForm))
            XtManageChild(slidersForm);

        n = 0;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNbottomWidget, slidersForm);
        n++;
        XtSetArg(args[n], XmNbottomOffset, OFFSET);
        n++;
        XtSetValues(buttonsForm, args, n);
        XtSetValues(wheelForm, args, n);

        n = 0;
        XtSetArg(args[n], XmNbottomOffset, 0);
        n++;
        XtSetValues(acceptButton, args, n);
    }
    else
    {
        // no sliders so don't use the slidersForm at all
        n = 0;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNbottomOffset, 0);
        n++;
        XtSetValues(buttonsForm, args, n);
        XtSetValues(wheelForm, args, n);

        if (XtIsManaged(slidersForm))
            XtUnmanageChild(slidersForm);

        n = 0;
        XtSetArg(args[n], XmNbottomOffset, OFFSET);
        n++;
        XtSetValues(acceptButton, args, n);
    }
}

////////////////////////////////////////////////////////////////////////
//
//    This routine is called (by fieldChangedCB) when the color field
//    has changed value.
//
// usage: protected

void
MyColorEditor::fieldChanged()
//
////////////////////////////////////////////////////////////////////////
{
    if (colorSF != NULL)
        setColor(colorSF->getValue());
    else
        setColor((*colorMF)[index]);
}

////////////////////////////////////////////////////////////////////////
//
//    Called when the color wheel changes the current color.
//
// usage: private
//
void
MyColorEditor::wheelChanged(const float hsv[3])
//
////////////////////////////////////////////////////////////////////////
{
    int i;

    // wheel can only change hue and saturation
    baseHSV[0] = hsv[0];
    baseHSV[1] = hsv[1];
    baseRGB.setHSVValue(baseHSV);

    ignoreCallback = TRUE;

    switch (whichSliders)
    {
    case NONE:
        break;
    case INTENSITY:
        sliders[5]->setBaseColor(baseHSV);
        break;
    case RGB:
    case RGB_V:
        for (i = 0; i < 3; i++)
            sliders[i]->setBaseColor(baseRGB.getValue());
        if (whichSliders == RGB_V)
            sliders[5]->setBaseColor(baseHSV);
        break;
    case HSV:
        for (i = 3; i < 6; i++)
            sliders[i]->setBaseColor(baseHSV);
        break;
    case RGB_HSV:
        for (i = 0; i < 3; i++)
            sliders[i]->setBaseColor(baseRGB.getValue());
        for (i = 3; i < 6; i++)
            sliders[i]->setBaseColor(baseHSV);
        break;
    }
    current->setColor(baseRGB);

    ignoreCallback = FALSE;

    if (updateFreq == CONTINUOUS)
        doUpdates();
}

////////////////////////////////////////////////////////////////////////
//
//    This routine is called when the sliders changes the current color.
//
// usage: private
//
void
MyColorEditor::sliderChanged(short id, float value)
//
////////////////////////////////////////////////////////////////////////
{
    int i;

    ignoreCallback = TRUE;

    switch (id)
    {
    case R_SLIDER_ID:
    case G_SLIDER_ID:
    case B_SLIDER_ID:
        baseRGB[id - R_SLIDER_ID] = value;
        baseRGB.getHSVValue(baseHSV);

        for (i = 0; i < 3; i++)
            if (i != id)
                sliders[i]->setBaseColor(baseRGB.getValue());

        if (whichSliders == RGB_V)
            sliders[5]->setBaseColor(baseHSV);
        else if (whichSliders == RGB_HSV)
            for (i = 3; i < 6; i++)
                sliders[i]->setBaseColor(baseHSV);
        wheel->setBaseColor(baseHSV);
        current->setColor(baseRGB);
        break;

    case H_SLIDER_ID:
    case S_SLIDER_ID:
    case V_SLIDER_ID:
        baseHSV[id - H_SLIDER_ID] = value;
        baseRGB.setHSVValue(baseHSV);

        switch (whichSliders)
        {
        case HSV:
        case RGB_HSV:
            for (i = 3; i < 6; i++)
                if (i != id)
                    sliders[i]->setBaseColor(baseHSV);
            if (whichSliders == RGB_HSV)
                for (i = 0; i < 3; i++)
                    sliders[i]->setBaseColor(baseRGB.getValue());
            break;
        case RGB_V:
            for (i = 0; i < 3; i++)
                sliders[i]->setBaseColor(baseRGB.getValue());
            break;
        case INTENSITY:
            break;
        case RGB: // not possible cases
        case NONE:
#ifdef DEBUG
            SoDebugError::post("MyColorEditor::sliderChanged",
                               "inconsitant state %d", id);
#endif
            break;
        }
        wheel->setBaseColor(baseHSV);
        current->setColor(baseRGB);
        break;

    default:
#ifdef DEBUG
        SoDebugError::post("MyColorEditor::sliderChanged",
                           "bad id %d", id);
#endif
        break;
    }

    ignoreCallback = FALSE;

    if (updateFreq == CONTINUOUS)
        doUpdates();
}

////////////////////////////////////////////////////////////////////////
//
//    This routine is called when the motif buttons are being pressed.
//
// usage: protected

void
MyColorEditor::buttonPressed(short id)
//
////////////////////////////////////////////////////////////////////////
{
    SbColor col;

    switch (id)
    {
    case SAVE_ID:
        previous->setColor(baseRGB);
        break;

    case SWAP_ID:
    case RESTORE_ID:
        col = previous->getColor();

        if (id == SWAP_ID)
            previous->setColor(baseRGB);

        // assign new color
        setColor(col);

        if (updateFreq != AFTER_ACCEPT)
            doUpdates();
        break;

    case ACCEPT_ID:
        doUpdates();
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
//    Do the updates - if node is attached, update it; if callback exists,
//  call it.
//
// usage: private
//
void
MyColorEditor::doUpdates()
//
////////////////////////////////////////////////////////////////////////
{
    // check for field update
    if (attached)
    {
        if (colorSF != NULL)
        {
            colorSF->setValue(baseRGB);
            if (colorSF->isIgnored())
                colorSF->setIgnored(FALSE);
        }
        else
        {
            colorMF->set1Value(index, baseRGB);
            if (colorMF->isIgnored())
                colorMF->setIgnored(FALSE);
        }
    }

    // check for callback
    void *hackage = (void *)&baseRGB;
    callbackList->invokeCallbacks(hackage);
}

////////////////////////////////////////////////////////////////////////
//
// convenience routine which returns the number of sliders, given
// MyColorEditorSliders id.
//
// usage: private

int
MyColorEditor::numberOfSliders(MyColorEditor::Sliders id)
//
////////////////////////////////////////////////////////////////////////
{
    switch (id)
    {
    case NONE:
        return 0;
    case INTENSITY:
        return 1;
    case RGB:
    case HSV:
        return 3;
    case RGB_V:
        return 4;
    case RGB_HSV:
        return 6;
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////
//
//  Copy the current color onto the clipboard.
//
//  Use: private
//
void
MyColorEditor::copy(Time eventTime)
//
////////////////////////////////////////////////////////////////////////
{
#ifdef DEBUG
    if (mgrWidget == NULL)
    {
        SoDebugError::post("MyColorEditor::copy", "widget is NULL\n");
        return;
    }
#endif

    if (clipboard == NULL)
        clipboard = new SoXtClipboard(mgrWidget);

    // copy the current color using a BaseColor node
    SoBaseColor *color = new SoBaseColor;
    color->ref();
    color->rgb.setValue(baseRGB);
    clipboard->copy(color, eventTime);
    color->unref();
}

////////////////////////////////////////////////////////////////////////
//
//  Retrieve the selection from the X server and paste it when it
//  arrives (in our pasteDone callback).
//
//  Use: private
//
void
MyColorEditor::paste(Time eventTime)
//
////////////////////////////////////////////////////////////////////////
{
#ifdef DEBUG
    if (mgrWidget == NULL)
    {
        SoDebugError::post("MyColorEditor::paste",
                           "widget is NULL\n");
        return;
    }
#endif

    if (clipboard == NULL)
        clipboard = new SoXtClipboard(mgrWidget);

    clipboard->paste(eventTime, MyColorEditor::pasteDoneCB, this);
}

////////////////////////////////////////////////////////////////////////
//
//  The X server has finished getting the selection data, and the
//  paste is complete. Look through the paste data for a base color node.
//
//  Use: private
//
void
MyColorEditor::pasteDone(SoPathList *pathList)
//
////////////////////////////////////////////////////////////////////////
{
    SoSearchAction sa;
    SoFullPath *fullPath = NULL;

    //
    // search for first base color node in that pasted scene
    //
    sa.setType(SoBaseColor::getClassTypeId());
    for (int i = 0; i < pathList->getLength(); i++)
    {
        sa.apply((*pathList)[i]);
        if ((fullPath = (SoFullPath *)sa.getPath()) != NULL)
        {

            // assign new color
            SoBaseColor *newColor = (SoBaseColor *)fullPath->getTail();
            setColor((newColor->rgb)[0]);

            break;
        }
    }

    //
    // else search for the first material and extract a color from it
    // (the diffuse color, which is better than doing nothing)
    //
    if (fullPath == NULL)
    {
        sa.setType(SoMaterial::getClassTypeId());
        for (int i = 0; i < pathList->getLength(); i++)
        {
            sa.apply((*pathList)[i]);
            if ((fullPath = (SoFullPath *)sa.getPath()) != NULL)
            {

                SoMaterial *mat = (SoMaterial *)fullPath->getTail();
                setColor(mat->diffuseColor[0].getValue());

                break;
            }
        }
    }

    // ??? We delete the callback data when done with it.
    delete pathList;
}

////////////////////////////////////////////////////////////////////////
//
//  Called by Xt when a menu is about to be displayed.
//  This gives us a chance to update any items in the menu.
//
//  Use: static private
//
void
MyColorEditor::menuDisplay(Widget, MyColorEditor *editor, XtPointer)
//
////////////////////////////////////////////////////////////////////////
{
    // turn them all off
    for (int i = 0; i < NUM_TOGGLES; i++)
        TOGGLE_OFF(editor->menuItems[i]);

    // set default toggles
    switch (editor->updateFreq)
    {
    case CONTINUOUS:
        TOGGLE_ON(editor->menuItems[CONTINUOUS_TOGGLE]);
        break;
    case AFTER_ACCEPT:
        TOGGLE_ON(editor->menuItems[ACCEPT_TOGGLE]);
        break;
    }

    if (editor->WYSIWYGmode)
        TOGGLE_ON(editor->menuItems[WYSIWYG_TOGGLE]);

    switch (editor->whichSliders)
    {
    case NONE:
        TOGGLE_ON(editor->menuItems[NONE_TOGGLE]);
        break;
    case INTENSITY:
        TOGGLE_ON(editor->menuItems[INTENSITY_TOGGLE]);
        break;
    case RGB:
        TOGGLE_ON(editor->menuItems[RGB_TOGGLE]);
        break;
    case HSV:
        TOGGLE_ON(editor->menuItems[HSV_TOGGLE]);
        break;
    case RGB_V:
        TOGGLE_ON(editor->menuItems[RGB_V_TOGGLE]);
        break;
    case RGB_HSV:
        TOGGLE_ON(editor->menuItems[RGB_HSV_TOGGLE]);
        break;
    }
}

//
// redefine those generic virtual functions
//
const char *
MyColorEditor::getDefaultWidgetName() const
{
    return thisClassName;
}

const char *
MyColorEditor::getDefaultTitle() const
{
    return "Color Editor";
}

const char *
MyColorEditor::getDefaultIconTitle() const
{
    return "Color Editor";
}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

//
// called whenever the component becomes visibble or not
//
void
MyColorEditor::visibilityChangeCB(void *pt, SbBool visible)
//
{
    MyColorEditor *p = (MyColorEditor *)pt;

    if (visible)
    {
        // attach sensor to top node for redrawing purpose
        if (p->editNode != NULL && p->colorSensor->getAttachedNode() == NULL)
            p->colorSensor->attach((SoNode *)p->editNode);
    }
    else
        // detach sensor
        p->colorSensor->detach();
}

void
MyColorEditor::wheelCallback(void *p, const float hsv[3])
{
    MyColorEditor *c = (MyColorEditor *)p;

    if (c->ignoreCallback)
        return;
    c->wheelChanged(hsv);
}

void
MyColorEditor::sliderCallback(void *p, float value)
{
    ColorEditorCBData *d = (ColorEditorCBData *)p;

    if (d->classPtr->ignoreCallback)
        return;
    d->classPtr->sliderChanged(d->id, value);
}

void
MyColorEditor::buttonsCallback(Widget, ColorEditorCBData *d, XtPointer)
{
    d->classPtr->buttonPressed(d->id);
}

void
MyColorEditor::fieldChangedCB(void *pt, SoSensor *)
{
    MyColorEditor *p = (MyColorEditor *)pt;
    if (!p->isVisible())
        return;

    p->fieldChanged();
}

void
MyColorEditor::editMenuCallback(
    Widget, ColorEditorCBData *d, XmAnyCallbackStruct *cb)
{
    Time eventTime = cb->event->xbutton.time;

    switch (d->id)
    {
    case CONTINUOUS_ID:
        d->classPtr->setUpdateFrequency(CONTINUOUS);
        break;
    case MANUAL_ID:
        d->classPtr->setUpdateFrequency(AFTER_ACCEPT);
        break;
    case WYSIWYG_ID:
        d->classPtr->setWYSIWYG( // toggle
            !d->classPtr->WYSIWYGmode);
        break;
    case COPY_ID:
        d->classPtr->copy(eventTime);
        break;
    case PASTE_ID:
        d->classPtr->paste(eventTime);
        break;
    case HELP_ID:
        d->classPtr->openHelpCard("MyColorEditor.help");
        break;
    }
}

void
MyColorEditor::sliderMenuCallback(Widget, ColorEditorCBData *d, XtPointer)
{
    switch (d->id)
    {
    case NONE_SLIDER_ID:
        d->classPtr->setCurrentSliders(NONE);
        break;
    case INTENSITY_SLIDER_ID:
        d->classPtr->setCurrentSliders(INTENSITY);
        break;
    case RGB_SLIDERS_ID:
        d->classPtr->setCurrentSliders(RGB);
        break;
    case HSV_SLIDERS_ID:
        d->classPtr->setCurrentSliders(HSV);
        break;
    case RGB_V_SLIDERS_ID:
        d->classPtr->setCurrentSliders(RGB_V);
        break;
    case RGB_HSV_SLIDERS_ID:
        d->classPtr->setCurrentSliders(RGB_HSV);
        break;
    }
}

void
MyColorEditor::pasteDoneCB(void *userData, SoPathList *pathList)
{
    ((MyColorEditor *)userData)->pasteDone(pathList);
}
