/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log: InvObjectEditor.C,v $
 * Revision 1.1  1994/07/17  13:39:31  zrfu0390
 * Initial revision
 * */

//**************************************************************************
//
// * Description    : Inventor Object Editor
//
// * Class(es)      : InvObjectEditor
//
// * inherited from : SoXtComponent
//
// * Author  : Dirk Rantzau
//
// * History : 17.07.95 V 1.0
//
//**************************************************************************

#include <string.h>
#include <math.h>

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
#include <Xm/RowColumn.h>

#include <Inventor/Xt/SoXt.h>
#include <Inventor/SbLinear.h>
#include <Inventor/SoDB.h>
#include <Inventor/SoInput.h>
#include <Inventor/SoPath.h>
#include <Inventor/SoLists.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/Xt/SoXtClipboard.h>
#include <Inventor/Xt/SoXtRenderArea.h>
#include <Inventor/errors/SoDebugError.h>

//#include "InvObjectList.h"
//#include "InvCoviseViewer.h"
#include "InvObjectEditor.h"
//#include "InvObjectManager.h"

//**************************************************************************
// CLASS InvObjList
//**************************************************************************
//=======================================================================
//
// Public constructor - build the widget right now
//
//=======================================================================
InvObjList::InvObjList(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool showName)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)

{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(showName, TRUE);
}

//=========================================================================
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
//
//=========================================================================
InvObjList::InvObjList(Widget parent, const char *name,
                       SbBool buildInsideParent, SbBool showName,
                       SbBool buildNow)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(showName, buildNow);
}

//=========================================================================
//
// Called by the constructors
//
// private
//
//=========================================================================
void InvObjList::constructorCommon(SbBool showName, SbBool buildNow)

{

    // init local vars
    setClassName("InvObjList");

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
}

//=========================================================================
//
//    Destructor.
//
InvObjList::~InvObjList()
//
//=========================================================================
{
}

//=========================================================================
//
// Description:
//  	Buils the editor layout
//
// Use: protected
//
//=========================================================================
Widget InvObjList::buildWidget(Widget parent)
{

    const int citem = 15; // no. of visible items
    int n, i;
    Arg args[12];

    // create a top level form to hold everything together
    n = 0;
    XtSetArg(args[n], XmNfractionBase, 100);
    n++;
    form = XmCreateForm(parent, "", args, n);

    // create row column widget for label and list
    rowcol = XtVaCreateManagedWidget("RowColumn",
                                     xmRowColumnWidgetClass, form,
                                     XmNshadowType, XmSHADOW_ETCHED_OUT,
                                     XmNpacking, XmPACK_TIGHT,
                                     XmNorientation, XmVERTICAL,
                                     NULL);

    // create list title
    label = XtVaCreateManagedWidget("Label",
                                    xmLabelWidgetClass, rowcol,
                                    XmNshadowType, XmSHADOW_ETCHED_OUT,
                                    XmNlabelString, XmStringCreateSimple((char *)getDefaultTitle()),
                                    NULL);

    // create scrolled list
    list = XmCreateScrolledList(rowcol, "List", NULL, 0);
    XtManageChild(list);
    XtVaSetValues(list,
                  XmNsensitive, True, // Changed from false
                  //XmNitemCount,		     nitem,
                  //XmNitems,	             items,
                  XmNselectionPolicy, XmSINGLE_SELECT,
                  XmNvisibleItemCount, citem,
                  XmNscrollBarDisplayPolicy, XmSTATIC,
                  //XmNsensitive,	             Master,
                  NULL);

    XtAddCallback(list, XmNsingleSelectionCallback,
                  (XtCallbackProc)InvObjList::objectListCB, (XtPointer) this);

    return form;
}

//=========================================================================
//
// redefine those generic virtual functions
//
//=========================================================================
const char *InvObjList::getDefaultWidgetName() const
{
    return "InvObjList";
}

const char *InvObjList::getDefaultTitle() const
{
    return "COVISE Object List";
}

const char *InvObjList::getDefaultIconTitle() const
{
    return "Object List";
}

//======================================================================
//
// Description:
// called by the list widget
//
// Use: private, static
//
//======================================================================

void InvObjList::objectListCB(Widget w, InvObjList *l, XmListCallbackStruct *list_data)
{

    char name[255];
    char *itemText;

    XmStringGetLtoR(list_data->item, XmSTRING_DEFAULT_CHARSET, &itemText);
    strcpy(name, itemText);

    XmListSelectItem(l->list, list_data->item, False);
}

//======================================================================
//
// Description:
//	update object list according to selected objects.
//
// Use: private
//
//======================================================================
void InvObjList::addToObjectList(char *name)
{

    XmListAddItem(list, XmStringCreateSimple(name), 0);
}

//======================================================================
//
// Description:
//	remove object from list according to selected objects.
//
// Use: private
//
//======================================================================
void InvObjList::removeFromObjectList(char *name)
{

    // first deselect all items
    XmListDeselectAllItems(list);

    // remove entry in list
    XmListDeleteItem(list, XmStringCreateSimple(name));
}

//======================================================================
//
// Description:
//	remove object from list according to selected objects.
//
// Use: private
//
//======================================================================
void InvObjList::updateObjectList(char *name, int isSelection)
{

    int itempos = XmListItemPos(list, XmStringCreateSimple(name));

    // select or deselect the itempos

    if (isSelection)
    {
        XmListSelectPos(list, itempos, False);
        XmUpdateDisplay(list);
    }
    else
    {
        XmListDeselectPos(list, itempos);
        XmUpdateDisplay(list);
    }
}

//**************************************************************************
//**************************************************************************
// CLASS InvObjectEditor
//**************************************************************************
//**************************************************************************

//=======================================================================
//
// Public constructor - build the widget right now
//
//=======================================================================
InvObjectEditor::InvObjectEditor(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool showName)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)

{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(showName, TRUE);
}

//=========================================================================
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
//
//=========================================================================
InvObjectEditor::InvObjectEditor(Widget parent, const char *name,
                                 SbBool buildInsideParent, SbBool showName,
                                 SbBool buildNow)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(showName, buildNow);
}

//=========================================================================
//
// Called by the constructors
//
// private
//
//=========================================================================
void InvObjectEditor::constructorCommon(SbBool showName, SbBool buildNow)

{

    // init local vars
    setClassName("InvObjectEditor");

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
}

//=========================================================================
//
//    Destructor.
//
InvObjectEditor::~InvObjectEditor()
//
//=========================================================================
{
}

//=========================================================================
//
// Description:
//  	Buils the editor layout
//
// Use: protected
//
//=========================================================================
Widget InvObjectEditor::buildWidget(Widget parent)
{

    int n, i;
    Arg args[12];

    // create a top level form to hold everything together
    n = 0;
    XtSetArg(args[n], XmNfractionBase, 100);
    n++;
    Widget form = XmCreateForm(parent, "", args, n);

    return form;
}

//=========================================================================
//
// redefine those generic virtual functions
//
//=========================================================================
const char *InvObjectEditor::getDefaultWidgetName() const
{
    return "InvObjectEditor";
}

const char *InvObjectEditor::getDefaultTitle() const
{
    return "COVISE Object Editor";
}

const char *InvObjectEditor::getDefaultIconTitle() const
{
    return "Object Editor";
}

//
//=========================================================================
// static callbacks stubs
//=========================================================================
//

//
// called by the color wheel when the color changes
//
/*
void InvObjectEditor::colWheelCB(void *pt, const float hsv[3])
{
    InvObjectEditor *p = (InvObjectEditor *)pt;
    if (p->ignoreCallback)
   return;

    // convert to rgb and update slider and material
    p->baseColor.setHSVValue(hsv);

    p->ignoreCallback = TRUE;
p->colSlider->setBaseColor(hsv);
p->ignoreCallback = FALSE;

p->updateMaterial();
}

//
// called by the XmNvalueChangedCallback. This sets a flag for later use
//
void
InvObjectEditor::fieldChangedCB(Widget, InvObjectEditor *p, void *)
{
p->fieldChanged = TRUE;
}

//
// called whenever the use types a new material name
//
void
InvObjectEditor::nameFieldCB(Widget w, InvObjectEditor *p, void *)
{
if (! p->fieldChanged)
return;
p->fieldChanged = FALSE;

// get the new material name
char *str = XmTextGetString(w);
if (p->materialName != NULL)
free(p->materialName);
p->materialName = (str[0] != '\0') ? strdup(str) : NULL;
XtFree(str);

// make the text field loose the focus
XmProcessTraversal(XtParent(w), XmTRAVERSE_CURRENT);
}

//
// called whenever any of the three sliders (metal, smooth, transp)
// changes values.
//
void
InvObjectEditor::sldWidgetsCB(Widget sld, InvObjectEditor *p, void *)
{
// get the slider new value
int v;
XmScaleGetValue(sld, &v);
float val = v / 1000.0;

//
// now update the material based on which slider changed
//

if (sld == p->sldWidgets[2]) {
// metalness has changed
p->metalness = val;
p->updateMaterial();
}
else if (sld == p->sldWidgets[1]) {
// smothness has changed
p->smoothness = val;
p->updateMaterial();
}
else {
// transparency has changed
p->material->transparency = val;
}
}

//
// called whenever the apply push button gets pressed
//
void
InvObjectEditor::applyCB(Widget, InvObjectEditor *p, void *)
{
p->callbackList.invokeCallbacks((void *)p);

p->saveMaterialFactors();

// save material name
if (p->savedMaterialName != NULL)
free(p->savedMaterialName);
p->savedMaterialName = (p->materialName != NULL) ? strdup(p->materialName) : NULL;
}

//
// called whenever the reset push button gets pressed
//
void
InvObjectEditor::resetCB(Widget, InvObjectEditor *p, void *)
{
//
// reset the material factors
//
p->copyMaterial(p->material, p->savedMaterial);
p->metalness = p->savedMetalness;
p->smoothness = p->savedSmoothness;
p->baseColor = p->savedBaseColor;

p->updateMaterialUI();

//
// reset the material name
//
if (p->materialName != NULL)
free(p->materialName);
p->materialName = (p->savedMaterialName != NULL) ? strdup(p->savedMaterialName) : NULL;
// update text field
if (p->nameField != NULL) {
char *str = (p->materialName != NULL) ? p->materialName : "";
XmTextSetString(p->nameField, str);
}
}
*/

//********************** Test the shit ! **************************

void main(int argc, char *argv[])
{

    Widget myWindow = SoXt::init(argv[0]);

    // InvObjectEditor *editor = new InvObjectEditor(myWindow);
    InvObjList *objectList = new InvObjList(myWindow);

    objectList->addToObjectList("First");
    objectList->addToObjectList("2");
    objectList->addToObjectList("3");
    objectList->addToObjectList("4");
    objectList->addToObjectList("5");

    objectList->show();
    // editor->show();

    SoXt::show(myWindow);
    SoXt::mainLoop();
}
