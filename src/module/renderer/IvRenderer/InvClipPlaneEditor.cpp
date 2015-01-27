/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description    : Inventor Clipping Plane Editor
//
// Class(es)      : InvClipPlaneEditor
//
// inherited from : SoXtComponent
//
// Author  : Reiner Beller
//
// History :
//
// **************************************************************************

#include "InvClipPlaneEditor.h"
#include "InvCoviseViewer.h"

//=======================================================================
//
// Public constructor - build the widget right now
//
//=======================================================================
InvClipPlaneEditor::InvClipPlaneEditor(Widget parent,
                                       const char *name,
                                       SbBool buildInsideParent)
    : SoXtComponent(parent, name, buildInsideParent)
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(TRUE);
}

//=========================================================================
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
//=========================================================================
InvClipPlaneEditor::InvClipPlaneEditor(Widget parent, const char *name,
                                       SbBool buildInsideParent,
                                       SbBool buildNow)
    : SoXtComponent(parent, name, buildInsideParent)
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(buildNow);
}

//=========================================================================
//
// Called by the constructors
//
// Use: protected
//
//=========================================================================
void InvClipPlaneEditor::constructorCommon(SbBool buildNow)
{

    // init local vars
    setClassName("InvClipPlaneEditor");

    // Default for point and normal
    pt[0] = 0.0;
    pt[1] = 0.0;
    pt[2] = 0.0;

    nl[0] = 0.0;
    nl[1] = 1.0;
    nl[2] = 0.0;

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
//=========================================================================
InvClipPlaneEditor::~InvClipPlaneEditor()
{
}

//=========================================================================
//
// Description:
//  	Builds the editor layout
//
// Use: protected
//
//=========================================================================
Widget InvClipPlaneEditor::buildWidget(Widget parent)
{
    int n;
    int i;
    const int numRows = 2;
    const int numValues = 3;
    Arg args[12];
    char value[255];

    // create a top level form to hold everything together
    n = 0;
    XtSetArg(args[n], XmNfractionBase, 100);
    n++;
    form = XmCreateForm(parent, (char *)"", args, n);

    push_button = XtVaCreateManagedWidget("Apply",
                                          xmPushButtonWidgetClass, form,
                                          XmNleftAttachment, XmATTACH_FORM,
                                          XmNrightAttachment, XmATTACH_FORM,
                                          XmNbottomAttachment, XmATTACH_FORM,
                                          XmNleftOffset, 5,
                                          XmNrightOffset, 5,
                                          XmNheight, 20,
                                          XmNbottomOffset, 5,
                                          NULL);

    XtAddCallback(push_button, XmNactivateCallback, (XtCallbackProc)InvClipPlaneEditor::equationCB, (XtPointer) this);

    // create frame widget
    frame = XtVaCreateManagedWidget("Frame",
                                    xmFrameWidgetClass, form,
                                    XmNleftAttachment, XmATTACH_FORM,
                                    XmNrightAttachment, XmATTACH_FORM,
                                    XmNleftOffset, 5,
                                    XmNrightOffset, 5,
                                    XmNbottomOffset, 5,
                                    XmNtopOffset, 5,
                                    XmNtopAttachment, XmATTACH_FORM,
                                    XmNbottomAttachment, XmATTACH_WIDGET,
                                    XmNbottomWidget, push_button,
                                    NULL);

    // another form widget
    display = XtVaCreateManagedWidget("Display",
                                      xmFormWidgetClass, frame,
                                      NULL);

    // create row column widget for label and text
    rowcol = XtVaCreateWidget("RowColumn",
                              xmRowColumnWidgetClass, display,
                              XmNshadowType, XmSHADOW_ETCHED_OUT,
                              XmNpacking, XmPACK_COLUMN,
                              XmNorientation, XmHORIZONTAL,
                              XmNnumColumns, numRows,
                              XmNleftAttachment, XmATTACH_FORM,
                              XmNrightAttachment, XmATTACH_FORM,
                              XmNtopAttachment, XmATTACH_FORM,
                              XmNisAligned, TRUE,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

    // Point
    //
    // create label widget for point
    pointLabel = XtVaCreateManagedWidget("PointLabel",
                                         xmLabelWidgetClass, rowcol,
                                         XmNshadowType, XmSHADOW_ETCHED_OUT,
                                         XmNlabelString, XmStringCreateSimple((char *)"Point:"),
                                         NULL);

    // create the  text widgets
    for (i = 0; i < numValues; i++)
    {
        sprintf(value, "%.3f", pt[i]);
        text[i] = XtVaCreateManagedWidget("Text",
                                          xmTextFieldWidgetClass, rowcol,
                                          XmNcolumns, 10,
                                          XmNvalue, value,
                                          NULL);
    }

    // Normal
    //
    // create label widget for normal
    normalLabel = XtVaCreateManagedWidget("NormalLabel",
                                          xmLabelWidgetClass, rowcol,
                                          XmNshadowType, XmSHADOW_ETCHED_OUT,
                                          XmNlabelString, XmStringCreateSimple((char *)"Normal:"),
                                          NULL);

    // create the  text widgets
    for (i = 0; i < numValues; i++)
    {
        sprintf(value, "%.3f", nl[i]);
        text[numValues + i] = XtVaCreateManagedWidget("Text",
                                                      xmTextFieldWidgetClass, rowcol,
                                                      XmNcolumns, 10,
                                                      XmNvalue, value,
                                                      NULL);
    }
    XtManageChild(rowcol);

    return form;
}

//
void InvClipPlaneEditor::setViewer(InvCoviseViewer *cov)
{
    covViewer = cov;
}

//=========================================================================
//
// redefine those generic virtual functions
//
//=========================================================================
const char *InvClipPlaneEditor::getDefaultWidgetName() const
{
    return "InvClipPlaneEditor";
}

const char *InvClipPlaneEditor::getDefaultTitle() const
{
    return "COVISE Clipping Plane Editor";
}

const char *InvClipPlaneEditor::getDefaultIconTitle() const
{
    return "Clipping Plane Editor";
}

//
// called whenever the "Apply" button is pushed
//
inline void InvClipPlaneEditor::equationMemberCB(Widget widget, XmPushButtonCallbackStruct *cbs)
{
    (void)widget;
    (void)cbs;
    covViewer->setClipPlaneEquation(&pt[0], &nl[0]);
}

//
// called whenever the "Apply" button is pushed
//
inline void InvClipPlaneEditor::equationCB(Widget widget, XtPointer client_data, XmPushButtonCallbackStruct *cbs)
{
    char *value;
    int i;

    InvClipPlaneEditor *p = (InvClipPlaneEditor *)client_data;

    for (i = 0; i < numText; i++)
    {
        value = XmTextFieldGetString(p->text[i]);
        if (i < 3)
            p->pt[i] = atof(value);
        else if (i < 6)
            p->nl[i - 3] = atof(value);
    }
    p->equationMemberCB(widget, cbs);
}
