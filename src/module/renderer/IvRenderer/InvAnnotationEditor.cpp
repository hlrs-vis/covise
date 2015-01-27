/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class InvAnnotationEditor             ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:  24.01.2002 (initial version)                                 ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "InvAnnotationEditor.h"
#include "InvAnnotationManager.h"

InvAnnotationEditor::InvAnnotationEditor(Widget parent,
                                         const char *name,
                                         SbBool buildInsideParent)
    :

    SoXtComponent(parent, name, buildInsideParent)
    , buildNow_(TRUE)
    , flag_(NULL)
{
}

InvAnnotationEditor::InvAnnotationEditor(Widget parent,
                                         const char *name,
                                         SbBool buildInsideParent,
                                         SbBool buildNow)
    :

    SoXtComponent(parent, name, buildInsideParent)
    , buildNow_(buildNow)
    , flag_(NULL)
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
Widget
InvAnnotationEditor::buildWidget(Widget parent)
{
    Arg args[12];
    // create a top level form to hold everything together
    XtSetArg(args[0], XmNfractionBase, 100);
    Widget form = XmCreateForm(parent, (char *)"", args, 1);
    Widget fr1 = XtVaCreateManagedWidget(
        (char *)"Frame1",
        xmFrameWidgetClass, form,
        XmNleftAttachment, XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNleftOffset, 5,
        XmNrightOffset, 5,
        XmNbottomOffset, 5,
        XmNtopOffset, 5,
        XmNtopAttachment, XmATTACH_FORM,
        XmNbottomAttachment, XmATTACH_WIDGET,
        //XmNbottomWidget,      push_button,
        NULL);

    createEditLine(fr1);

    return form;
}

// redefine these generic virtual functions
const char *
InvAnnotationEditor::getDefaultWidgetName() const
{
    return "COVISE Annotation Editor";
}

const char *
InvAnnotationEditor::getDefaultTitle() const
{
    return "COVISE Annotation Editor";
}

const char *
InvAnnotationEditor::getDefaultIconTitle() const
{
    return "COVISE Annotation Editor";
}

void
InvAnnotationEditor::setViewer(InvCoviseViewer *v)
{
    viewer_ = v;
}

void
InvAnnotationEditor::show()
{
    // init local vars
    setClassName("InvAnnotationEditor");

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow_)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
    switchTag = VIS;
    SoXtComponent::show();
}

//
// create a LABEL + Edit line + an apply button
//
void
InvAnnotationEditor::createEditLine(Widget parent)
{
    Arg args[5];
    int i(0);

    Widget rc = XmCreateRowColumn(parent, (char *)"rc", NULL, 0);
    //    XtSetArg(args[i], XmNlabelString, descStr); i++;

    i = 0;
    XmString descStr = XmStringCreateSimple((char *)"enter annotation text:");
    XtSetArg(args[i], XmNlabelString, descStr);
    i++;
    Widget description = XmCreateLabel(rc, (char *)"description", args, 1);

    mTextField_ = XmCreateTextField(rc, (char *)"tf", NULL, 0);

    Widget applyBut = XmCreatePushButton(rc, (char *)"Apply", NULL, 0);

    XtAddCallback(applyBut, XmNactivateCallback, editApplyCB, (XtPointer) this);

    // manage all
    XtManageChild(rc);
    XtManageChild(description);
    XtManageChild(mTextField_);
    XtManageChild(applyBut);
}

void
InvAnnotationEditor::editApplyCB(Widget widget, XtPointer client_data, XtPointer call_data)
{

    InvAnnotationEditor *me = static_cast<InvAnnotationEditor *>(client_data);

    char *text = XmTextFieldGetString(me->mTextField_);
    (void)widget;
    (void)call_data;

    if (me->flag_)
    {
        me->flag_->clearText();
        me->flag_->setText(text);
        Annotations->sendParameterData();
    }
    me->hide();
}

std::string
InvAnnotationEditor::getText()
{
    return text_;
}

//
// Destructor
//
InvAnnotationEditor::~InvAnnotationEditor()
{
}
