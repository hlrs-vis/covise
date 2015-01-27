/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description    : Inventor Part Editor
//
// Class(es)      : InvPartEditor
//
// inherited from : SoXtComponent
//
// Author  : Reiner Beller, Christof Schwenzer
//
// History :
//
// **************************************************************************

#include "InvPartEditor.h"
#include "InvCoviseViewer.h"
#include <regex.h>
//=======================================================================
//
// Public constructor - build the widget right now
//
//=======================================================================
InvPartEditor::InvPartEditor(Widget parent,
                             const char *name,
                             SbBool buildInsideParent)
    : SoXtComponent(parent, name, buildInsideParent)
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    buildNow_ = TRUE;
    parts_ = NULL;
}

//=========================================================================
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
//=========================================================================
InvPartEditor::InvPartEditor(Widget parent, const char *name,
                             SbBool buildInsideParent,
                             SbBool buildNow)
    : SoXtComponent(parent, name, buildInsideParent)
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    numberOfParts_ = 0;
    buildNow_ = buildNow;
    parts_ = NULL;
}

InvPartEditor::PartListEntry::PartListEntry()
{
    visible_ = NULL;
    referenced_ = NULL;
    name_ = NULL;
    key_ = -1;
}

InvPartEditor::PartListEntry::~PartListEntry()
{
    if (NULL != name_)
    {
        delete[] name_;
    }
}

//=========================================================================
//
// Called by the constructors
//
// Use: protected
//
//=========================================================================
void InvPartEditor::show()
{

    // init local vars
    setClassName("InvPartEditor");

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow_)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
    switchTag = VIS;
    referenceTag = false;
    transTag = UNDIS;
    SoXtComponent::show();
}

//=========================================================================
//
//    Destructor.
//
//=========================================================================
InvPartEditor::~InvPartEditor()
{
    deleteAllItems();
}

int InvPartEditor::getNumberOfParts()
{
    return numberOfParts_;
}

void InvPartEditor::setNumberOfParts(int number)
{
    numberOfParts_ = number;
}

//=========================================================================
//
// Description: add an item to the part list
//
// Use: public
//
//=========================================================================
void InvPartEditor::addToPartList(char *name, int key)
{
    // appends the item to the end of the list
    parts_[lastPart_].name_ = new char[1 + strlen(name)];
    strcpy(parts_[lastPart_].name_, name);
    parts_[lastPart_].key_ = key;
    lastPart_++;
}

//=========================================================================
void InvPartEditor::allocPartList(int number)
{
    // appends the item to the end of the list
    numberOfParts_ = number;
    parts_ = new PartListEntry[number];
    lastPart_ = 0;
}

void InvPartEditor::useReferenceButtons(InvPartEditor *p, Bool setValue)
{
    int i;
    for (i = 0; i < p->numberOfParts_; i++)
    {
        XtSetSensitive(p->parts_[i].referenced_, setValue);
    }
    p->apply();
}

#if 1
#define showerr(x)
#else

//for testing only
void showerr(int err)
{
    switch (err)
    {
    case REG_BADRPT:
        cerr << "REG_BADRPT:" << endl;
    case REG_BADBR:
        cerr << "REG_BADBR:" << endl;
        break;
    case REG_EBRACE:
        cerr << "REG_EBRACE:" << endl;
        break;
    case REG_EBRACK:
        cerr << "REG_EBRACK:" << endl;
        break;
    case REG_ERANGE:
        cerr << "REG_ERANGE:" << endl;
        break;
    case REG_ECTYPE:
        cerr << "REG_ECTYPE:" << endl;
        break;
    case REG_ECOLLATE:
        cerr << "REG_ECOLLATE:" << endl;
        break;
    case REG_EPAREN:
        cerr << "REG_EPAREN:" << endl;
        break;
    case REG_ESUBREG:
        cerr << "REG_ESUBREG:" << endl;
        break;
    //case REG_EEND: cerr << "REG_EEND:" << endl; break;
    case REG_EESCAPE:
        cerr << "REG_EESCAPE:" << endl;
        break;
    case REG_BADPAT:
        cerr << "REG_BADPAT:" << endl;
        break;
    //case REG_ESIZE: cerr << "REG_ESIZE:" << endl; break;
    case REG_ESPACE:
        cerr << "REG_ESPACE:" << endl;
        break;
    }
}
#endif

void InvPartEditor::invertByName(char *regExp)
{
    regex_t preg;
    int err = regcomp(&preg, regExp, REG_ICASE | REG_NOSUB);
    if (0 != err)
    {
        //obviously the regular expression was erroneous
        showerr(err);
        return;
    }
    int i;
    for (i = 0; i < numberOfParts_; i++)
    {
        if (0 == regexec(&preg, parts_[i].name_, 0, NULL, REG_NOSUB))
        {
            Bool setVal;
            setVal = XmToggleButtonGetState(parts_[i].visible_);
            XmToggleButtonSetState(parts_[i].visible_, !setVal, False);
        }
    }
    apply();
}

void InvPartEditor::unselectByName(char *regExp)
{
    regex_t preg;
    int err = regcomp(&preg, regExp, REG_ICASE | REG_NOSUB);
    if (0 != err)
    {
        //obviously the regular expression was erroneous
        showerr(err);
        return;
    }
    int i;
    for (i = 0; i < numberOfParts_; i++)
    {
        if (0 == regexec(&preg, parts_[i].name_, 0, NULL, REG_NOSUB))
        {
            XmToggleButtonSetState(parts_[i].visible_, False, False);
        }
    }
    apply();
}

void InvPartEditor::selectByName(char *regExp)
{
    regex_t preg;
    int err = regcomp(&preg, regExp, REG_ICASE | REG_NOSUB);
    if (0 != err)
    {
        //obviously the regular expression was erroneous
        showerr(err);
        return;
    }
    int i;
    for (i = 0; i < numberOfParts_; i++)
    {
        if (0 == regexec(&preg, parts_[i].name_, 0, NULL, REG_NOSUB))
        {
            XmToggleButtonSetState(parts_[i].visible_, True, False);
        }
    }
    apply();
}

void InvPartEditor::invertByNameCB(Widget widget, XtPointer client_data, XtPointer call_data)
{

    char *name = XmTextFieldGetString(((InvPartEditor *)client_data)->partname_);
    (void)widget;
    (void)call_data;
    ((InvPartEditor *)client_data)->invertByName(name);
}

void InvPartEditor::unselectByNameCB(Widget widget, XtPointer client_data, XtPointer call_data)
{

    char *name = XmTextFieldGetString(((InvPartEditor *)client_data)->partname_);
    (void)widget;
    (void)call_data;
    ((InvPartEditor *)client_data)->unselectByName(name);
}

void InvPartEditor::selectByNameCB(Widget widget, XtPointer client_data, XtPointer call_data)
{
    char *name = XmTextFieldGetString(((InvPartEditor *)client_data)->partname_);
    (void)widget;
    (void)call_data;
    ((InvPartEditor *)client_data)->selectByName(name);
}

void InvPartEditor::selectReferenceByName(char *name)
{
    regex_t preg;
    int err = regcomp(&preg, name, REG_ICASE | REG_NOSUB);
    if (0 != err)
    {
        //obviously the regular expression was erroneous
        return;
    }
    int i;
    for (i = 0; i < numberOfParts_; i++)
    {
        if (0 == regexec(&preg, parts_[i].name_, 0, NULL, REG_NOSUB))
        {
            XmToggleButtonSetState(parts_[i].referenced_, True, True);
            break;
        }
    }
}

void InvPartEditor::referencePartNameCB(Widget widget, XtPointer client_data, XtPointer call_data)
{

    char *name = XmTextFieldGetString(widget);
    (void)call_data;
    ((InvPartEditor *)client_data)->selectReferenceByName(name);
}

void InvPartEditor::useReferenceCB(Widget widget, XtPointer client_data, XtPointer call_data)
{
    bool setValue;
    Arg args[1];
    (void)call_data;
    XtSetArg(args[0], XmNset, &setValue);
    XtGetValues(widget, args, 1);
    ((InvPartEditor *)client_data)->useReferenceButtons((InvPartEditor *)client_data, setValue);
    ((InvPartEditor *)client_data)->referenceTag = (bool)setValue;
    ((InvPartEditor *)client_data)->apply();
}

void InvPartEditor::invertAllToggleButtons(InvPartEditor *p)
{
    int i;
    for (i = 0; i < p->numberOfParts_; i++)
    {
        XmToggleButtonSetState(p->parts_[i].visible_, !XmToggleButtonGetState(p->parts_[i].visible_), False);
    }
    p->apply();
}

void InvPartEditor::invertAllToggleButtonsCB(Widget widget, XtPointer client_data, XtPointer call_data)
{
    (void)widget;
    (void)call_data;
    ((InvPartEditor *)client_data)->invertAllToggleButtons((InvPartEditor *)client_data);
}

void InvPartEditor::selectAllToggleButtons(InvPartEditor *p, Bool setVal)
{
    Arg arg[1];
    int i;
    XtSetArg(arg[0], XmNset, setVal);
    for (i = 0; i < p->numberOfParts_; i++)
    {
        XtSetValues(p->parts_[i].visible_, arg, 1);
    }
    p->apply();
}

void InvPartEditor::selectAllCB(Widget widget, XtPointer client_data, XtPointer call_data)
{
    (void)widget;
    (void)call_data;
    ((InvPartEditor *)client_data)->selectAllToggleButtons((InvPartEditor *)client_data, True);
}

void InvPartEditor::unselectAllCB(Widget widget, XtPointer client_data, XtPointer call_data)
{
    (void)widget;
    (void)call_data;
    ((InvPartEditor *)client_data)->selectAllToggleButtons((InvPartEditor *)client_data, False);
}

void InvPartEditor::toggleCB(Widget widget, XtPointer client_data, XtPointer call_data)
{
    (void)widget;
    (void)call_data;
    ((InvPartEditor *)client_data)->apply();
}

//=========================================================================
//
// Description: delete all items from the part list
//
// Use: public
//
//=========================================================================
void InvPartEditor::deleteAllItems()
{
    if (NULL != parts_)
    {
        delete[] parts_;
    }
    lastPart_ = 0;
    parts_ = NULL;
}

//=========================================================================
//
// Description:
//  	Buils the editor layout
//
// Use: protected
//
//=========================================================================
//=========================================================================
//
// Description:
//  	Builds the editor layout
//
// Use: protected
// The gui is built in accordance with the widget hierarchy
// sketched in parteditor.flw ( made with kivio ) or
// parteditor.ps
//=========================================================================
Widget InvPartEditor::buildWidget(Widget parent)
{
    Arg args[12];
    // create a top level form to hold everything together
    XtSetArg(args[0], XmNfractionBase, 100);
    Widget form = XmCreateForm(parent, (char *)"", args, 1);
    Widget fr1 = XtVaCreateManagedWidget(
        "Frame1",
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
    createGuiRowCol(fr1);
    return form;
}

//create an XmRowColumns containing
//a scrolled list for the parts in questions
//and a couple of buttons to take some actions
//with the parts whenn pressed
void InvPartEditor::createGuiRowCol(Widget parent)
{
    Arg args[5];
    XtSetArg(args[0], XmNorientation, XmVERTICAL);
    Widget guiRowCol = XtCreateManagedWidget("guiRowCol", xmRowColumnWidgetClass, parent, args, 1);

    XtSetArg(args[0], XmNscrollingPolicy, XmAUTOMATIC);
    XtSetArg(args[1], XmNvisualPolicy, XmCONSTANT);
    XtSetArg(args[2], XmNheight, 300);
    XtSetArg(args[3], XmNwidth, 300);
    Widget scrolled = XmCreateScrolledWindow(guiRowCol, (char *)"Beispiel", args, 4);
    XtManageChild(scrolled);
    createPartsButtons(scrolled);
    createActionButtons(guiRowCol);
}

//create the buttons needed
void InvPartEditor::createActionButtons(Widget parent)
{
    //First the buttons and textfield for selecting parts to be shown/hidden
    Arg args[5];
    XtSetArg(args[0], XmNorientation, XmHORIZONTAL);
    Widget actionButtons = XtCreateManagedWidget("actionbuttons", xmRowColumnWidgetClass, parent, args, 1);

    createSelectButtons(actionButtons);
    createReferenceButtons(actionButtons);
}

//create toggle/readio buttons for choosing/referencing parts.
//For each existing part there is on toggle button
//to change visibility
//and a radio button to specify the reference part

void InvPartEditor::createPartsButtons(Widget parent)
{
    Arg args[5];
    XtSetArg(args[0], XmNorientation, XmHORIZONTAL);
    Widget cols = XtCreateManagedWidget("cols", xmRowColumnWidgetClass, parent, args, 1);
    XtSetArg(args[0], XmNorientation, XmVERTICAL);
    Widget buttons1 = XtCreateManagedWidget("buttons1", xmRowColumnWidgetClass, cols, args, 1);
    XtSetArg(args[1], XmNradioBehavior, True);
    Widget buttons2 = XtCreateManagedWidget("buttons2", xmRowColumnWidgetClass, cols, args, 2);
    // create the lists of radio/togglebuttons
    int i = 0;
    for (i = 0; i < numberOfParts_; i++)
    {
        char buf[4096];
        strcpy(buf, parts_[i].name_);
        strcat(buf, "                                  ");
        parts_[i].visible_ = XmCreateToggleButton(buttons1, buf, NULL, 0);
        parts_[i].referenced_ = XmCreateToggleButton(buttons2, (char *)" ", NULL, 0);
        XtManageChild(parts_[i].visible_);
        XtManageChild(parts_[i].referenced_);
        XtAddCallback(parts_[i].referenced_, XmNvalueChangedCallback, toggleCB, (XtPointer) this);
        XtAddCallback(parts_[i].visible_, XmNvalueChangedCallback, toggleCB, (XtPointer) this);
    }
}

//create buttons
void InvPartEditor::createSelectButtons(Widget parent)
{
    Arg args[5];

    //We want to order the buttons in a column
    XtSetArg(args[0], XmNorientation, XmVERTICAL);
    Widget partSelection = XtCreateManagedWidget("partSelection", xmRowColumnWidgetClass, parent, args, 1);

    //A button to select all part-ids
    Widget selectAll = XmCreatePushButton(partSelection, (char *)"All visible", NULL, 0);
    XtAddCallback(selectAll, XmNactivateCallback, selectAllCB, (XtPointer) this);
    //Probably the user wants to see all parts by default
    //when he starts
    selectAllToggleButtons(this, True);

    //A Button to unselect all part-ids
    Widget unselectAll = XmCreatePushButton(partSelection, (char *)"All invisible", NULL, 0);
    XtAddCallback(unselectAll, XmNactivateCallback, unselectAllCB, (XtPointer) this);

    //A Button to invert the selection
    //after pressing this button formerly unselected parts are selected
    //and formerly selected parts are unselected.
    Widget invert = XmCreatePushButton(partSelection, (char *)"Invert selection", NULL, 0);
    XtAddCallback(invert, XmNactivateCallback, invertAllToggleButtonsCB, (XtPointer) this);

    createRegFrame(partSelection);

    XtManageChild(selectAll);
    XtManageChild(unselectAll);
    XtManageChild(invert);
}

void InvPartEditor::createRegFrame(Widget parent)
{
    Widget regFrame = XtVaCreateManagedWidget(
        "RegFrame",
        xmFrameWidgetClass, parent,
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
    Arg args[5];
    //Now the togglebutton and textfield for the reference part
    XtSetArg(args[0], XmNorientation, XmVERTICAL);
    Widget regCol = XtCreateManagedWidget("parent", xmRowColumnWidgetClass, regFrame, args, 1);

    XmString descStr = XmStringCreateSimple((char *)"Enter regular expression to specify parts");
    XtSetArg(args[0], XmNlabelString, descStr);
    Widget description = XmCreateLabel(regCol, (char *)"description", args, 1);
    Widget visible = XmCreatePushButton(regCol, (char *)"visible", NULL, 0);
    Widget invisible = XmCreatePushButton(regCol, (char *)"invisible", NULL, 0);
    Widget invert = XmCreatePushButton(regCol, (char *)"invert", NULL, 0);
    XtManageChild(description);
    XtManageChild(visible);
    XtManageChild(invisible);
    XtManageChild(invert);
    //A text field to specify  part-ids by a given regular expression
    //since this widget is referenced by several callback
    //it is no local variable but a member of the class.
    partname_ = XmCreateTextField(regCol, (char *)"partname_", NULL, 0);
    XtManageChild(partname_);
    XtAddCallback(invert, XmNactivateCallback, invertByNameCB, (XtPointer) this);
    XtAddCallback(invisible, XmNactivateCallback, unselectByNameCB, (XtPointer) this);
    XtAddCallback(visible, XmNactivateCallback, selectByNameCB, (XtPointer) this);
}

//a radio button to specify whether
//to use a reference part or not
void InvPartEditor::createReferenceButtons(Widget parent)
{
    Arg args[5];
    //Now the togglebutton and textfield for the reference part
    XtSetArg(args[0], XmNorientation, XmVERTICAL);
    Widget referencePartSelection = XtCreateManagedWidget("parent", xmRowColumnWidgetClass, parent, args, 1);
    Widget useReference = XmCreateToggleButton(referencePartSelection, (char *)"Use reference Part", NULL, 0);
    XtAddCallback(useReference, XmNvalueChangedCallback, useReferenceCB, (XtPointer) this);
    useReferenceButtons(this, false);
    XtManageChild(useReference);
    Widget referencedPartName = XmCreateTextField(referencePartSelection, (char *)"referencedPartname", NULL, 0);
    XtAddCallback(referencedPartName, XmNactivateCallback, referencePartNameCB, (XtPointer) this);
    XtManageChild(referencedPartName);
}

//
void InvPartEditor::setViewer(InvCoviseViewer *cov)
{
    covViewer = cov;
}

//=========================================================================
//
// redefine those generic virtual functions
//
//=========================================================================
const char *InvPartEditor::getDefaultWidgetName() const
{
    return "InvPartEditor";
}

const char *InvPartEditor::getDefaultTitle() const
{
    return "COVISE Part Editor";
}

const char *InvPartEditor::getDefaultIconTitle() const
{
    return "Part Editor";
}

//Called whenever something changes
void InvPartEditor::apply()
{
    static int count = 0; // important in case of cooperative working
    int partKey;
    int refPartKey;
    int i;

    for (i = 0; i < numberOfParts_; i++)
    {
        partKey = parts_[i].key_;
        //make the part visible if selected in list
        if (XmToggleButtonGetState(parts_[i].visible_))
        {
            covViewer->switchPart(partKey, VIS);
        }
        else
        {
            covViewer->switchPart(partKey, INVIS);
        }
    }
    if (referenceTag) //do we use reference a reference part?
    {
        for (i = 0; i < numberOfParts_; i++)
        {
            bool setValue;
            Arg args[1];
            XtSetArg(args[0], XmNset, &setValue);
            XtGetValues(parts_[i].referenced_, args, 1);
            if (setValue)
            {
                //A part to be referenced is found.
                break;
            }
        }
        if (i < numberOfParts_)
        {
            //we found really a part to be referenced
            if (transTag == DIS || !count)
            {
                covViewer->resetTransformedScene();
            }
            refPartKey = parts_[i].key_;
            covViewer->setReferencePoint(refPartKey);
            covViewer->transformScene(refPartKey);
            transTag = DIS;
        }
    }
    else
    {
        // undo transformation if necessary
        if (transTag == DIS || !count)
        {
            covViewer->resetTransformedScene();
            transTag = UNDIS;
        }
    }
    count++;
}
