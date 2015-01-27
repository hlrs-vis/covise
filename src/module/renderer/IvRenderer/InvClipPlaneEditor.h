/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_CLIP_PLANE_EDITOR_
#define _INV_CLIP_PLANE_EDITOR_

//**************************************************************************
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
//**************************************************************************

#include <covise/covise.h>

// OSF Motif
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
#include <Xm/Frame.h>
#include <Xm/PushBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/Text.h>
#include <Xm/TextF.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/RowColumn.h>

// Open Inventor
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodekits/SoInteractionKit.h>

// pre-declarations
class InvCoviseViewer;

//============================================================================
//
//  Class: InvClipPlaneEditor
//
//  This editor lets you interactively set a clipping plane.
//
//============================================================================
class InvClipPlaneEditor : public SoXtComponent
{

public:
    InvClipPlaneEditor(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE);

    ~InvClipPlaneEditor();

    void setViewer(InvCoviseViewer *);

    // member callbacks
    void equationMemberCB(Widget, XmPushButtonCallbackStruct *);

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call InvObjectEditor::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    InvClipPlaneEditor(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        SbBool buildNow);

    // redefine these generic virtual functions
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    // build routines
    Widget buildWidget(Widget parent);

private:
    // number of text widgets
    enum
    {
        numText = 6
    };

    // Widgets
    Widget form, frame, display, push_button, rowcol, pointLabel, normalLabel;
    Widget text[numText];

    // reference attribute to COVISE viewer
    InvCoviseViewer *covViewer;

    // Point and Normal defining the clipping plane
    double pt[3], nl[3];

    // this is called by both constructors
    void constructorCommon(SbBool buildNow);

    // static callback routines
    static void equationCB(Widget, XtPointer, XmPushButtonCallbackStruct *);
};
#endif // _INV_CLIP_PLANE_EDITOR_
