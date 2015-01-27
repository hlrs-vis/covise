/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    InvAnnotationEditor
//
// Description: Editor for annotation texts
//
// Initial version: 24.01.2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef INVANNOTATIONEDITOR_H
#define INVANNOTATIONEDITOR_H

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
#include <Xm/ScrolledW.h>

// Open Inventor
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodekits/SoInteractionKit.h>

#include <string>
#include "InvAnnotationFlag.h"

// pre-declarations
class InvCoviseViewer;

class InvAnnotationEditor : public SoXtComponent
{
public:
    enum Visibility
    {
        INVIS = SO_SWITCH_NONE,
        VIS = SO_SWITCH_ALL
    };

    /// default CONSTRUCTOR
    InvAnnotationEditor(Widget parent = NULL,
                        const char *name = NULL,
                        SbBool buildInsideParent = TRUE);

    /// DESTRUCTOR
    ~InvAnnotationEditor();

    void setViewer(InvCoviseViewer *v);

    void show();

    void setFlagObject(const InvAnnoFlag *af)
    {
        flag_ = const_cast<InvAnnoFlag *>(af);
    };

protected:
    /** This constructor takes a boolean whether to build the widget now.
       * Subclasses can pass FALSE, then call InvObjectEditor::buildWidget()
       * when they are ready for it to be built.
       */
    SoEXTENDER
    InvAnnotationEditor(Widget parent,
                        const char *name,
                        SbBool buildInsideParent,
                        SbBool buildNow);

    // redefine these generic virtual functions
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    /** build the gui of the part editor
       * @param parent  the parent widget of the editor
       */
    Widget buildWidget(Widget parent);

    std::string getText();

private:
    SbBool buildNow_;

    // tag for visibility
    Visibility switchTag;

    // reference attribute to COVISE viewer
    InvCoviseViewer *viewer_;

    std::string text_;

    Widget mTextField_;

    void createEditLine(Widget parent);

    static void editApplyCB(Widget widget, XtPointer client_data, XtPointer call_data);

    InvAnnoFlag *flag_;
};
#endif
