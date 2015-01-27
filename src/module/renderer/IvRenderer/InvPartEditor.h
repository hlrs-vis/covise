/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_PART_EDITOR_
#define _INV_PART_EDITOR_

//**************************************************************************
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
#include <Xm/ScrolledW.h>

// Open Inventor
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodekits/SoInteractionKit.h>

// pre-declarations
class InvCoviseViewer;

enum Visibility
{
    INVIS = SO_SWITCH_NONE,
    VIS = SO_SWITCH_ALL
};

enum Transform
{
    UNDIS,
    DIS
};

/**
 *  This editor lets you interactively select parts ( previously set by
 *  attribute and stored in a hash-table)
 * after construction a part editor object first
 * allocate space for the parts with allocPartList
 * then insert them into the list with addToPartList.
 * After that the editor's gui can be shown with show()
 */
class InvPartEditor : public SoXtComponent
{

public:
    /** needed to manage parts
       * and the toggle-buttons assigned to them.
       */
    class PartListEntry
    {
    public:
        /// default constructor
        PartListEntry();
        ~PartListEntry();
        Widget visible_;
        Widget referenced_;
        // the name of the part
        char *name_;
        //the key of the part
        int key_;
    };
    // array of part list entries
    PartListEntry *parts_;

    /** allocate space for the parts to manage
       * @param number amount of parts
       */
    void allocPartList(int number);

    /// default constructor
    InvPartEditor(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE);

    ~InvPartEditor();

    /** insert a part into the part list
       * @param name     The name of the part
       * @param key      The key of the part in the hashtables
       */
    void addToPartList(char *name, int key);

    /** Delete the complete list of parts
       */
    void deleteAllItems();

    void setViewer(InvCoviseViewer *);

    /** called whenever a change in the gui occurs.
       * The gui itself is adapted then.
       *
       */
    void apply();

    /** set the number of parts
       * @param number  the new number of parts
       */
    void setNumberOfParts(int number);

    /** get th enumber of parts
       * @return the current number of parts
       */
    int getNumberOfParts();

    /** The part editor's gui can be shown after adding
       * all parts to the part list
       */
    void show();

protected:
    /** This constructor takes a boolean whether to build the widget now.
       * Subclasses can pass FALSE, then call InvObjectEditor::buildWidget()
       * when they are ready for it to be built.
       */
    SoEXTENDER
    InvPartEditor(
        Widget parent,
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

private:
    Widget partname_;
    int lastPart_;
    /** create a toggle button to indicate
       * whether a reference part is used  and a
       * text field for searching it
       * the gui's structure is shown in 'parteditor-flw'
       * or 'parteditor.ps' resp
       * @param parent  the parent widget in which the
       * this part of the gui is made
       */
    void createReferenceButtons(Widget parent);

    /** a RowCol widget including the complete gui
       * the gui's structure is shown in 'parteditor-flw'
       * or 'parteditor.ps' resp
       * @param parent  the parent widget
       */
    void createGuiRowCol(Widget fr1);

    /** create that buttons of the gui
       * used to handle part selection
       * the gui's structure is shown in 'parteditor-flw'
       * or 'parteditor.ps' resp
       * @param parent  the parent widget
       */
    void createPartsButtons(Widget parent);

    /** create a part of the gui
       * the gui's structure is shown in 'parteditor-flw'
       * or 'parteditor.ps' resp
       * @param parent  the parent widget
       */
    void createSelectButtons(Widget parent);

    /** create
       * the gui's structure is shown in 'parteditor-flw'
       * or 'parteditor.ps' resp
       * @param parent  the parent widget
       */
    void createActionButtons(Widget frame);

    /** create a part of the gui
       * the gui's structure is shown in 'parteditor-flw'
       * or 'parteditor.ps' resp
       * @param parent  the parent widget
       */
    void useReferenceButtons(InvPartEditor *p, Bool setValue);

    /** create a part of the gui
       * the gui's structure is shown in 'parteditor-flw'
       * or 'parteditor.ps' resp
       * @param parent  the parent widget
       */
    void createRegFrame(Widget parent);
    /** a callback used when a toggle- or radio button
       *for selecttin parts is pressed
       * @param widget  the widget causing the event
       * @param client_data cleint data is a pointer to the part editor
       * @param call_data  not used here
       */
    static void toggleCB(Widget widget, XtPointer client_data, XtPointer call_data);

    /** this callback is invoked when the user decides to use a reference part
       * by pressing the according toggle button
       * @param widget  the widget causing the event
       * @param client_data cleint data is a pointer to the part editor
       * @param call_data  not used here
       */
    static void useReferenceCB(Widget widget, XtPointer client_data, XtPointer call_data);

    /** callback function called when
       * 'invert' button is pressed
       */
    static void invertByNameCB(Widget widget, XtPointer client_data, XtPointer call_data);

    /** all formerly selected parts matching the regular expression
       * are unselected and vice versa
       * @param regExp regular expression
       */
    void invertByName(char *regExp);

    /** callback function called when
       * 'invisible' button is pressed
       */
    static void unselectByNameCB(Widget widget, XtPointer client_data, XtPointer call_data);
    /** all parts matching the regular expression
       * are unselected
       * @param regExp regular expression
       */
    void unselectByName(char *name);

    /** callback function called when
       * 'visible' button is pressed
       */
    static void selectByNameCB(Widget widget, XtPointer client_data, XtPointer call_data);
    /** all parts matching the regular expression
       * are unselected
       * @param regExp regular expression
       */
    void selectByName(char *name);

    void selectReferenceByName(char *name);
    static void referencePartNameCB(Widget widget, XtPointer client_data, XtPointer call_data);

    /** callback function called when
       * 'Invert selection' button is pressed
       */
    static void invertAllToggleButtonsCB(Widget widget, XtPointer client_data, XtPointer call_data);

    /**
       * method to implement the actual functionality
       * of invertAllToggleButtonsCB
       * @param p      The callback functions are static
       *               so the nonstatic functions called
       *               by them need a pointer to the object.
       */
    void invertAllToggleButtons(InvPartEditor *p);

    /** callback function called when
       * 'All visible' button is pressed
       */
    static void selectAllCB(Widget widget, XtPointer client_data, XtPointer call_data);

    /**
       * method to implement the actual functionality
       * of selectAllCB and unselectAllCB
       * @param p      The callback functions are static
       *               so the nonstatic functions called
       *               by them need a pointer to the object.
       * @param setVal specifies whether to select or unselect all parts
       */
    void selectAllToggleButtons(InvPartEditor *p, Bool setVal);

    /** callback function called when
       * 'All invisible' button is pressed
       */
    static void unselectAllCB(Widget widget, XtPointer client_data, XtPointer call_data);

    SbBool buildNow_;
    int numberOfParts_;
    // tag for visibility
    Visibility switchTag;

    // tag for reference part
    bool referenceTag;

    // tag for transformed scene
    Transform transTag;

    // reference attribute to COVISE viewer
    InvCoviseViewer *covViewer;
};
#endif // _INV_PART_EDITOR_
