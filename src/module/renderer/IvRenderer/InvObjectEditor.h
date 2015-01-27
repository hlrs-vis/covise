/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_OBJECT_EDITOR_
#define _INV_OBJECT_EDITOR_

/* $Log: InvObjectEditor.h,v $
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

#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/misc/SoCallbackList.h>

class InvObjectEditor;
class InvObjList;

// callback function prototypes
typedef void InvObjListCB(void *userData, InvObjList *list);

//============================================================================
//
//  Class: InvObjList
//
//  Maintain a list of objects
//
//============================================================================
class InvObjList : public SoXtComponent
{
public:
    InvObjList(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE,
        SbBool showMaterialName = FALSE);
    ~InvObjList();

    void addToObjectList(char *name);
    void removeFromObjectList(char *name);
    void updateObjectList(char *name, int isSelection);

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call InvObjectEditor::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    InvObjList(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        SbBool showMaterialName,
        SbBool buildNow);

    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    // build routines
    Widget buildWidget(Widget parent);

private:
    Widget label, list, form, rowcol;

    static void objectListCB(Widget w, InvObjList *, XmListCallbackStruct *);

    // this is called by both constructors
    void constructorCommon(SbBool showMaterialName, SbBool buildNow);
};

// callback function prototypes
typedef void InvObjectEditorCB(void *userData, InvObjectEditor *editor);

//============================================================================
//
//  Class: InvObjectEditor
//
//  This editor  lets you interactively edit object properties
//
//============================================================================

class InvObjectEditor : public SoXtComponent
{
public:
    InvObjectEditor(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE,
        SbBool showMaterialName = FALSE);
    ~InvObjectEditor();

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call InvObjectEditor::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    InvObjectEditor(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        SbBool showMaterialName,
        SbBool buildNow);

    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    // build routines
    Widget buildWidget(Widget parent);

private:
    InvObjList *objectList;
    Widget label, form, list;

    // this is called by both constructors
    void constructorCommon(SbBool showMaterialName, SbBool buildNow);
};
#endif // _INV_OBJECT_EDITOR_
