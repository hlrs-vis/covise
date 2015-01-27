/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: InvMaterialPalette.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvMaterialPalette.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#ifndef _SO_XT_MATERIAL_PALETTE_GIZMO_H_
#define _SO_XT_MATERIAL_PALETTE_GIZMO_H_

#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/misc/SoCallbackList.h>
#include <Inventor/SbPList.h>

class SoMaterial;
class SoSwitch;
class SoXtRenderArea;
class SoNode;
class MySimpleMaterialEditor;
class SoAction;
class SoXtInputFocus;
class SoXtClipboard;
class SoTranslation;
class SoPathList;

struct MaterialNameStruct;

// callback function prototypes
typedef void MyMaterialPaletteCB(void *userData, const SoMaterial *mtl);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MyMaterialPalette
//
//
//////////////////////////////////////////////////////////////////////////////

// C-api: prefix=SoXtMtlPalGiz
class MyMaterialPalette : public SoXtComponent
{
public:
    // pass the home directory of the material palettes as dir
    MyMaterialPalette(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE,
        const char *dir = NULL);
    ~MyMaterialPalette();

    // deselect the currently selected item in the palette, if something
    // is selected. This should be called when the thing the palette
    // is affecting doesn't match any of the palette's choices (because
    // the palette is read only and not an editor).
    void deselectCurrentItem();

    // Callbacks - register functions that will be called whenever the user
    // chooses a new material from the palette.
    // (This component cannot be attached to a database - it is read only)
    // C-api: name=addCB
    void addCallback(
        MyMaterialPaletteCB *f,
        void *userData = NULL)
    {
        callbackList.addCallback((SoCallbackListCB *)f, userData);
    }

    // C-api: name=removeCB
    void removeCallback(
        MyMaterialPaletteCB *f,
        void *userData = NULL)
    {
        callbackList.removeCallback((SoCallbackListCB *)f, userData);
    }

    // redefine these to also show/hide material editor
    virtual void show();
    virtual void hide();

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call MyMaterialPalette::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    MyMaterialPalette(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        const char *dir,
        SbBool buildNow);

    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    // Support for menus in the popup planes
    Widget popupWidget;
    virtual void afterRealizeHook();

private:
    char *paletteDir;
    SoCallbackList callbackList;
    SoXtRenderArea *ra;
    Widget *widgetList;
    SbPList paletteList;
    MaterialNameStruct *mtlNames;
    int curPalette;
    SoSwitch *itemSwitch;
    int selectedItem, currentItem;
    MySimpleMaterialEditor *matEditor;
    SoXtInputFocus *focus;
    Time prevTime;
    SoTranslation *overlayTrans1, *overlayTrans2;

    void createSceneGraph();
    void getPaletteNamesAndLoad();
    void loadPaletteItems();
    SoNode *getMaterialFromFile(char *file);
    SbBool handleEvent(XAnyEvent *);
    void updateMaterialName();
    void updateFileMenu();
    void updateEditMenu();
    void findCurrentItem(int x, int y);
    void updateOverlayFeedback();
    void updateWindowTitle();
    void createNewPalette(char *name);
    void savePalette();
    void savePaletteAs(char *name);
    void switchPalette();

    // dialog routines and vars
    SbBool paletteChanged;
    int whatToDoNext, nextPalette;
    void createSaveDialog();
    void createDeleteDialog(const char *title, const char *str1, const char *str2);
    void createPromptDialog(const char *title, const char *str);
    static void saveDialogCB(Widget, MyMaterialPalette *, XmAnyCallbackStruct *);
    static void promptDialogCB(Widget, MyMaterialPalette *, XmAnyCallbackStruct *);
    static void deleteDialogCB(Widget, MyMaterialPalette *, XmAnyCallbackStruct *);

    // component callbacks
    static SbBool raEventCB(void *, XAnyEvent *);
    static void matEditorCB(void *pt, MySimpleMaterialEditor *ed);

    // motif static callbacks
    static void menuCB(Widget, int, XmAnyCallbackStruct *);
    static void paletteMenuCB(Widget, int, void *);

protected:
    // Build routines
    Widget buildWidget(Widget parent);
    Widget buildMenu(Widget parent);
    void buildPaletteSubMenu();
    Widget buildPaletteMenuEntry(long id);

private:
    // cut/copy/paste/delete vars and functions
    SoXtClipboard *clipboard;
    void deleteCurrentMaterial();
    static void pasteDone(void *, SoPathList *);

    // this is called by both constructors
    void constructorCommon(const char *dir, SbBool buildNow);
};
#endif // _INV_XT_MATERIAL_PALETTE_GIZMO_H_
