/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MY_MATERIAL_EDITOR_
#define _MY_MATERIAL_EDITOR_

/* $Id: InvSimpleMaterialEditor.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvSimpleMaterialEditor.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/misc/SoCallbackList.h>
#include <Inventor/SbColor.h>

class SoMaterial;
class SoXtRenderArea;
class MyColorWheel;
class MyColorSlider;
class SoXtClipboard;
class SoPathList;
class MySimpleMaterialEditor;

// callback function prototypes
typedef void MySimpleMaterialEditorCB(void *userData,
                                      MySimpleMaterialEditor *editor);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MySimpleMaterialEditor
//
//  This editor Gizmo lets you interactively edit a material
//
//////////////////////////////////////////////////////////////////////////////

// C-api: prefix=SoXtMtlEd
class MySimpleMaterialEditor : public SoXtComponent
{
public:
    MySimpleMaterialEditor(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE,
        SbBool showMaterialName = FALSE);
    ~MySimpleMaterialEditor();

    // sets/gets the current material
    // C-api: name=setMtl
    void setMaterial(const SoMaterial *mtl);
    // C-api: name=getMtl
    const SoMaterial *getMaterial() const
    {
        return material;
    }

    // specify whether the material name is being displayed
    // as an editable text field. The user can then use the set/get
    // material name methods to display and retreive the name.
    // (default set in constructor to FALSE).
    //
    // C-api: name=isMtlNameVisible
    SbBool isMaterialNameVisible()
    {
        return nameVisible;
    }

    // set/gets the material name which will be displayed (if set to
    // be visible) in the text field.
    // C-api: name=setMtlName
    void setMaterialName(const char *name);
    // C-api: name=getMtlName
    const char *getMaterialName() const
    {
        return materialName;
    }

    // Callbacks - register functions that will be called whenever the user
    // chooses a new material (accept button)
    // C-api: name=addCB
    void addCallback(
        MySimpleMaterialEditorCB *f,
        void *userData = NULL)
    {
        callbackList.addCallback((SoCallbackListCB *)f, userData);
    }

    // C-api: name=removeCB
    void removeCallback(
        MySimpleMaterialEditorCB *f,
        void *userData = NULL)
    {
        callbackList.removeCallback((SoCallbackListCB *)f, userData);
    }

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call MySimpleMaterialEditor::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    MySimpleMaterialEditor(
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
    SoCallbackList callbackList;
    SbBool ignoreCallback;
    SbBool fieldChanged;
    SoMaterial *material, *savedMaterial;
    char *materialName, *savedMaterialName;
    SoXtRenderArea *renderArea;
    MyColorWheel *colWheel;
    MyColorSlider *colSlider;
    Widget sldWidgets[3], nameField;
    float metalness, savedMetalness, smoothness, savedSmoothness;
    SbColor baseColor, savedBaseColor;
    SoXtClipboard *clipboard;
    SbBool nameVisible;

    void copyMaterial(SoMaterial *mat1, const SoMaterial *mat2);
    void updateMaterial();
    void calculateMaterialFactors();
    void updateMaterialUI();
    void saveMaterialFactors();

    static void colWheelCB(void *, const float hsv[3]);
    static void colSliderCB(void *, float v);
    static void nameFieldCB(Widget, MySimpleMaterialEditor *, void *);
    static void fieldChangedCB(Widget, MySimpleMaterialEditor *, void *);
    static void sldWidgetsCB(Widget, MySimpleMaterialEditor *, void *);
    static void applyCB(Widget, MySimpleMaterialEditor *, void *);
    static void resetCB(Widget, MySimpleMaterialEditor *, void *);
    static void copyCB(Widget, MySimpleMaterialEditor *, XmAnyCallbackStruct *);
    static void pasteCB(Widget, MySimpleMaterialEditor *, XmAnyCallbackStruct *);
    static void pasteDone(void *, SoPathList *);

    // this is called by both constructors
    void constructorCommon(SbBool showMaterialName, SbBool buildNow);
};
#endif // _MY_MATERIAL_EDITOR_
