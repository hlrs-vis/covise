/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvSimpleMtlEditor.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <Xm/Text.h>
#include <Xm/Form.h>
#include <Xm/LabelG.h>
#include <Xm/Scale.h>
#include <Xm/PushBG.h>

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

#include "InvColorWheel.h"
#include "InvColorSlider.h"
#include "InvSimpleMaterialEditor.h"

char *STRDUP(const char *s)
{
    char *s2 = new char[strlen(s) + 1];
    strcpy(s2, s);
    return s2;
}

/*
 * static vars
 */

static const char *geometryBuffer = "\
#Inventor V1.0 ascii\n\
Separator { \
    OrthographicCamera { \
	position 0 0 2 \
	nearDistance 1 \
	farDistance 3 \
	height 2 \
    } \
    LightModel { model BASE_COLOR } \
    BaseColor { rgb [.3 .3 .3, .5 .5 .5, .5 .5 .5, .3 .3 .3] } \
    MaterialBinding { value PER_FACE } \
    Coordinate3 { point [ \
	-3 3 0,  0 3 0,  3 3 0, \
	-3 0 0,  0 0 0,  3 0 0, \
	-3 -3 0, 0 -3 0, 3 -3 0 ] \
    } \
    QuadMesh { verticesPerColumn 3 verticesPerRow 3 } \
    LightModel { model PHONG } \
    DirectionalLight { direction .556 -.623 -.551} \
    DirectionalLight { direction -.556 -.623 -.551} \
    Material {} \
    Complexity { value .8 } \
    Sphere { radius .85 } \
} ";

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
MySimpleMaterialEditor::MySimpleMaterialEditor(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool showName)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(showName, TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
MySimpleMaterialEditor::MySimpleMaterialEditor(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool showName,
    SbBool buildNow)
    : SoXtComponent(
          parent,
          name,
          buildInsideParent)
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(showName, buildNow);
}

////////////////////////////////////////////////////////////////////////
//
// Called by the constructors
//
// private
//
void
MySimpleMaterialEditor::constructorCommon(SbBool showName, SbBool buildNow)
//
//////////////////////////////////////////////////////////////////////
{
    // init local vars
    setClassName("MySimpleMaterialEditor");
    materialName = NULL;
    savedMaterialName = NULL;
    savedMaterial = new SoMaterial;
    savedMaterial->ref();
    ignoreCallback = FALSE;
    clipboard = NULL;
    nameVisible = showName;

    // init widget vars
    nameField = NULL;
    for (int i = 0; i < 3; i++)
        sldWidgets[i] = NULL;

    renderArea = NULL;
    colWheel = NULL;
    colSlider = NULL;

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
}

////////////////////////////////////////////////////////////////////////
//
//    Destructor.
//
MySimpleMaterialEditor::~MySimpleMaterialEditor()
//
////////////////////////////////////////////////////////////////////////
{
    delete renderArea;
    delete colWheel;
    delete colSlider;
    delete clipboard;
    savedMaterial->unref();
    if (materialName != NULL)
        delete[] materialName;
    if (savedMaterialName != NULL)
        delete[] savedMaterialName;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	set the new material
//
// Use: public
void
MySimpleMaterialEditor::setMaterial(const SoMaterial *mtl)
//
////////////////////////////////////////////////////////////////////////
{
    // out with the old...
    copyMaterial(material, mtl);

    // update the screen and save the values
    calculateMaterialFactors();
    updateMaterialUI();
    saveMaterialFactors();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	set the new material name
//
// Use: public
void
MySimpleMaterialEditor::setMaterialName(const char *name)
//
////////////////////////////////////////////////////////////////////////
{
    if (!nameVisible)
        return;

    // out with the old...
    if (materialName != NULL)
        delete[] materialName;
    materialName = (name != NULL) ? STRDUP(name) : NULL;

    // save the name
    if (savedMaterialName != NULL)
        delete[] savedMaterialName;
    savedMaterialName = (materialName != NULL) ? STRDUP(materialName) : NULL;

    // update text field
    if (nameField != NULL)
    {
        char *str = (materialName != NULL) ? materialName : STRDUP("");
        XmTextSetString(nameField, str);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	Buils the editor layout
//
// Use: protected
Widget
MySimpleMaterialEditor::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
#if 0
   // force the window to resize in fixed increments from the
   // window minimum size. This will guarantee that the color wheel
   // has a 1/1 aspect ratio.
   int baseHeight = nameVisible ? 275 : 240;
   if ( XtIsShell(parent) )
      XtVaSetValues(parent,
         XmNbaseWidth, 210,
         XmNbaseHeight, baseHeight,
         XmNminAspectX, 2,
         XmNmaxAspectX, 2,
         XmNminAspectY, 1,
         XmNmaxAspectY, 1,
         NULL);
#endif

    int n, i;
    Arg args[12];

    // create a top level form to hold everything together
    n = 0;
    XtSetArg(args[n], XmNfractionBase, 100);
    n++;
    Widget form = XmCreateForm(parent, (char *)"", args, n);

    // create all the parts
    renderArea = new SoXtRenderArea(form);
    renderArea->setSize(SbVec2s(1, 1)); // col wheel will set the window size
    // spheres are last
    renderArea->setTransparencyType(SoGLRenderAction::BLEND);
    Widget raWidget = renderArea->getWidget();

    colWheel = new MyColorWheel(form);
    colWheel->setSize(SbVec2s(126, 120));
    colWheel->addValueChangedCallback(MySimpleMaterialEditor::colWheelCB, this);
    colWheel->setWYSIWYG(TRUE);
    Widget colWheelWidget = colWheel->getWidget();

    colSlider = new MyColorSlider(form, NULL, TRUE, MyColorSlider::VALUE_SLIDER);
    colSlider->setNumericFieldVisible(FALSE);
    colSlider->setSize(SbVec2s(126, 24));
    colSlider->addValueChangedCallback(MySimpleMaterialEditor::colSliderCB, this);
    colSlider->setWYSIWYG(TRUE);
    Widget colSliderWidget = colSlider->getWidget();

    //
    // read scene graph in
    //
    SoInput in;
    in.setBuffer((void *)geometryBuffer, (size_t)strlen(geometryBuffer));
    SoNode *node;
    SbBool ok = SoDB::read(&in, node);
    if (!ok || node == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MySimpleMaterialEditor::buildWidget",
                           "couldn't read geometry");
#endif
        exit(1);
    }
    renderArea->setSceneGraph(node);

    // search for the material node
    SoSearchAction sa;
    sa.setType(SoMaterial::getClassTypeId());
    sa.apply(node);
    SoFullPath *fullPath = (SoFullPath *)sa.getPath();
    if (fullPath == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("MySimpleMaterialEditor::buildWidget",
                           "couldn't find material node");
#endif
        exit(1);
    }
    material = (SoMaterial *)fullPath->getTail();

    // create the slider labels
    Widget sldLabels[6];
    sldLabels[0] = XmCreateLabelGadget(form, (char *)"opaque", NULL, 0);
    sldLabels[1] = XmCreateLabelGadget(form, (char *)"transp", NULL, 0);
    sldLabels[2] = XmCreateLabelGadget(form, (char *)"rough", NULL, 0);
    sldLabels[3] = XmCreateLabelGadget(form, (char *)"smooth", NULL, 0);
    sldLabels[4] = XmCreateLabelGadget(form, (char *)"plastic", NULL, 0);
    sldLabels[5] = XmCreateLabelGadget(form, (char *)"metal", NULL, 0);

    // create the sliders
    n = 0;
    XtSetArg(args[n], XmNminimum, 0);
    n++;
    XtSetArg(args[n], XmNmaximum, 1000);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNorientation, XmHORIZONTAL);
    n++;
    for (i = 0; i < 3; i++)
    {
        sldWidgets[i] = XmCreateScale(form, (char *)"sld", args, n);
        XtAddCallback(sldWidgets[i], XmNvalueChangedCallback,
                      (XtCallbackProc)MySimpleMaterialEditor::sldWidgetsCB, (XtPointer) this);
        XtAddCallback(sldWidgets[i], XmNdragCallback,
                      (XtCallbackProc)MySimpleMaterialEditor::sldWidgetsCB, (XtPointer) this);
    }

    // create the text field and label
    Widget fieldLabel = 0;
    if (nameVisible)
    {
        fieldLabel = XmCreateLabelGadget(form, (char *)"Name:", NULL, 0);
        char *str = (materialName != NULL) ? materialName : STRDUP("");
        n = 0;
        XtSetArg(args[n], XmNvalue, str);
        n++;
        XtSetArg(args[n], XmNhighlightThickness, 1);
        n++;
        nameField = XmCreateText(form, (char *)"text", args, n);

        fieldChanged = FALSE;
        XtAddCallback(nameField, XmNvalueChangedCallback,
                      (XtCallbackProc)MySimpleMaterialEditor::fieldChangedCB, (XtPointer) this);
        XtAddCallback(nameField, XmNactivateCallback,
                      (XtCallbackProc)MySimpleMaterialEditor::nameFieldCB, (XtPointer) this);
        XtAddCallback(nameField, XmNlosingFocusCallback,
                      (XtCallbackProc)MySimpleMaterialEditor::nameFieldCB, (XtPointer) this);
    }

    // create the push buttons
    Widget buttons[4];
    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    buttons[0] = XmCreatePushButtonGadget(form, (char *)"Apply", args, n);
    buttons[1] = XmCreatePushButtonGadget(form, (char *)"Reset", args, n);
    buttons[2] = XmCreatePushButtonGadget(form, (char *)"Copy", args, n);
    buttons[3] = XmCreatePushButtonGadget(form, (char *)"Paste", args, n);
    XtAddCallback(buttons[0], XmNactivateCallback,
                  (XtCallbackProc)MySimpleMaterialEditor::applyCB, (XtPointer) this);
    XtAddCallback(buttons[1], XmNactivateCallback,
                  (XtCallbackProc)MySimpleMaterialEditor::resetCB, (XtPointer) this);
    XtAddCallback(buttons[2], XmNactivateCallback,
                  (XtCallbackProc)MySimpleMaterialEditor::copyCB, (XtPointer) this);
    XtAddCallback(buttons[3], XmNactivateCallback,
                  (XtCallbackProc)MySimpleMaterialEditor::pasteCB, (XtPointer) this);

    // makes sure things are up to date
    setMaterial(material);

//
// Layout
//
#define SX 3
#define DX 1
    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 7);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;

    XtSetArg(args[n], XmNleftPosition, SX);
    XtSetArg(args[n + 1], XmNrightPosition, 25 - DX);
    XtSetValues(buttons[0], args, n + 2);
    XtSetArg(args[n], XmNleftPosition, 25 + DX);
    XtSetArg(args[n + 1], XmNrightPosition, 50 - DX);
    XtSetValues(buttons[1], args, n + 2);
    XtSetArg(args[n], XmNleftPosition, 50 + DX);
    XtSetArg(args[n + 1], XmNrightPosition, 75 - DX);
    XtSetValues(buttons[2], args, n + 2);
    XtSetArg(args[n], XmNleftPosition, 75 + DX);
    XtSetArg(args[n + 1], XmNrightPosition, 100 - SX);
    XtSetValues(buttons[3], args, n + 2);

    if (nameVisible)
    {
        n = 0;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNleftOffset, 5);
        n++;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNbottomWidget, buttons[0]);
        n++;
        XtSetArg(args[n], XmNbottomOffset, 12);
        n++;
        XtSetValues(fieldLabel, args, n);

        n = 0;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[n], XmNbottomWidget, fieldLabel);
        n++;
        XtSetArg(args[n], XmNbottomOffset, -4);
        n++;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNleftWidget, fieldLabel);
        n++;
        XtSetArg(args[n], XmNleftOffset, 4);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNrightOffset, 5);
        n++;
        XtSetValues(nameField, args, n);
    }

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNleftPosition, 30);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNrightPosition, 70);
    n++;

    XtSetArg(args[n], XmNbottomWidget, nameVisible ? nameField : buttons[0]);
    XtSetArg(args[n + 1], XmNbottomOffset, 10);
    XtSetValues(sldWidgets[0], args, n + 2);
    XtSetArg(args[n], XmNbottomWidget, sldWidgets[0]);
    XtSetArg(args[n + 1], XmNbottomOffset, 6);
    XtSetValues(sldWidgets[1], args, n + 2);
    XtSetArg(args[n], XmNbottomWidget, sldWidgets[1]);
    XtSetValues(sldWidgets[2], args, n + 2);

    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    for (i = 0; i < 3; i++)
    {
        XtSetArg(args[n], XmNrightWidget, sldWidgets[i]);
        XtSetArg(args[n + 1], XmNbottomWidget, sldWidgets[i]);
        XtSetValues(sldLabels[i * 2], args, n + 2);
    }

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    for (i = 0; i < 3; i++)
    {
        XtSetArg(args[n], XmNleftWidget, sldWidgets[i]);
        XtSetArg(args[n + 1], XmNbottomWidget, sldWidgets[i]);
        XtSetValues(sldLabels[i * 2 + 1], args, n + 2);
    }

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, sldWidgets[2]);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 10);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNrightPosition, 50);
    n++;
    XtSetValues(raWidget, args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNleftPosition, 50);
    n++;
    XtSetArg(args[n], XmNleftOffset, 4);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, raWidget);
    n++;
    XtSetValues(colSliderWidget, args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_POSITION);
    n++;
    XtSetArg(args[n], XmNleftPosition, 50);
    n++;
    XtSetArg(args[n], XmNleftOffset, 4);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, colSliderWidget);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetValues(colWheelWidget, args, n);

    // manage those children
    XtManageChildren(buttons, 4);
    if (nameVisible)
    {
        XtManageChild(fieldLabel);
        XtManageChild(nameField);
    }
    XtManageChildren(sldWidgets, 3);
    XtManageChildren(sldLabels, 6);
    XtManageChild(raWidget);
    XtManageChild(colSliderWidget);
    XtManageChild(colWheelWidget);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	copies material2 onto material1
//
// Use: private
void
MySimpleMaterialEditor::copyMaterial(SoMaterial *mat1, const SoMaterial *mat2)
//
////////////////////////////////////////////////////////////////////////
{
    mat1->ambientColor = mat2->ambientColor[0];
    mat1->diffuseColor = mat2->diffuseColor[0];
    mat1->specularColor = mat2->specularColor[0];
    mat1->emissiveColor = mat2->emissiveColor[0];
    mat1->shininess = mat2->shininess[0];
    mat1->transparency = mat2->transparency[0];
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	saves the material factors for later reset.
//
// Use: private
void
MySimpleMaterialEditor::saveMaterialFactors()
//
////////////////////////////////////////////////////////////////////////
{
    copyMaterial(savedMaterial, material);
    savedMetalness = metalness;
    savedSmoothness = smoothness;
    savedBaseColor = baseColor;
}

const float AMB_FACT = 0.25f;
const float SHIN_FACT = 1.2f;

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	update the material node based on the current mat factors.
//
// Use: private
void
MySimpleMaterialEditor::updateMaterial()
//
////////////////////////////////////////////////////////////////////////
{
    float trnsp = material->transparency[0];
    float smooth = smoothness * 0.8;
    float rd = (1.0 - smooth) * (1.0 - trnsp);
    SbVec3f vecOne(1, 1, 1);

    material->diffuseColor = (baseColor * rd * (1.0f - metalness * smooth)).getValue();
    material->ambientColor = AMB_FACT * material->diffuseColor[0];
    material->specularColor = (1.0f - trnsp - rd) * (vecOne + metalness * (baseColor - vecOne));
    material->shininess = powf(smoothness, SHIN_FACT);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	given the current material, figure the best metalness and
//  smoothness factors which represent it.
//
// Use: private
void
MySimpleMaterialEditor::calculateMaterialFactors()
//
////////////////////////////////////////////////////////////////////////
{
    // given that
    //	    shin = smooth ^ SHIN_FACT;
    //
    smoothness = powf(material->shininess[0], 1.0f / SHIN_FACT);

    // make a best guess at what the baseColor could be
    // based on the diffuse color.
    baseColor = material->diffuseColor[0];
    float hsv[3];
    baseColor.getHSVValue(hsv);
    hsv[2] = 1; // scale intensity all the way
    baseColor.setHSVValue(hsv);

    // now find the metalness based on the object specular
    // color, given that
    //	    spec = (1 - trnsp - rd) * (1 + met * (color -1))
    //
    float trnsp = material->transparency[0];
    float smooth = smoothness * 0.8;
    float rd = (1 - smooth) * (1 - trnsp);
    float A = (1 - trnsp - rd);
    if (A != 0)
    {
        // get the metalness by averaging the values found for r,g and b
        int num = 0;
        float met = 0.0;
        SbColor spec = material->specularColor[0];
        for (int i = 0; i < 3; i++)
            if (baseColor[i] != 1 && spec[0] != A)
            {
                met += (spec[0] - A) / (A * (baseColor[i] - 1));
                num++;
            }
        if (num != 0)
            metalness = met / num;
        else
            metalness = 0;
    }
    else
        metalness = 0;

    if (metalness < 0)
        metalness = 0;
    else if (metalness > 1)
        metalness = 1;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	updates the color wheel, color slider and motif sliders to the
//  current material.
//
// Use: private
void
MySimpleMaterialEditor::updateMaterialUI()
//
////////////////////////////////////////////////////////////////////////
{
    // if nothing built, return (they will be updated when built)
    if (sldWidgets[0] == NULL)
        return;

    // update the metal/smooth/transp sliders
    int val = (int)(metalness * 1000);
    XmScaleSetValue(sldWidgets[2], val);
    val = (int)(smoothness * 1000);
    XmScaleSetValue(sldWidgets[1], val);
    val = (int)(material->transparency[0] * 1000);
    XmScaleSetValue(sldWidgets[0], val);

    // update color slider and color wheel
    float hsv[3];
    baseColor.getHSVValue(hsv);
    ignoreCallback = TRUE;
    colWheel->setBaseColor(hsv);
    colSlider->setBaseColor(hsv);
    ignoreCallback = FALSE;
}

//
// redefine those generic virtual functions
//
const char *
MySimpleMaterialEditor::getDefaultWidgetName() const
{
    return "MySimpleMaterialEditor";
}

const char *
MySimpleMaterialEditor::getDefaultTitle() const
{
    return "Simple Material Editor";
}

const char *
MySimpleMaterialEditor::getDefaultIconTitle() const
{
    return "Mat Editor";
}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

//
// called by the color wheel when the color changes
//
void
MySimpleMaterialEditor::colWheelCB(void *pt, const float hsv[3])
{
    MySimpleMaterialEditor *p = (MySimpleMaterialEditor *)pt;
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
// called by color slider when the color changes
//
void
MySimpleMaterialEditor::colSliderCB(void *pt, float)
{
    MySimpleMaterialEditor *p = (MySimpleMaterialEditor *)pt;
    if (p->ignoreCallback)
        return;

    // assign new color, update color wheel and material
    const float *hsv = p->colSlider->getBaseColor();
    p->baseColor.setHSVValue(hsv);

    p->ignoreCallback = TRUE;
    p->colWheel->setBaseColor(hsv);
    p->ignoreCallback = FALSE;

    p->updateMaterial();
}

//
// called by the XmNvalueChangedCallback. This sets a flag for later use
//
void
MySimpleMaterialEditor::fieldChangedCB(Widget, MySimpleMaterialEditor *p, void *)
{
    p->fieldChanged = TRUE;
}

//
// called whenever the use types a new material name
//
void
MySimpleMaterialEditor::nameFieldCB(Widget w, MySimpleMaterialEditor *p, void *)
{
    if (!p->fieldChanged)
        return;
    p->fieldChanged = FALSE;

    // get the new material name
    char *str = XmTextGetString(w);
    if (p->materialName != NULL)
        delete[] p -> materialName;
    p->materialName = (str[0] != '\0') ? STRDUP(str) : NULL;
    XtFree(str);

    // make the text field loose the focus
    XmProcessTraversal(XtParent(w), XmTRAVERSE_CURRENT);
}

//
// called whenever any of the three sliders (metal, smooth, transp)
// changes values.
//
void
MySimpleMaterialEditor::sldWidgetsCB(Widget sld, MySimpleMaterialEditor *p, void *)
{
    // get the slider new value
    int v;
    XmScaleGetValue(sld, &v);
    float val = v / 1000.0;

    //
    // now update the material based on which slider changed
    //

    if (sld == p->sldWidgets[2])
    {
        // metalness has changed
        p->metalness = val;
        p->updateMaterial();
    }
    else if (sld == p->sldWidgets[1])
    {
        // smothness has changed
        p->smoothness = val;
        p->updateMaterial();
    }
    else
    {
        // transparency has changed
        p->material->transparency = val;
    }
}

//
// called whenever the apply push button gets pressed
//
void
MySimpleMaterialEditor::applyCB(Widget, MySimpleMaterialEditor *p, void *)
{
    p->callbackList.invokeCallbacks((void *)p);

    p->saveMaterialFactors();

    // save material name
    if (p->savedMaterialName != NULL)
        delete[] p -> savedMaterialName;
    p->savedMaterialName = (p->materialName != NULL) ? STRDUP(p->materialName) : NULL;
}

//
// called whenever the reset push button gets pressed
//
void
MySimpleMaterialEditor::resetCB(Widget, MySimpleMaterialEditor *p, void *)
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
        delete[] p -> materialName;
    p->materialName = (p->savedMaterialName != NULL) ? STRDUP(p->savedMaterialName) : NULL;
    // update text field
    if (p->nameField != NULL)
    {
        char *str = (p->materialName != NULL) ? p->materialName : STRDUP("");
        XmTextSetString(p->nameField, str);
    }
}

//
// called whenever the copy push button gets pressed
//
void
MySimpleMaterialEditor::copyCB(Widget, MySimpleMaterialEditor *p, XmAnyCallbackStruct *cb)
{
    if (p->clipboard == NULL)
        p->clipboard = new SoXtClipboard(p->getWidget());

    //
    // copy the material and also copy the current color (from the
    // wheel) to enable pasting into the color editor.
    //

    // construct a path list which has the Material and BaseColor nodes
    SoBaseColor *color = new SoBaseColor;
    color->rgb.setValue(p->baseColor);
    SoPathList *pathList = new SoPathList(2);
    SoPath *path1 = new SoPath(p->material);
    SoPath *path2 = new SoPath(color);
    path1->ref();
    path2->ref();
    pathList->append(path1);
    pathList->append(path2);

    Time eventTime = cb->event->xbutton.time;
    p->clipboard->copy(pathList, eventTime);

    // delete the paths and the list    color->unref();
    path1->unref();
    path2->unref();
    delete pathList;
}

//
// called whenever the paste push button gets pressed
//
void
MySimpleMaterialEditor::pasteCB(Widget, MySimpleMaterialEditor *p, XmAnyCallbackStruct *cb)
{
    if (p->clipboard == NULL)
        p->clipboard = new SoXtClipboard(p->getWidget());

    Time eventTime = cb->event->xbutton.time;
    p->clipboard->paste(eventTime, MySimpleMaterialEditor::pasteDone, p);
}

//
// called whenever the X server is done doing the paste
//
void
MySimpleMaterialEditor::pasteDone(void *pt, SoPathList *pathList)
{
    MySimpleMaterialEditor *p = (MySimpleMaterialEditor *)pt;

    SoSearchAction sa;
    SoFullPath *fullPath = NULL;

    //
    // search for first material in that pasted scene
    //
    sa.setType(SoMaterial::getClassTypeId());
    for (int i = 0; i < pathList->getLength(); i++)
    {
        sa.apply((*pathList)[i]);
        if ((fullPath = (SoFullPath *)sa.getPath()) != NULL)
        {

            // assign new material
            p->copyMaterial(p->material, (SoMaterial *)fullPath->getTail());
            p->calculateMaterialFactors();
            p->updateMaterialUI();

            break;
        }
    }

    //
    // else search for the first base color in the scene, which will
    // be used for the color wheel.
    //
    if (fullPath == NULL)
    {
        sa.setType(SoBaseColor::getClassTypeId());
        for (int i = 0; i < pathList->getLength(); i++)
        {
            sa.apply((*pathList)[i]);
            if ((fullPath = (SoFullPath *)sa.getPath()) != NULL)
            {

                // assign new color, update color UI + material
                float hsv[3];
                SoBaseColor *node = (SoBaseColor *)fullPath->getTail();
                p->baseColor = node->rgb[0];
                p->baseColor.getHSVValue(hsv);

                p->ignoreCallback = TRUE;
                p->colSlider->setBaseColor(hsv);
                p->colWheel->setBaseColor(hsv);
                p->ignoreCallback = FALSE;

                p->updateMaterial();

                break;
            }
        }
    }

    // ??? We delete the callback data when done with it.
    delete pathList;
}
