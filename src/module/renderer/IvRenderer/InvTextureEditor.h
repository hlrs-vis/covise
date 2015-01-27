/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_TEXTURE_EDITOR_
#define _INV_TEXTURE_EDITOR_

/* $Id: InvTextureEditor.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvTextureEditor.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <GL/glx.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/misc/SoCallbackList.h>
#include <Inventor/SbPList.h>

class SoPath;
class SoSensor;
class SoNodeSensor;
class SoTexture2;
class SoTexture2Transform;
class SoTextureCoordinateFunction;
class SoNode;
class SoSeparator;
class SoXtExaminerViewer;
class MyColorWheel;
class MyColorSlider;
class MyThumbWheel;

struct TextureNameStruct;
class MyTextureEditor;

// callback function prototypes
typedef void MyTextureEditorCB(void *userData, MyTextureEditor *editor);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MyTextureEditor
//
//  This editor components lets you interactively specify a texture map to
//  use (from palettes of textures) as well as modify how the texture is
//  applied to the object (translate, rotate, scale as well as different
//  mapping and other functions).
//
//////////////////////////////////////////////////////////////////////////////

class MyTextureEditor : public SoXtComponent
{
public:
    // Pass the home directory of the texture palettes to use.
    // By default the system installed location is used, in addition
    // to looking at the SO_TEXTURE_DIR environment variable.
    //
    MyTextureEditor(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE,
        const char *dir = NULL);
    ~MyTextureEditor();

    //
    // specify the geometry to display as textured map in the editor
    // viewing window.
    //
    void setObjectGeometry(SoNode *geomRoot);

    //
    // set/get the SoTexture2 and the SoTextureCoordinateFunction nodes.
    // Those routines should be used in conjunction to the callbacks to
    // set and get the editor's texture nodes.
    //
    void setTextureNode(const SoTexture2 *txtNode);
    const SoTexture2 *getTextureNode() const
    {
        return texNode;
    }

    void setTransformNode(const SoTexture2Transform *xf);
    const SoTexture2Transform *getTransformNode() const
    {
        return texXfNode;
    }

    void setFunctionNode(const SoTextureCoordinateFunction *func);
    const SoTextureCoordinateFunction *getFunctionNode() const
    {
        return texFuncNode;
    }

    //
    // Callbacks - register functions that will be called whenever the user
    // accepts the current texture settings (press the accept button).
    //
    // Note: the Editor class pointer is passed as the callback data, so
    // the user should use the get methods to access the texture nodes
    //
    void addCallback(
        MyTextureEditorCB *f,
        void *userData = NULL)
    {
        callbackList.addCallback((SoCallbackListCB *)f, userData);
    }

    void removeCallback(
        MyTextureEditorCB *f,
        void *userData = NULL)
    {
        callbackList.removeCallback((SoCallbackListCB *)f, userData);
    }

protected:
    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

private:
    // scene graph and vars
    SoSeparator *sceneRoot;
    SoTexture2 *texNode;
    SoTexture2Transform *texXfNode;
    SoTextureCoordinateFunction *texFuncNode;
    SoNode *userGeometry;
    SbBool repeatState;

    SoCallbackList callbackList;

    void updateTexture2UI();
    void updateTextureXfUI();
    void updateTextureFuncUI();
    void updateTextureFieldAndSlider(int fieldID);
    SbBool setRepeatState(SbBool flag);

    void deselectCurrentItem(SbBool drawHighlight = TRUE);
    void updateTextureName();
    void updateTextureNode();
    void getPaletteNames();
    void loadPaletteItems();
    void savePalette();
    void deleteTextureEntry(int id);
    SbBool addTextureEntry(int id, char *fullName);
    SbBool readScaledImage(char *fileName, int xsize, int ysize,
                           char *buf, int &zsize);
    char *readImage(char *file, int &xsize, int &ysize, int &zsize);
    void switchPalette(int id);
    void updateWindowTitle();
    void createNewPalette(char *palName);

    // component vars and callbacks
    SbBool ignoreCallback;
    MyColorWheel *colWheel;
    MyColorSlider *colSlider;
    SoXtExaminerViewer *viewer;
    MyThumbWheel *scaleXThumb, *scaleYThumb;
    float oldXThumbVal, oldYThumbVal;
    static void colWheelCB(void *, const float hsv[3]);
    static void colSliderCB(void *, float);
    static void scaleXThumbCB(void *, float);
    static void scaleYThumbCB(void *, float);

    // Build routines
    Widget buildWidget(Widget parent);
    Widget buildMenu(Widget parent);
    void buildPaletteSubMenu();
    Widget buildPaletteMenuEntry(long id);
    Widget buildSliders(Widget parent);
    Widget buildButtons(Widget parent);
    Widget buildTexturePaletteWidget(Widget parent);

    // motif vars and callbacks
    Widget *widgetList;
    SbBool fieldChanged;
    static void fieldChangedCB(Widget, MyTextureEditor *, void *);
    static void fieldsCB(Widget, int, void *);
    static void slidersCB(Widget, int, void *);
    static void mappingMenuCB(Widget, int, void *);
    static void optionMenuCB(Widget, int, void *);
    static void acceptCB(Widget, MyTextureEditor *, void *);

    // menu vars and callbacks
    void createNewDialog();
    void createDeleteDialog(const char *title, const char *str1, const char *str2);
    void updateFileMenu();
    static void fileMenuCB(Widget, int id, void *);
    static void paletteMenuCB(Widget w, int num, void *);
    static void newDialogCB(Widget, MyTextureEditor *, XmAnyCallbackStruct *);
    static void deleteDialogCB(Widget, MyTextureEditor *, XmAnyCallbackStruct *);

    // texture palette vars and callbacks
    GLXContext paletteCtx;
    SbBool loadedPalette;
    char *paletteDir;
    Time prevTime;
    int selectedItem, currentItem;
    int curPalette;
    TextureNameStruct *textureNames;
    SbPList paletteList;
    void redrawPalette();
    void drawTextureTile(int id, int style);
    void handleEvent(XAnyEvent *xe);
    static void glxExposeCB(Widget, MyTextureEditor *, void *);
    static void glxInitCB(Widget, MyTextureEditor *, void *);
    static void glxEventCB(Widget, MyTextureEditor *, XAnyEvent *, Boolean *);

    // image dialog vars and callbacks
    GLXContext imageDialogCtx;
    char *dialogImage;
    char *dialogImageName, *dialogImageInfo;
    int dialogImageSize[3];
    short buttonsTotalWidth;
    void openImageDialog();
    void redrawImageDialog();
    void setNewDialogImage(char *fileName = NULL);
    static void imageDialogDestroyCB(Widget, MyTextureEditor *, void *);
    static void imageDialogExposeCB(Widget, MyTextureEditor *, void *);
    static void imageDialogInitCB(Widget, MyTextureEditor *, void *);
    static void imageDialogOpenCB(Widget, MyTextureEditor *, void *);
    static void imageDialogClearCB(Widget, MyTextureEditor *, void *);
    static void imageDialogApplyCB(Widget, MyTextureEditor *, void *);
    static void imageDialogCloseCB(Widget, MyTextureEditor *, void *);
    static void fileDialogOkCB(Widget, MyTextureEditor *, XmFileSelectionBoxCallbackStruct *);
};
#endif // _INV_TEXTURE_EDITOR_
