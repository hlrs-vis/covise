/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>
#else
#include <unistd.h> // for access
#include <sys/param.h> // for MAXPATHLEN
#include <sys/time.h>
#include <sys/time.h>
#endif

#include <util/coTypes.h>
#include <util/unixcompat.h>

#include <qevent.h>
#include <qnamespace.h>
#include <QListWidget>
#include <QListView>
#include <qwidget.h>
#include <QMouseEvent>
#include <QKeyEvent>

//
// Inventor stuff
//
#include <Inventor/SoDB.h>
#include <Inventor/SoNodeKitPath.h>
#include <Inventor/SoPickedPoint.h>
#include <Inventor/SoOffscreenRenderer.h>
#include <Inventor/Qt/SoQt.h>
#include <Inventor/Qt/editors/SoQtColorEditor.h>

#ifdef HAVE_EDITORS
#include <Inventor/Qt/editors/SoQtMaterialEditor.h> // for MAXPATHLEN
#endif

#include <Inventor/Qt/nodes/SoGuiColorEditor.h>
#include <Inventor/actions/SoBoxHighlightRenderAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/details/SoNodeKitDetail.h>
#include <Inventor/draggers/SoDirectionalLightDragger.h>
#include <Inventor/draggers/SoTabBoxDragger.h>
#include <Inventor/draggers/SoCenterballDragger.h>
#include <Inventor/draggers/SoJackDragger.h>
#include <Inventor/manips/SoCenterballManip.h>
#include <Inventor/manips/SoDirectionalLightManip.h>
#include <Inventor/manips/SoHandleBoxManip.h>
#include <Inventor/manips/SoJackManip.h>
#include <Inventor/manips/SoPointLightManip.h>
#include <Inventor/manips/SoSpotLightManip.h>
#include <Inventor/manips/SoTabBoxManip.h>
#include <Inventor/manips/SoTrackballManip.h>
#include <Inventor/manips/SoTransformManip.h>
#include <Inventor/manips/SoTransformBoxManip.h>
#include <Inventor/nodekits/SoBaseKit.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoEnvironment.h>
#include <Inventor/nodes/SoLabel.h>
#include <Inventor/SoLists.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/nodes/SoLight.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPointLight.h>
#include <Inventor/nodes/SoSelection.h>
#include <Inventor/nodes/SoShape.h>
#include <Inventor/nodes/SoSpotLight.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/nodes/SoClipPlane.h>
#include <Inventor/engines/SoTransformVec3f.h>
#include <Inventor/engines/SoComposeMatrix.h>
#include <Inventor/nodes/SoEventCallback.h>

#include <Inventor/events/SoEvent.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <Inventor/events/SoMotion3Event.h>

#include <util/coStringTable.h>
#include <config/CoviseConfig.h>

#ifdef YAC
#include <util/coErr.h>
#endif

// renderer stuff
//
#include "InvViewer.h"
#include "InvDefs.h"
#include "InvTelePointer.h"
#include "InvObjectList.h"
#include "InvError.h"
#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif
#ifndef YAC
#include "InvCommunicator.h"
#endif
#ifndef YAC
#include "InvObjectManager.h"
#else
#include "InvObjectManager_yac.h"
#endif
#include "InvManipList.h"
#include "InvPlaneMover.h"
#include "InvClipPlaneEditor.h"
#ifndef YAC
#include "InvObjects.h"
#else
#include "InvObjects_yac.h"
#endif
#include "CoviseWindowCapture.h"
#include "InvComposePlane.h"

int InvViewer::isSelected = 0;
int InvViewer::c_first_time = 0;

InvViewer *coviseViewer = NULL;

//
//
// axis data

static const char *axis = "Separator {\n"
                          "    LightModel { model BASE_COLOR }\n"
                          "    MaterialBinding { value PER_FACE }\n"
                          "    DrawStyle { lineWidth 2 }\n"
                          "    Coordinate3 { point [0 0 0, 1 0 0, 0 1 0, 0 0 1] }\n"
                          "    BaseColor { rgb [1 0 0, 0 1 0, 0 0 1 ] }\n"
                          "    IndexedLineSet {\n"
                          "          coordIndex [0, 1, -1, 0, 2, -1, 0, 3] }\n"
                          "    Separator {\n"
                          "        BaseColor{ rgb 1 0 0  }\n"
                          "        Translation { translation 1 0 0 }\n"
                          "        RotationXYZ { axis Z angle -1.570796327 }\n"
                          "        Cone { bottomRadius .05 height .15 }\n"
                          "        Translation { translation 0 .15 0 }\n"
                          "        Text2 { string "
                          "X"
                          " }\n"
                          "    }\n"
                          "    Separator {\n"
                          "        BaseColor{ rgb 0 1 0 }\n"
                          "        Translation { translation 0 1 0 }\n"
                          "        Cone { bottomRadius .05 height .15 }\n"
                          "        Translation { translation 0 .15 0 }\n"
                          "        Text2 { string "
                          "Y"
                          " }\n"
                          "    }\n"
                          "    Separator {\n"
                          "        BaseColor{ rgb 0 0 1}\n"
                          "        Translation { translation 0 0 1 }\n"
                          "        RotationXYZ { axis X angle 1.570796327 }\n"
                          "        Cone { bottomRadius .05 height .15 }\n"
                          "        Translation { translation 0 .15 0 }\n"
                          "        Text2 { string "
                          "Z"
                          " }\n"
                          "    }\n"
                          "}\n";

bool InvViewer::m_bRenderWindowCapture = false;
bool InvViewer::m_bFramePerSecondOutputConsole = false;
CCoviseWindowCapture *InvViewer::m_pCapture = NULL;

// creates a part of a scene-graph which is a plane suitable for the decoration
// of our SoJackDragger
SoGroup *makePlane()
{

    static float vertexPos[4][3] = {
        { 0, 0.5, -0.5 },
        { 0, 0.5, 0.5 },
        { 0, -0.5, 0.5 },
        { 0, -0.5, -0.5 }
    };

    static int indices[4] = { 3, 2, 1, 0 };

    SoGroup *plane = new SoGroup;
    plane->ref();

    SoMaterial *mat = new SoMaterial;

    mat->ambientColor.setValue(0.3f, 0.1f, 0.1f);
    mat->diffuseColor.setValue(0.8f, 0.7f, 0.2f);
    mat->specularColor.setValue(0.4f, 0.3f, 0.1f);
    mat->transparency = 0.3f;
    plane->addChild(mat);

    SoMaterialBinding *bndng = new SoMaterialBinding;
    bndng->value = SoMaterialBinding::DEFAULT;
    plane->addChild(bndng);

    SoCoordinate3 *coords = new SoCoordinate3;
    coords->point.setValues(0, 4, vertexPos);

    plane->addChild(coords);

    SoIndexedFaceSet *faceSet = new SoIndexedFaceSet;
    faceSet->coordIndex.setValues(0, 4, indices);

    plane->addChild(faceSet);

    return plane;
}

// creates a part of a scene-graph which is an arrow suitable for the decoration
// of our SoJackDragger
SoGroup *makeArrow()
{
    static float vertexPos[2][3] = {
        { 0.0f, 0.0f, 0.0f },
        { 0.0f, -0.7f, 0.0f }
    };

    SoGroup *arrow = new SoGroup;

    SoLineSet *line = new SoLineSet;

    SoCoordinate3 *coords = new SoCoordinate3;
    coords->point.setValues(0, 2, vertexPos);

    arrow->addChild(coords);

    arrow->addChild(line);

    SoTranslation *transl = new SoTranslation;
    transl->translation.setValue(0.0f, -0.85f, 0.0f);
    arrow->addChild(transl);

    SoRotationXYZ *rot = new SoRotationXYZ;
    rot->angle = (float)M_PI;
    rot->axis = SoRotationXYZ::Z;
    arrow->addChild(rot);

    SoCone *tip = new SoCone;
    tip->bottomRadius = 0.1f;
    tip->height = 0.3f;

    arrow->addChild(tip);

    return arrow;
}

void
decorateJackDragger(SoJackDragger *jack)
{
    SoDrawStyle *handleDrawStyle_ = new SoDrawStyle;
    handleDrawStyle_->style.setValue(SoDrawStyle::FILLED);

    SoSeparator *plane[6];
    for (int ii = 0; ii < 6; ii++)
    {
        plane[ii] = new SoSeparator;
        plane[ii]->addChild(handleDrawStyle_);
        plane[ii]->addChild(makePlane());
    }

    SoSeparator *empty[6];
    for (int ii = 0; ii < 6; ii++)
    {
        empty[ii] = new SoSeparator;
        empty[ii]->addChild(handleDrawStyle_);
    }

    SoSeparator *scale[2];
    for (int ii = 0; ii < 2; ii++)
    {
        scale[ii] = new SoSeparator;
        scale[ii]->addChild(handleDrawStyle_);
        //scale[ii]->addChild(makeArrow());
    }

    SoSeparator *arrow[2];
    for (int ii = 0; ii < 2; ii++)
    {
        arrow[ii] = new SoSeparator;
        arrow[ii]->addChild(handleDrawStyle_);
        arrow[ii]->addChild(makeArrow());
    }

    jack->setPart("rotator.rotator", arrow[0]);
    jack->setPart("rotator.rotatorActive", arrow[1]);

    jack->setPart("translator.yTranslator.translator", plane[0]);
    jack->setPart("translator.xTranslator.translator", plane[2]);
    jack->setPart("translator.zTranslator.translator", plane[4]);
    jack->setPart("translator.yTranslator.translatorActive", plane[1]);
    jack->setPart("translator.xTranslator.translatorActive", plane[3]);
    jack->setPart("translator.zTranslator.translatorActive", plane[5]);

    jack->setPart("translator.yzTranslator.translatorActive", empty[0]);
    jack->setPart("translator.xzTranslator.translatorActive", empty[1]);
    jack->setPart("translator.xyTranslator.translatorActive", empty[2]);
    jack->setPart("translator.yzTranslator.translator", empty[3]);
    jack->setPart("translator.xzTranslator.translator", empty[4]);
    jack->setPart("translator.xyTranslator.translator", empty[5]);
}

void
myMousePressCB(void *userData, SoEventCallback *eventCB)
{
    (void)userData;
    //SoSelection *selection = (SoSelection *) userData;
    const SoEvent *event = eventCB->getEvent();
    const SoPickedPoint *point = eventCB->getPickedPoint();
    if (point)
    {
        SoPath *path = point->getPath();
        if (path)
            fprintf(stderr, "pickfilter: %s\n", path->getTail()->getName().getString());
    }

    // Check for the Up and Down arrow keys being pressed.
    if (SO_MOUSE_PRESS_EVENT(event, BUTTON1))
    {
        fprintf(stderr, "mouse press event\n");
    }
}

//======================================================================
//
// Description:
//	Constructor for the Renderer.
//      Creates the Topbar menu
//
// Use: public
//======================================================================
InvViewer::InvViewer(QWidget *parent, const char *name)
    : SoQtExaminerViewer(parent, name)
{
    coviseViewer = this;

    tpShow_ = false;
    keyState = -1;
    mouseX = 0;
    mouseY = 0;
    m_current_tswitch = NULL;

    // viewer edit state = CO_OFF at default
    viewer_edit_state = CO_OFF;

    // make top node for scene graph
    sceneGraph = new SoSeparator();
    sceneGraph->ref();

    setSceneGraph(sceneGraph);

    // An event callback node so we can receive key press events
    SoEventCallback *myEventCB = new SoEventCallback;
    myEventCB->addEventCallback(
        SoMouseButtonEvent::getClassTypeId(),
        myMousePressCB, sceneGraph);
    sceneGraph->addChild(myEventCB);

    permanentSelection = new SoGroup;
    permanentSelection->ref();

    //
    // add axis
    int axStatus;
    if (renderer->renderConfig->isOn("Renderer.ShowAxis", "true"))
        axStatus = CO_ON;
    else
        axStatus = CO_OFF;

    axis_switch = new SoSwitch;
    axis_switch->whichChild.setValue(0);
    axis_switch->addChild(makeAxis());
    sceneGraph->addChild(axis_switch);
    setAxis(axStatus);

    text_manager = new InvTextManager();
    sceneGraph->addChild(text_manager->getRootNode());

    // add Telepointer
    tpHandler = new TPHandler();
    sceneGraph->addChild(tpHandler->getRoot());

    // add start and finish edit callback
    addStartCallback(InvViewer::viewerStartEditCB, NULL);
    addFinishCallback(InvViewer::viewerFinishEditCB, NULL);

    //
    // Editors
    //
    ignoreCallback = FALSE;

#ifdef HAVE_EDITORS
    materialEditor = NULL;
    colorEditor = NULL;
#endif
    backgroundColorEditor = NULL;
    //transformSliderSet = NULL;
    //partEditor_ = NULL;
    //annoEditor_ = NULL;
    clippingPlaneEditor = NULL;

    //
    // Manips
    //
    curManip = SV_NONE;
    curManipReplaces = true;
    maniplist = new InvManipList;

    // current transform node to watch for data change
    currTransformNode = NULL;

    // List of transform nodes related to the current transform node
    transformNode = NULL;
    //transformSensor = new SoNodeSensor( InvViewer::transformCallback, this);

    // build a object list
    list = new InvObjectList();

    m_iBillboardRenderingMethod = 0;
    m_bBillboardRenderingBlending = true;
    m_bRenderWindowCapture = false;

    m_bFramePerSecondOutputConsole = false;

    m_pCapture = new CCoviseWindowCapture;
    m_pCapture->Init();

    // add node for COVISE objects
    selection = new SoSelection();
    selection->setName("Selection");

    sceneGraph->addChild(selection);
    selection->addSelectionCallback(InvViewer::selectionCallback, this);
    selection->addDeselectionCallback(InvViewer::deselectionCallback, this);
    selection->setPickFilterCallback(InvViewer::pickFilterCB, this);
    selectionCallbackInactive = false;
    selection->addChild(permanentSelection);

    // clipping
    clipSwitch = new SoSwitch;
    //sceneGraph->addChild(clipSwitch);

    clipState = CO_OFF;
    clipPlane = new SoClipPlane;
    clipPlane->on.setValue(clipState);
    permanentSelection->addChild(clipPlane);

    clipDragger = new SoJackDragger;
    //decorateJackDragger(clipDragger);

    SoComposeMatrix *composeMat = new SoComposeMatrix;
    composeMat->rotation.connectFrom(&clipDragger->rotation);

    SoTransformVec3f *transformVec = new SoTransformVec3f;
    transformVec->vector.setValue(SbVec3f(1.0, 0.0, 0.0));
    transformVec->matrix.connectFrom(&composeMat->matrix);
    InvComposePlane *composePlane = new InvComposePlane;
    composePlane->normal.connectFrom(&transformVec->normalDirection);
    composePlane->point.connectFrom(&clipDragger->translation);
    clipPlane->plane.connectFrom(&composePlane->plane);
    clipSwitch->addChild(clipDragger);
    setClipping(clipState);

    highlightRA = new SoBoxHighlightRenderAction;
    highlightRA->setVisible(true);
    setGLRenderAction(highlightRA);
    redrawOnSelectionChange(selection);

    snapshotCallback = new SoCallback;
    snapshotCallback->setCallback(InvViewer::snapshotCB, this);
    sceneGraph->addChild(snapshotCallback);

    // add a plane for moving cutting planes
    pm_ = new InvPlaneMover();
    sceneGraph->addChild(pm_->getSeparator());

    // make blended transparency the default
    this->setTransparencyType(SoGLRenderAction::SORTED_OBJECT_BLEND);

    rightWheelControlsVolumeSampling = false;
    volumeSamplingAccuracy = 2.f;
    savedDollyValue = getRightWheelValue(); // member variable needs default value!
    enableRightWheelSampleControl(rightWheelControlsVolumeSampling);

    globalLutUpdated = false;
    globalLut = NULL;
    numGlobalLutEntries = 0;
}

//======================================================================
//
// Description:
//    Destructor.
//
// Use: public
//======================================================================
InvViewer::~InvViewer()
{
    delete text_manager;

    // detach and delete the manips
    detachManipFromAll();
    delete maniplist;

    // detach and delete the viewers
    setSceneGraph(NULL);
    sceneGraph->unref();

// Editor components
#ifdef HAVE_EDITORS
    delete materialEditor;
//delete colorEditor;
#endif
    delete backgroundColorEditor;

    //delete transformSliderSet;
    //delete ambientColorEditor;
    //delete partEditor_;
    //delete annoEditor_;*/
    delete clippingPlaneEditor;

    if (m_pCapture != NULL)
    {
        delete m_pCapture;
        m_pCapture = NULL;
    }

    delete[] globalLut;
}

void InvViewer::rightWheelMotion(float value)
{
    if (rightWheelControlsVolumeSampling)
    {
        if (value < 0.01)
        {
            value = 0.01f;
            setRightWheelValue(value);
        }
        volumeSamplingAccuracy = value;

        char buf[1024];
        sprintf(buf, "Volume sampling: %.2f", value);
        setRightWheelString(buf);
    }
    else
    {
        SoQtExaminerViewer::rightWheelMotion(value);
    }
}

void InvViewer::rightWheelFinish()
{
    if (rightWheelControlsVolumeSampling)
    {
        render();
    }
    else
    {
        SoQtExaminerViewer::rightWheelFinish();
    }
}

void InvViewer::enableRightWheelSampleControl(bool state)
{
    rightWheelControlsVolumeSampling = state;

    if (state)
    {
        savedDollyValue = getRightWheelValue();
        setRightWheelValue(volumeSamplingAccuracy);
        char buf[1024];
        sprintf(buf, "Volume sampling: %.2f", volumeSamplingAccuracy);
        setRightWheelString(buf);
    }
    else
    {
        setRightWheelValue(savedDollyValue);
        setRightWheelString("Dolly");
    }
}

float InvViewer::getVolumeSamplingAccuracy()
{
    return volumeSamplingAccuracy;
}

const uchar *InvViewer::getGlobalLut()
{
    return globalLut;
}

int InvViewer::getNumGlobalLutEntries()
{
    return numGlobalLutEntries;
}

void InvViewer::setGlobalLut(int numEntries, const uchar *lut)
{
    delete[] globalLut;
    globalLut = new uchar[4 * numEntries];
    memcpy(globalLut, lut, 4 * numEntries);
    numGlobalLutEntries = numEntries;
    globalLutUpdated = true;
    render();
}

bool InvViewer::isGlobalLutUpdated()
{
    return globalLutUpdated;
}

void InvViewer::render()
{
    SoQtExaminerViewer::render();
    globalLutUpdated = false;
}

//======================================================================
//
// Description:
//	axis setup.
//
//
// Use: private
//======================================================================
SoNode *InvViewer::makeAxis()
{
    SoInput in;

    in.setBuffer((void *)axis, strlen(axis));
    SoNode *result;

    SoDB::read(&in, result);

    return result;
}

//======================================================================
//
// Description:
//  Adds the given geometry under the selection
//  node.
//
// Use: public
//
//======================================================================
void InvViewer::addToSceneGraph(SoGroup *child, const char *name, SoGroup *root)
{
    // add nodes under 'root', not sceneGraph
    if (root != NULL)
    {
        root->addChild(child);
    }
    else
    {
        selection->addChild(child);
    }

    (void)name;
    //addToObjectList((char *)name);
}

//======================================================================
//
// Description:
//    Hide editors in slave
//
//======================================================================
void InvViewer::removeEditors()
{
    if (!renderer->isMaster())
    {

#ifdef HAVE_EDITORS
        // Editor components
        if (materialEditor != NULL)
        {
            if (materialEditor->isAttached())
                materialEditor->detach();
            materialEditor->hide();
        }

/*  if ( backgroundColorEditor != NULL )
      {
         if ( backgroundColorEditor->isAttached() )
            backgroundColorEditor->detach();
         backgroundColorEditor->hide();
      }*/

/* if ( colorEditor != NULL )
      {
         if ( colorEditor->isAttached() )
            colorEditor->detach();
         colorEditor->hide();
      }*/
#endif
        /*
      if ( transformSliderSet != NULL )
      {                                           // hope that helps
         transformSliderSet->hide();;
      }

      if ( ambientColorEditor != NULL )
      {
         if ( ambientColorEditor->isAttached() )
            ambientColorEditor->detach();
         ambientColorEditor->hide() ;
      }*/
    }
}

//======================================================================
//
// Description:
//    Bring opened editors back on the screen
//
//======================================================================
void InvViewer::showEditors()
{

    if (!renderer->isMaster())
    {

#ifdef HAVE_EDITORS
        // Editor components
        if (materialEditor != NULL)
        {
            if (!materialEditor->isVisible())
                materialEditor->show();
        }

/* if ( colorEditor != NULL )
      {
         if ( !colorEditor->isVisible() )
            colorEditor->show();
      }*/
#endif

        /*
      if ( transformSliderSet != NULL )
      {
         if ( !transformSliderSet->isVisible() )
            transformSliderSet->show();;
      }

      if ( ambientColorEditor != NULL )
      {
         if ( !ambientColorEditor->isVisible() )
            ambientColorEditor->show() ;
      }*/

        if (backgroundColorEditor != NULL)
        {
            if (!backgroundColorEditor->isVisible())
                backgroundColorEditor->show();
        }
    }
}

#ifdef HAVE_EDITORS

//======================================================================
//
// Description:
//	Create a color editor for the currently selected object.
//      Attachment code copied from SoXformManip.c++
//
//======================================================================
void InvViewer::createColorEditor()
{
    /*
   editMaterial = findMaterialForAttach( NULL );

   if (colorEditor == NULL)
   {
      colorEditor = new SoQtColorEditor;
      colorEditor->setTitle("Diffuse Color");
   }

   if (editMaterial != NULL)
   {
      colorEditor->attach(&(editMaterial->diffuseColor) );
   }

   colorEditor->show();
   */
}

//======================================================================
//
// Description:
//	Create a material editor for the currently selected object.
//
//======================================================================
void InvViewer::createMaterialEditor()
{
    if (materialEditor == NULL)
        materialEditor = new SoQtMaterialEditor;
    materialEditor->show();

    materialEditor->attach(findMaterialForAttach(NULL));
}
#endif

//======================================================================
//
// Description:
//  Adds the given geometry under the selection
//  node. If the node didn't have any children, the viewAll() method is
//  automatically called.
//
// Use: public
//
//======================================================================
void InvViewer::removeFromSceneGraph(SoGroup *delroot, const char *name)
{

    selection->deselectAll();
    if (selection->findChild(delroot) >= 0)
    {
        selection->removeChild(delroot);
    }

    // now that we have removed the subgroup from our
    // selection we delete every child node under delroot

    (void)name;
    //removeFromObjectList(name);
}

//======================================================================
//
// Description:
//  Replaces the given geometry under the selection
//  node. Camera position does not change
//
// Use: public
//
//======================================================================
void InvViewer::replaceSceneGraph(SoNode *root)
{

    // remove old scene
    deleteScene();

    // add new nodes under selection, not sceneGraph
    selection->addChild(root);

    // if you want to view the whole scene when new data arrives
    // uncomment the four following lines
    //   SbBool hadNoChildren = (selection->getNumChildren() == 0);
    //   if (hadNoChildren) {
    //	   viewAll();
    //	   saveHomePosition(); }
}

//======================================================================
//
// Description:
//	detach everything and nuke the existing scene.
//
// Use: private
//
//======================================================================
void InvViewer::deleteScene()
{
    // deselect everything (also detach manips)
    selection->deselectAll();

    // remove the geometry under the selection node
    for (int i = selection->getNumChildren(); i > 0; i--)
    {
        selection->removeChild(i - 1);
    }

    // add permanent part
    selection->addChild(permanentSelection);
}

//======================================================================
//
//  sets the axis on or off
//
//  Use: static private
//
//
//
//======================================================================
void InvViewer::setAxis(int onoroff)
{

    axis_state = onoroff;
    if (onoroff == CO_OFF)
        axis_switch->whichChild.setValue(SO_SWITCH_NONE);
    else
        axis_switch->whichChild.setValue(0);
}

//======================================================================
//
// Description:
//	set new transformation
//
//======================================================================
void InvViewer::setTransformation(float pos[3], float ori[4], int view,
                                  float aspect, float mynear, float myfar,
                                  float focal, float angleORheightangle)

{

    SoPerspectiveCamera *myPerspCamera = (SoPerspectiveCamera *)this->getCamera();
    SoOrthographicCamera *myOrthoCamera = (SoOrthographicCamera *)this->getCamera();

    if (renderer->isMaster() == false)
    {

        if (this->getCameraType() == SoPerspectiveCamera::getClassTypeId())
        {

            SbVec3f nPos(pos[0], pos[1], pos[2]);
            /*float oLen = myPerspCamera->position.getValue().length();
         float nLen = nPos.length();

         if (oLen != 0.0)
            {
            Annotations->reScale( nLen / oLen );
            }*/

            myPerspCamera->heightAngle = angleORheightangle;
            myPerspCamera->position = nPos;
            myPerspCamera->orientation = SbRotation(ori[0], ori[1], ori[2], ori[3]);
            myPerspCamera->viewportMapping = view;
            myPerspCamera->aspectRatio = aspect;
            myPerspCamera->nearDistance = mynear;
            myPerspCamera->farDistance = myfar;
            myPerspCamera->focalDistance = focal;
        }

        else
        {
            myOrthoCamera->position = SbVec3f(pos[0], pos[1], pos[2]);
            myOrthoCamera->orientation = SbRotation(ori[0], ori[1], ori[2], ori[3]);
            myOrthoCamera->viewportMapping = view;
            myOrthoCamera->aspectRatio = aspect;
            myOrthoCamera->nearDistance = mynear;
            myOrthoCamera->farDistance = myfar;
            myOrthoCamera->focalDistance = focal;
            myOrthoCamera->height = angleORheightangle;
        }
    }
}

//======================================================================
//
// Description:
//	get actual transformation
//
//======================================================================
void InvViewer::getTransformation(float pos[3], float ori[4], int *view,
                                  float *aspect, float *mynear, float *myfar,
                                  float *focal, float *angleORheightangle)

{
    SbVec3f p;
    SbRotation r;

    p = this->getCamera()->position.getValue();
    r = this->getCamera()->orientation.getValue();
    *view = this->getCamera()->viewportMapping.getValue();
    *aspect = this->getCamera()->aspectRatio.getValue();
    *mynear = this->getCamera()->nearDistance.getValue();
    *myfar = this->getCamera()->farDistance.getValue();
    *focal = this->getCamera()->focalDistance.getValue();

    if (this->getCameraType() == SoPerspectiveCamera::getClassTypeId())
        *angleORheightangle = ((SoPerspectiveCamera *)this->getCamera())->heightAngle.getValue();
    else
        *angleORheightangle = ((SoOrthographicCamera *)this->getCamera())->height.getValue();

    p.getValue(pos[0], pos[1], pos[2]);
    r.getValue(ori[0], ori[1], ori[2], ori[3]);
}

//======================================================================
//
// Description:
//	projection of telepointer
//
//======================================================================
void InvViewer::projectTP(int mousex, int mousey, SbVec3f &intersection, float &aspectRatio)
{
    // take the x,y position of the mouse, and normalize to [0,1].
    // X windows have 0,0 at the upper left,
    // Inventor expects 0,0 to be the lower left.

    const int xOffset = 61;
    const int yOffset = 33;

    float mx_norm;
    float my_norm;

    SbVec2s size = getSize();
    // !! Attention: SoXtRenderArea with offset
    size[0] = size[0] - xOffset;
    size[1] = size[1] - yOffset;

    mx_norm = float(mousex) / size[0];
    my_norm = float(size[1] - mousey) / size[1];

    // viewport mapping
    SoOrthographicCamera *cam = (SoOrthographicCamera *)tpHandler->getCamera();

    // set aspect ratio explicitely
    aspectRatio = size[0] / (float)size[1];
    cam->aspectRatio.setValue(aspectRatio);

    // default setting
    cam->height.setValue(2.0);

    // scale height
    cam->scaleHeight(1 / aspectRatio);

    // get view volume -> rectangular box
    SbViewVolume viewVolume = tpHandler->getCamera()->getViewVolume();

    // project the mouse point to a line
    SbVec3f p0, p1;
    viewVolume.projectPointToLine(SbVec2f(mx_norm, my_norm), p0, p1);

    // take the midpoint of the line as telepointer position
    intersection = (p0 + p1) / (float)2.0;
}

//====================================================================
// X event handling routine called before events are passed to inventor
// for telepointer handling
// Several changes by Uwe Woessner
//====================================================================
void InvViewer::processEvent(QEvent *anyevent)
{
    QMouseEvent *me;
    QKeyEvent *ke;
    //int               mouseX, mouseY;
    float aspRat;
    SbVec3f intersection;
    SbMatrix mx;
    float px, py, pz;
    //int               keyState;

    char message[300];
    char mess[150];
    float pos[3];
    float ori[4];
    int view;
    float aspect;
    float mynear;
    float myfar;
    float focal;
    float angleORheightangle; // depends on camera type !

    //
    // other X stuff (telepointer, mouse movement etc.)
    //

    switch (anyevent->type())
    {

    case QEvent::MouseMove:

        me = (QMouseEvent *)anyevent;
        mouseX = me->x();
        mouseY = me->y();

        // update slaves!
        if ((me->buttons() & Qt::LeftButton) || (me->buttons() & Qt::MidButton))
        {

            if (renderer->isMaster() && renderer->getSyncMode() == InvMain::SYNC_TIGHT)
            {
                getTransformation(pos, ori, &view, &aspect, &mynear, &myfar, &focal, &angleORheightangle);
                sprintf(message, "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %5d %7.3f %7.3f %7.3f %7.3f %7.3f",
                        pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3], view, aspect, mynear, myfar, focal, angleORheightangle);

#ifndef YAC
                renderer->cm->sendCameraMessage(message);
#endif
            }
        }

        if (keyState != -1)
        {
            projectTP(mouseX, mouseY, intersection, aspRat);
            intersection.getValue(px, py, pz);

            // test locally
            if (tpShow_)
            {
                sprintf(mess, "%s %d %f %f %f %f", (const char *)renderer->getUsername().toLatin1(), CO_ON, px, py, pz, aspRat);
                sendTelePointer(renderer->getUsername(), CO_ON, px, py, pz, aspRat);
                tpHandler->handle(mess);
            }

            //if(vrml_syn_) rm_sendVRMLTelePointer(mess);
        }
        break;

    case QEvent::KeyPress:

        ke = (QKeyEvent *)anyevent;

        if (ke->key() == Qt::Key_Shift)
        {
            keyState = 1;
            projectTP(mouseX, mouseY, intersection, aspRat);
            intersection.getValue(px, py, pz);
            sendTelePointer(renderer->getUsername(), CO_ON, px, py, pz, aspRat);

            // test locally
            sprintf(mess, "%s %d %f %f %f %f", (const char *)renderer->getUsername().toLatin1(), CO_ON, px, py, pz, aspRat);

            tpHandler->handle(mess);

            tpShow_ = true;
            //if(vrml_syn_) rm_sendVRMLTelePointer(mess);
        }

        break;

    case QEvent::KeyRelease:

        ke = (QKeyEvent *)anyevent;
        if (ke->key() == Qt::Key_Shift)
        {
            keyState = 0;
            projectTP(mouseX, mouseY, intersection, aspRat);
            intersection.getValue(px, py, pz);
            sendTelePointer(renderer->getUsername(), CO_RMV, px, py, pz, aspRat);

            // test locally
            sprintf(mess, "%s %d %f %f %f %f", (const char *)renderer->getUsername().toLatin1(), CO_RMV, px, py, pz, aspRat);
            tpHandler->handle(mess);

            //if(vrml_syn_) rm_sendVRMLTelePointer(mess);

            tpShow_ = false;
        }
        else if (ke->key() == Qt::Key_P) // snapshot
        {
            int width, height;
            width = getRenderAreaWidget()->width();
            height = getRenderAreaWidget()->height();
            setRenderWindowCaptureSize(width, height);
            writeRenderWindowSnapshot();
        }
        else if (ke->key() == Qt::Key_V) // video
        {
            int width, height;
            width = getRenderAreaWidget()->width();
            height = getRenderAreaWidget()->height();
            setRenderWindowCaptureSize(width, height);
            if (isEnabledRenderWindowCapture() == true)
                enableRenderWindowCapture(false);
            else
                enableRenderWindowCapture(true);
        }
        else if (ke->key() == Qt::Key_F) // frames per second
        {
            if (isEnabledFramePerSecondOutputConsole() == true)
            {
                enableFramePerSecondOutputConsole(false);
            }
            else
            {
                enableFramePerSecondOutputConsole(true);
            }
        }
        break;

    default:
        keyState = -1;
        break;
    }

    SoQtExaminerViewer::processEvent(anyevent);
}

//======================================================================
//
// Description:
//	user starts editing in viewer.
//
//======================================================================
void InvViewer::viewerStartEditCB(void *, SoQtViewer *sv)
{

    InvViewer *v = (InvViewer *)sv;

    if (renderer->isMaster())
    {
        v->viewer_edit_state = CO_ON;
    }
}

//======================================================================
//
// Description:
//	user finishs editing in viewer.
//
//======================================================================
void InvViewer::viewerFinishEditCB(void *, SoQtViewer *sv)
{
    InvViewer *v = (InvViewer *)sv;

    // let's send a transformation
    if (renderer->isMaster() && renderer->getSyncMode() != InvMain::SYNC_LOOSE)
    {
        v->viewer_edit_state = CO_OFF;

        //
        // get the current camera and pass the values to the communication
        // manager
        //
        //   longer messages for NaN
        char message[300];
        float pos[3];
        float ori[4];
        int view;
        float aspect;
        float mynear;
        float myfar;
        float focal;
        float angleORheightangle; // depends on camera type !

        v->getTransformation(pos, ori, &view, &aspect, &mynear, &myfar, &focal, &angleORheightangle);

        //
        // pack into character string

        sprintf(message, "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %5d %7.3f %7.3f %7.3f %7.3f %7.3f",
                pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3], view, aspect, mynear, myfar, focal, angleORheightangle);

#ifndef YAC
        renderer->cm->sendCameraMessage(message);
#endif
    }
}

//======================================================================
//
// Description:
//	receive new telepointer
//
//======================================================================
void InvViewer::sendTelePointer(QString &tpname, int state, float px, float py, float pz,
                                float aspectRatio)
{
    char message[200];

    sprintf(message, "%s %d %f %f %f %f", (const char *)tpname.toLatin1(), state, px, py, pz, aspectRatio);

// set the transformation in the correct node
#ifndef YAC
    renderer->cm->sendTelePointerMessage(message);
#endif
}

//======================================================================
//
//  Manage the changes in the selected node(s)
//
//======================================================================
// object
void InvViewer::deselectionCallback(void *userData, SoPath *deselectedObject)
{
    char objName[255];
    int num;
    SbName string;
    const SoPathList *deselectedObjectsList;

    InvViewer *sv = (InvViewer *)userData;

    if (sv->selectionCallbackInactive)
        return; // don't do anything (destructive)

    // now, turn off the sel/desel callbacks.
    // we'll turn them on again after we've adjusted the selection
    sv->selection->removeSelectionCallback(InvViewer::selectionCallback, (void *)sv);
    sv->selection->removeDeselectionCallback(InvViewer::deselectionCallback, (void *)sv);
    sv->selectionCallbackInactive = true;

    InvPlaneMover::deSelectionCB(sv->pm_, deselectedObject);

    // remove the manip
    sv->detachManip(deselectedObject);

// Remove editors
#ifdef HAVE_EDITORS
    if (sv->materialEditor)
        sv->materialEditor->detach();

/* if (sv->colorEditor)
      sv->colorEditor->detach();*/
#endif
    /*if (sv->transformSliderSet)
      sv->transformSliderSet->setNode( NULL );    // ??? same as detach ??? rc
   */

    // reset current transform node to NULL
    if (sv->currTransformNode != NULL)
    {
        sv->transformSensor->detach();
        sv->currTransformPath = NULL;
        sv->currTransformNode = NULL;
        sv->transformNode = NULL;
        sv->transformPath = NULL;
    }

    else
    {
#ifdef YAC
        LOGINFO("ERROR: Cannot detach transformCallback from node");
#else
        cerr << "ERROR: Cannot detach transformCallback from node" << endl;
#endif
    }

    // get the object-names from the selected objects
    num = sv->selection->getNumSelected();
    deselectedObjectsList = sv->selection->getList();

    // enable/disable cmd key shortcuts and menu items
    //sv->updateCommandAvailability();

    // update the object list
    sv->updateObjectList(deselectedObject, (SbBool) false);

    // send to other renderer's
    for (int i = 0; i < num; i++)
    {
        string = (*deselectedObjectsList)[i]->getTail()->getName();
        strcpy(objName, string.getString());
#ifndef YAC
        renderer->cm->sendDeselectionMessage(objName);
#endif
        renderer->om->updateObjectList(objName, false);
    }
    string = deselectedObject->getTail()->getName();
    strcpy(objName, string.getString());

// update the object list
#ifndef YAC
    renderer->cm->sendDeselectionMessage(objName);
#endif

    // turn on callbacks again
    sv->selection->addSelectionCallback(InvViewer::selectionCallback, (void *)sv);
    sv->selection->addDeselectionCallback(InvViewer::deselectionCallback, (void *)sv);
    sv->selectionCallbackInactive = false;
}

//======================================================================
//
//  Manage the changes in the selected node(s)
//
//======================================================================
// object
void InvViewer::selectionCallback(void *userData, SoPath *selectedObject)
{
    int num;
    SbName string;
    //SoPath      *editTransformPath;
    const SoPathList *selectedObjectsList;
    SoNodeList *nodeList;
    SoPathList *pathList;
    char objName[255];

    InvViewer *sv = (InvViewer *)userData;

    fprintf(stderr, "selection: %d - %s\n",
            (int)sv->selectionCallbackInactive,
            selectedObject->getTail()->getName().getString());

    if (sv->selectionCallbackInactive)
        return; // don't do anything (destructive)

    /*if (Annotations->isActive())
      {

      Widget widget = sv->getRenderAreaWidget();
      SbVec2s sz;
      if (widget != NULL)
         {
         Arg args[2];
         int n = 0;
         XtSetArg(args[n], XtNwidth, &sz[0]);  n++;
         XtSetArg(args[n], XtNheight, &sz[1]); n++;
   XtGetValues(widget, args, n);
   }

   SbViewportRegion vpReg;
   vpReg.setWindowSize(sz);

   vpReg = sv->getCamera()->getViewportBounds(vpReg);

   SoGetBoundingBoxAction bBoxAct(vpReg);
   bBoxAct.apply( sv->selection );

   SbBox3f bb = bBoxAct.getBoundingBox();

   Annotations->setSize( bb );
   Annotations->selectionCB( Annotations, selectedObject );

   sv->selection->deselectAll();
   return;
   }*/

    // first we try to find an object with interaction capabilities
    // (at the moment (22.08.01) CuttingSurface is the only one)
    int len = selectedObject->getLength();
    int ii;
    char *selObjNm;
    int csFlg = 0;
    for (ii = 0; ii < len; ii++)
    {
        SoNode *obj = selectedObject->getNode(ii);
        char *tmp = (char *)obj->getName().getString();
        selObjNm = new char[1 + strlen(tmp)];
        strcpy(selObjNm, tmp);
        if (strncmp(selObjNm, "CuttingSurface", 14) == 0)
        {
            csFlg = 1;
            break;
        }

        if (strncmp(selObjNm, "VectorField", 11) == 0)
        {
            csFlg = 1;
            break;
        }
        delete[] selObjNm;
    }

    if (csFlg)
    {
        //Widget widget = sv->getRenderAreaWidget();
        SbVec2s sz = sv->getSize();
        /*if (widget != NULL)
      {
         Arg args[2];
         int n = 0;
         XtSetArg(args[n], XtNwidth, &sz[0]);  n++;
         XtSetArg(args[n], XtNheight, &sz[1]); n++;
         XtGetValues(widget, args, n);
      }*/

        SbViewportRegion vpReg;
        vpReg.setWindowSize(sz);

        vpReg = sv->getCamera()->getViewportBounds(vpReg);

        SoGetBoundingBoxAction bBoxAct(vpReg);
        bBoxAct.apply(selectedObject);

        SbBox3f bb = bBoxAct.getBoundingBox();
        sv->pm_->setSize(bb);
        InvPlaneMover::selectionCB(sv->pm_, selectedObject);
    }

    // now, turn off the sel/desel callbacks.
    // we'll turn them on again after we've adjusted the selection
    sv->selection->removeSelectionCallback(InvViewer::selectionCallback, (void *)sv);
    sv->selection->removeDeselectionCallback(InvViewer::deselectionCallback, (void *)sv);
    sv->selectionCallbackInactive = true;

    if (!selectedObject->containsNode(sv->clipDragger))
    {
        // attach the manip
        sv->attachManip(sv->curManip, selectedObject);
    }

//
// every time an object gets selected we should check if the spacemouse
// was successfully created at startup time, if not we can do this now
//

#ifdef HAVE_EDITORS
    //
    // If active, attach editors to new selection.
    //
    SoMaterial *mtl = NULL;

    if (sv->materialEditor && sv->materialEditor->isVisible())
    {
        mtl = sv->findMaterialForAttach(selectedObject);
        if (mtl != NULL)
            sv->materialEditor->attach(mtl);
    }

/*  if (sv->colorEditor && sv->colorEditor->isVisible())
   {
      if (mtl == NULL)
         mtl = sv->findMaterialForAttach(selectedObject);
      if (mtl != NULL)
         sv->colorEditor->attach(&(mtl->diffuseColor), 0, mtl);
   }*/
#endif

    /*if ( sv->transformSliderSet && sv->transformSliderSet->isVisible() )
      {
      SoPath      *editTransformPath;
      editTransformPath = sv->findTransformForAttach( selectedObject );
      if ( editTransformPath == NULL )
         {
         sv->transformSliderSet->setNode( NULL );
         }
      else
         {
         editTransformPath->ref();
   sv->transformSliderSet->setNode(((SoFullPath *)editTransformPath)->getTail() );
   editTransformPath->unref();
   }
   }*/

    // get the object-names from the selected objects
    num = sv->selection->getNumSelected();
    selectedObjectsList = sv->selection->getList();

    nodeList = new SoNodeList(num);
    pathList = new SoPathList(num);

    /*int i;
   for (i=0; i<num; i++)
      {

      sv->findObjectName(&objName[0], (*selectedObjectsList)[i]);

      if (objName != NULL)
         {

         // find the transform nodes and attach the sensors:
         // The last selected object is the current selected object
   editTransformPath = sv->findTransformForAttach( (*selectedObjectsList)[i] );
   if (  editTransformPath !=NULL )
   {
   if ( i<num-1 )
   {
   nodeList->append((SoTransform *)(editTransformPath->getTail() ));
   pathList->append(editTransformPath);
   }
   else
   {
   sv->currTransformPath = editTransformPath;
   sv->currTransformNode = (SoTransform *)(editTransformPath->getTail() );
   sv->transformSensor->attach(sv->currTransformNode) ;
   }
   }
   else
   print_comment(__LINE__,__FILE__,"ERROR: no object for selection found in sectionCB");
   }
   }*/

    sv->transformNode = nodeList;
    sv->transformPath = pathList;

    // enable/disable cmd key shortcuts and menu items
    //sv->updateCommandAvailability();

    // update the object list
    sv->updateObjectList(selectedObject, (SbBool) true);

// send selection info to other renderer's for all selected objects
#ifndef YAC
    renderer->cm->sendSelectionMessage("DESELECT");
#endif
    for (int i = 0; i < num; i++)
    {
        string = (*selectedObjectsList)[i]->getTail()->getName();
        strcpy(objName, string.getString());
#ifndef YAC
        renderer->cm->sendSelectionMessage(objName);
#endif
    }

    // turn on callbacks again
    sv->selection->addSelectionCallback(InvViewer::selectionCallback, (void *)sv);
    sv->selection->addDeselectionCallback(InvViewer::deselectionCallback, (void *)sv);
    sv->selectionCallbackInactive = false;
}

//===================================================================
//
//  Manage the changes in the selected node(s)
//
//===================================================================
SoPath *InvViewer::pickFilterCB(void *userData, const SoPickedPoint *pick)
{
    InvViewer *sv = (InvViewer *)userData;
    SoPath *filteredPath = NULL;

    // If there are any transform manips along the path, check if they
    // belong to our personal set of manips.
    // If so, change the path so it points to the object the manip
    // is attached to.

    SoFullPath *fullP = (SoFullPath *)pick->getPath();

    fprintf(stderr, "pickfilter: %s\n", fullP->getTail()->getName().getString());

    /*if (Annotations->isActive())
      {
      return Annotations->pickFilterCB( Annotations, pick );
      }*/

    SbVec3f point = pick->getPoint();

    sv->pm_->setPosition(point);

    SoNode *n;
    for (int i = 0; i < fullP->getLength(); i++)
    {
        n = fullP->getNode(i);
        if (n->isOfType(SoTransformManip::getClassTypeId()))
        {
            int which = sv->maniplist->find((SoTransformManip *)n);
            if (which != -1)
            {
                filteredPath = sv->maniplist->getSelectionPath(which);
                return filteredPath;
            }
        }
    }

    // If we didn't pick one of our manipulators, then return the pickPath
    filteredPath = pick->getPath();
    return filteredPath;
}

//======================================================================
//
// Description:
//	update object list according to selected objects.
//      is called from selection and deselection callbacks
//
//======================================================================
void InvViewer::updateObjectList(SoPath *selectionPath, SbBool isSelection)
{
    int j;
    const char *name;
    char objName[255];
    SbName string;

    SoNode *node = selectionPath->getTail();

    //
    // look on the left side if there is
    //
    SoGroup *sep = (SoGroup *)selectionPath->getNodeFromTail(1);
    (void)node;

    //
    // should be a separator !

    if (sep->isOfType(SoSeparator::getClassTypeId()))
    {
        // look for the label under the separator
        for (j = 0; j < sep->getNumChildren(); j++)
        {
            SoNode *n = sep->getChild(j);
            if (n->isOfType(SoLabel::getClassTypeId()))
            {
                // look into the label
                SoLabel *l = (SoLabel *)sep->getChild(j);
                string = l->label.getValue();
                name = string.getString();
                strcpy(objName, name);

                // select or deselect the itempos
                if (isSelection)
                {
                    renderer->om->updateObjectList(objName, true);
                    break;
                }

                else
                {
                    renderer->om->updateObjectList(objName, false);
                    break;
                }
            }
        }
    }

    else
    {
        sep = (SoGroup *)selectionPath->getNodeFromTail(2);
        if (sep->isOfType(SoSeparator::getClassTypeId()))
        {

            for (j = 0; j < sep->getNumChildren(); j++)
            {
                SoNode *n = sep->getChild(j);
                if (n->isOfType(SoLabel::getClassTypeId()))
                {
                    // look into the label
                    SoLabel *l = (SoLabel *)sep->getChild(j);
                    string = l->label.getValue();
                    name = string.getString();
                    strcpy(objName, name);

                    // select or deselect the itempos
                    if (isSelection)
                    {
                        renderer->om->updateObjectList(objName, true);
                        break;
                    }

                    else
                    {
                        renderer->om->updateObjectList(objName, false);
                        break;
                    }
                }
            }
        }
    }
}

//======================================================================
//
// Description:
//	select object according to pick in list
//
//======================================================================
void InvViewer::objectListCB(InvObject *item, bool Selected)
{
    static QString oldname;
    char buffer[255];
    SoSearchAction action;

    QString name; // initialized to null
    if (item)
    {
        name = item->getName();
    }
    else
    {
        return;
    }

    // now, turn off the sel/desel callbacks.
    // we'll turn them on again after we've adjusted the selection
    selection->removeSelectionCallback(InvViewer::selectionCallback, (void *)this);
    selection->removeDeselectionCallback(InvViewer::deselectionCallback, (void *)this);
    selectionCallbackInactive = true;

    if (c_first_time == 0)
    {
        oldname = "DEFAULT";
        c_first_time = 1;
    }

    detachManipFromAll();
    selection->deselectAll();

    // if item was selected select it, if item was deselected select it

    if (!name.isNull())
    {
        if (Selected) // toggle
        {

            sprintf(buffer, "G_%s", (const char *)name.toLatin1());
#ifndef YAC
            if (renderer->isMaster() && renderer->getSyncMode() != InvMain::SYNC_LOOSE)

                renderer->cm->sendSelectionMessage(buffer);
#endif

            action.setFind(SoSearchAction::NAME);
            action.setName(SbName(buffer));
            action.setInterest(SoSearchAction::FIRST);
            action.setSearchingAll(true);
            action.apply(selection);
            if (action.getPath())
            {
                selection->select(action.getPath());
                attachManip(curManip, action.getPath());
            }
            else
            {
                // try to find the object without G_ (for groups)

                action.setFind(SoSearchAction::NAME);
                action.setName(SbName((const char *)name.toLatin1()));
                action.setInterest(SoSearchAction::FIRST);
                action.setSearchingAll(true);
                action.apply(selection);
                if (action.getPath())
                {
                    selection->select(action.getPath());
                    attachManip(curManip, action.getPath());
                }
            }
            //findPlaneMover(action.getPath());
        }
        else
        { // select same item
            isSelected = 1;
            //item->setSelected(true);
            //renderer->objListBox->setSelected(item, true);
            QByteArray ba = name.toLatin1();
            sprintf(buffer, "G_%s", (const char *)ba);
            if (renderer->isMaster() && renderer->getSyncMode() != InvMain::SYNC_LOOSE)
                renderer->cm->sendDeselectionMessage(buffer);
            action.setFind(SoSearchAction::NAME);
            action.setName(SbName(buffer));
            action.setInterest(SoSearchAction::FIRST);
            action.setSearchingAll(true);
            action.apply(selection);

            if (action.getPath())
            {
                selection->deselect(action.getPath());
                attachManip(curManip, action.getPath());
                //findPlaneMover(action.getPath());
            }
            else
            {
                // try to find the object without G_ (for groups)

                action.setFind(SoSearchAction::NAME);
                action.setName(SbName((const char *)name.toLatin1()));
                action.setInterest(SoSearchAction::FIRST);
                action.setSearchingAll(true);
                action.apply(selection);
                if (action.getPath())
                {
                    selection->deselect(action.getPath());
                    attachManip(curManip, action.getPath());
                }
            }
        }
    }
    //
    // turn on again
    selection->addSelectionCallback(InvViewer::selectionCallback, (void *)this);
    selection->addDeselectionCallback(InvViewer::deselectionCallback, (void *)this);
    selectionCallbackInactive = false;
}

void InvViewer::onSequencerValueChanged(int val)
{
    if (m_current_tswitch)
    {
        if (val < m_current_tswitch->getNumChildren())
        {
            m_current_tswitch->whichChild = val;
            //SoQtExaminerViewer::render();
            // LOGINFO("onSequencerValueChanged(%d)", val);
        }
    }
}

//======================================================================
//
//
//======================================================================
void InvViewer::findPlaneMover(SoPath *selectedObject)
{
    int len = selectedObject->getLength();
    int ii;
    char *selObjNm;
    int csFlg = 0;

    for (ii = 0; ii < len; ii++)
    {
        SoNode *obj = selectedObject->getNode(ii);
        char *tmp = (char *)obj->getName().getString();
        selObjNm = new char[1 + strlen(tmp)];
        strcpy(selObjNm, tmp);
        if (strncmp(selObjNm, "CuttingSurface", 14) == 0)
        {
            csFlg = 1;
            break;
        }

        if (strncmp(selObjNm, "VectorField", 11) == 0)
        {
            csFlg = 1;
            break;
        }
        delete[] selObjNm;
    }

    if (csFlg)
    {
        SbVec2s sz;
        SbViewportRegion vpReg;

        vpReg.setWindowSize(sz);
        vpReg = getCamera()->getViewportBounds(vpReg);

        SoGetBoundingBoxAction bBoxAct(vpReg);
        bBoxAct.apply(selectedObject);
        SbBox3f bb = bBoxAct.getBoundingBox();
        pm_->setSize(bb);
        InvPlaneMover::selectionCB(pm_, selectedObject);
    }
}

//======================================================================
//
// Description:
//   Find the appropriate shape node in the scene graph for a
//   given object name.
//
//======================================================================
SoNode *InvViewer::findShapeNode(const char *Name)
{

    const char *name;
    SbName string;
    int i;

    SoSearchAction saLabel;
    SoPathList listLabel;
    SoLabel *label;

    saLabel.setFind(SoSearchAction::TYPE);
    saLabel.setInterest(SoSearchAction::ALL);
    saLabel.setType(SoLabel::getClassTypeId());
    saLabel.apply(selection);

    // get the list of paths
    listLabel = saLabel.getPaths();

    // cycle through the list and find (first) match
    if (listLabel.getLength() != 0)
    {
        for (i = 0; i < listLabel.getLength(); i++)
        {
            label = (SoLabel *)(listLabel[i]->getTail());
            string = label->label.getValue();
            name = string.getString();

            /*if (strcmp(name,Name) == 0 )
            {
            SbName sname;
            SoGroup *group = (SoGroup *)(listLabel[i]->getNodeFromTail(1));
            return group;
            }*/

            if (strcmp(name, Name) == 0)
            {

                SoGroup *group = (SoGroup *)(listLabel[i]->getNodeFromTail(1));
                SoNode *n = NULL;
                SbName sname;

                for (int j = 0; j < group->getNumChildren(); j++)
                {
                    n = group->getChild(j);
                    sname = n->getName();
                }
                return n;
            }
        }
    }

    else
    {
#ifdef YAC
        LOGINFO("ERROR: findShapeNode : no object with this name found");
#else
        cerr << "ERROR: findShapeNode : no object with this name found" << endl;
#endif
        return NULL;
    }

    return NULL;
}

//======================================================================
//
//  sets the selection
//
//======================================================================
void InvViewer::setSelection(const char *name)
{
    SoSearchAction action;

    if (name != NULL)
    {
        // first, detach manips from all selected objects
        //detachManipFromAll();

        // now, turn off the sel/desel callbacks.
        // we'll turn them on again after we've adjusted the selection
        selection->removeSelectionCallback(InvViewer::selectionCallback, this);
        selection->removeDeselectionCallback(InvViewer::deselectionCallback, this);
        selectionCallbackInactive = true;

        // deselect all if special message arrives,
        // otherwise build up a selection from the pasted paths
        if (strcmp(name, "DESELECT") == 0)
            selection->deselectAll();

        else
        {

            action.setFind(SoSearchAction::NAME);
            action.setName(SbName(name));
            action.setInterest(SoSearchAction::FIRST);
            action.setSearchingAll(true);
            action.apply(selection);

            selection->select(action.getPath());

            // update the object list item
            QString nname = name;
            nname.remove("G_");
            renderer->om->updateObjectList((const char *)nname.toLatin1(), true);

            selection->addSelectionCallback(InvViewer::selectionCallback, this);
            selection->addDeselectionCallback(InvViewer::deselectionCallback, this);
            selectionCallbackInactive = false;
        }
    }

    else
    {
#ifdef YAC
        LOGINFO("Name of object to select is NULL");
#else
        cerr << "Name of object to select is NULL" << endl;
#endif
    }
}

//======================================================================
//
//  sets the deselection
//
//======================================================================
void InvViewer::setDeselection(const char *name)
{
    SoSearchAction action;

    // first, detach manips from all selected objects
    //detachManipFromAll();

    if (name)
    {
        // now, turn off the sel/desel callbacks.
        // we'll turn them on again after we've adjusted the selection
        selection->removeSelectionCallback(InvViewer::selectionCallback, this);
        selection->removeDeselectionCallback(InvViewer::deselectionCallback, this);
        selectionCallbackInactive = true;

        //action.setFind(SoSearchAction::NAME);
        action.setName(SbName(name));
        action.setInterest(SoSearchAction::FIRST);
        action.setSearchingAll(true);
        action.apply(selection);

        selection->deselect(action.getPath());

        // update the object list item
        QString nname = name;
        nname.remove("G_");
        renderer->om->updateObjectList((const char *)nname.toLatin1(), false);

        selection->addSelectionCallback(InvViewer::selectionCallback, this);
        selection->addDeselectionCallback(InvViewer::deselectionCallback, this);
        selectionCallbackInactive = false;
    }
}

//======================================================================
//
// Description:
//	New data is going to be coming into the viewer.  Time to disconnect all
//  manipulators and picking, and wait for new information.  Might as well go
//  into a viewing mode as well, this gets rid of the manipulators, and puts
//  the user in control of viewing when new data shows up.
//
//======================================================================
void InvViewer::newData()
{
    selection->deselectAll();
}

//======================================================================
//
// Description:
//	This routine first detaches manipulators from all selected objects,
//      then attaches a manipulator to all selected objects.
//
//======================================================================
void InvViewer::replaceAllManips(InvEManipMode manipMode)
{
    detachManipFromAll();
    attachManipToAll(manipMode);
}

//======================================================================
//
// Description:
//	This routine attaches a manipulator to all selected objects.
//
//======================================================================
void InvViewer::attachManipToAll(InvEManipMode manipMode)
{
    int i;

    for (i = 0; i < selection->getNumSelected(); i++)
    {
        SoPath *p = (*selection)[i];
        attachManip(manipMode, p);
    }
}

//======================================================================
//
// Description:
//	This routine attaches and activates a manipulator.
//
//======================================================================
void InvViewer::attachManip(InvEManipMode manipMode, SoPath *selectedPath)
{
    SoTransformManip *theXfManip;
    SoPath *xfPath;

    //
    // Attach to a manipulator.
    //

    if (manipMode == SV_NONE)
        return;

    xfPath = findTransformForAttach(selectedPath);
    xfPath->ref();
    theXfManip = NULL;

    switch (manipMode)
    {
    case SV_TRACKBALL:
        theXfManip = new SoTrackballManip;
        break;

    case SV_HANDLEBOX:
        theXfManip = new SoHandleBoxManip;

        break;

    case SV_JACK:
        theXfManip = new SoJackManip;
        break;

    case SV_CENTERBALL:
        theXfManip = new SoCenterballManip;
        break;

    case SV_XFBOX:
        theXfManip = new SoTransformBoxManip;
        break;

    case SV_TABBOX:
        theXfManip = new SoTabBoxManip;
        break;

    case SV_NONE:
        return;
    }

    if (theXfManip)
    {

        SoFullPath *fp = (SoFullPath *)xfPath;
        SoTransform *oldXf = (SoTransform *)fp->getTail();
        oldXf->ref();
        theXfManip->ref();

        if (!theXfManip->replaceNode(xfPath))
        {
            theXfManip->unref();
        }

        // If the transformSliderSet is attached to the oldXf, then attach
        // it to the new manip instead.
        /*if ( transformSliderSet && transformSliderSet->isVisible()
         && transformSliderSet->getNode() == oldXf)
         {
         transformSliderSet->setNode(theXfManip);
         }*/

        // Add manip and paths to the maniplist (maniplist will ref/unref)
        ///	maniplist->append(selectedPath, theXfManip, xfPath );
        maniplist->append(selectedPath, theXfManip, (SoPath *)xfPath);
        theXfManip->unref();
        oldXf->unref();

        if (manipMode == SV_TABBOX)
        {
            // Special case!  When using a  tab box, we want to adjust the
            // scale tabs upon viewer finish.
            //addFinishCallback(InvViewer::adjustScaleTabSizeCB, theXfManip->getDragger());
        }

        if (manipMode == SV_JACK)
        {
            SoFullPath *jackP = (SoFullPath *)selectedPath;
            SoType bkType = SoBaseKit::getClassTypeId();
            int lastKitInd = -1;
            for (int i = jackP->getLength() - 1; i >= 0; i--)
            {
                if (jackP->getNode(i)->isOfType(bkType))
                {
                    lastKitInd = i;
                    break;
                }
            }

            // If there's a lastKitInd, make jackP be a copy of
            // selectedPath, but only up to lastKitInd.
            if (lastKitInd != -1)
                jackP = (SoFullPath *)selectedPath->copy(0, lastKitInd + 1);

            // Get the dragger from the manip (the manip contains the dragger,
            // and the dragger has the parts).
            SoDragger *d = theXfManip->getDragger();

            // Use jackP to set the translator parts, then discard (unref) it:
            jackP->ref();
            d->setPartAsPath("translator.yzTranslator.translator", jackP);
            d->setPartAsPath("translator.xzTranslator.translator", jackP);
            d->setPartAsPath("translator.xyTranslator.translator", jackP);
            d->setPartAsPath(
                "translator.yzTranslator.translatorActive", jackP);
            d->setPartAsPath(
                "translator.xzTranslator.translatorActive", jackP);
            d->setPartAsPath(
                "translator.xyTranslator.translatorActive", jackP);
            jackP->unref();
        }
    }

    xfPath->unref();
}

//======================================================================
//
// Description:
//	This routine detaches the manipulators from all selected objects.
//
//======================================================================
void InvViewer::detachManipFromAll()
{
    //
    // Loop from the end of the list to the start.
    //
    for (int i = selection->getNumSelected() - 1; i >= 0; i--)
    {
        SoPath *p = (SoPath *)(*selection)[i];
        detachManip(p);
    }
}

//======================================================================
//
// Description:
//	This routine detaches a manipulator.
//
// Use: private
//
//======================================================================
void InvViewer::detachManip(SoPath *p)
{
    //
    // Detach manip and remove from scene graph.
    //
    int which = maniplist->find(p);
    // See if this path is registered in the manip list.
    if (which != -1)
    {
        // remove from scene graph
        SoTransformManip *manip = (SoTransformManip *)maniplist->getManip(which);

        if (manip->isOfType(SoTabBoxManip::getClassTypeId()))
        {
            // Special case!  When using a  tab box, we want to adjust the
            // scale tabs upon viewer finish.
            //removeFinishCallback(InvViewer::adjustScaleTabSizeCB, manip->getDragger() );
        }

        SoPath *xfPath = maniplist->getXfPath(which);
        SoTransform *newXf = new SoTransform;
        newXf->ref();
        manip->ref();

        // replace the manip
        manip->replaceManip(xfPath, newXf);

        // If the transformSliderSet is attached to the manip, then attach
        // it to the new node instead.
        /*if ( transformSliderSet && transformSliderSet->isVisible()
         && transformSliderSet->getNode() == manip)
         transformSliderSet->setNode(newXf);*/

        manip->unref();
        newXf->unref();

        // remove from maniplist
        maniplist->remove(which);
    }
}

//======================================================================
//
// Description:
//	Added as a finish callback to the current viewer. It makes sure
//      the scale tab size gets changed when a viewer gesture is
//      completed.
//
//======================================================================
void InvViewer::adjustScaleTabSizeCB(void *userData, InvViewer *)
{
    SoTabBoxDragger *dragger = (SoTabBoxDragger *)userData;
    dragger->adjustScaleTabSize();
}

//=======================================================================
//
// Description:
//	Temporarily remove manips from the scene.
// Restore them with a call to restoreManips().
//
//=======================================================================
void InvViewer::removeManips()
{
    // temporarily replace all manips with regular transform nodes.
    for (int m = 0; m < maniplist->getLength(); m++)
    {
        SoTransformManip *manip = (SoTransformManip *)maniplist->getManip(m);
        SoPath *xfPath = maniplist->getXfPath(m);
        manip->replaceManip(xfPath, NULL);
    }
}

//=======================================================================
//
// Description:
//	Restore manips that were removed with removeManips().
//
//=======================================================================
void InvViewer::restoreManips()
{
    // Now put the manips back in the scene graph.
    for (int m = 0; m < maniplist->getLength(); m++)
    {
        SoTransformManip *manip = (SoTransformManip *)maniplist->getManip(m);
        SoPath *xfPath = maniplist->getXfPath(m);
        manip->replaceNode(xfPath);
    }
}

//======================================================================
//
// Description:
//   Find the appropriate label node in the scene graph for sending to
//   slave renderers.
//   The given target path goes from the selection node to a selected
//   shape node
//
//======================================================================
// path to start search from
void InvViewer::findObjectName(char *objName, const SoPath *selectionPath)
{
    int j;
    const char *name;
    SbName string;

    SoNode *node = selectionPath->getTail();

    //  fprintf(stderr,"In findObjectName");

    //
    // look on the left side if there is
    //
    SoGroup *sep = (SoGroup *)selectionPath->getNodeFromTail(1);
    (void)node;
    //
    // should be a separator !

    if (sep->isOfType(SoSeparator::getClassTypeId()))
    {
        // look for the label under the separator

        for (j = 0; j < sep->getNumChildren(); j++)
        {
            SoNode *n = sep->getChild(j);

            if (n->isOfType(SoLabel::getClassTypeId()))
            {
                // look into the label
                SoLabel *l = (SoLabel *)sep->getChild(j);
                string = l->label.getValue();
                name = string.getString();
                strcpy(objName, name);
                break;
            }
        }
    }
}

//======================================================================
//
// Description:
//   Find the appropriate transform node in the scene graph for attaching a
//   transform editor or manipulator.
//
//   How we treat the 'center' field of the transform node:
//   If we need to create a new transform node:
//       set the 'center' to be the geometric center of all objects
//       affected by that transform.
//   If we find a transform node that already exists:
//       'center' will not be changed.
//
//   Three possible cases:
//        [1] The path-tail is a node kit. Just ask the node kit for a path
//            to the part called "transform"
//        [2] The path-tail is NOT a group.  We search the siblings of the path
//            tail (including the tail itself) from right to left for a node
//	      that is affected by transforms (shapes, groups, lights,cameras).
//            We stop the search if we come to a transform node to the left of
//	      the pathTail.  If we find a node that IS affected by transform,
//	      we will insert a transform node just before the path-tail. This is
//            because the editor should not affect nodes that appear
//            before attachPath in the scene graph.
//        [3] The path-tail IS a group.  We search the children from left to
//            right for transform nodes.
//            We stop the search if we come to a transform node.
//            If we find a node that is affected by transform, we will insert a
//	      transform just before this node. This is because the editor for a
//	      group should affect ALL nodes within that group.
//
// NOTE: For the purposes of this routine, we consider SoSwitch as different
//       from other types of group. This is because we don't want to put
//       the new node underneath the switch, but next to it.
//
// Use: private
//
//
//======================================================================
// path to start search from
SoPath *InvViewer::findTransformForAttach(const SoPath *target)
{
    int pathLength;
    SoPath *selectionPath;
    SoTransform *editXform;

    // fprintf(stderr,"In findTransformForAttach");

    if ((selectionPath = (SoPath *)target) == NULL)
    {
        //
        //  If no selection path is specified, then use the LAST path in the
        //  current selection list.
        //
        selectionPath = (*selection)[selection->getNumSelected() - 1];
    }
    pathLength = selectionPath->getLength();

    if (pathLength <= 0)
    {
        fprintf(stderr, "No objects currently selected...\n");
        return NULL;
    }

    // find 'group' and try to find 'editXform'
    SoGroup *group = NULL;
    SoNode *node;
    int index, i;
    SbBool isTailGroup, isTailKit;
    SbBool existedBefore = FALSE;
    SoPath *pathToXform = NULL;

    editXform = NULL;

    isTailGroup = (selectionPath->getTail()->isOfType(SoGroup::getClassTypeId())
                   && !selectionPath->getTail()->isOfType(SoSwitch::getClassTypeId()));

    isTailKit = selectionPath->getTail()->isOfType(SoBaseKit::getClassTypeId());

    //    CASE 1: The path-tail is a node kit.
    if (isTailKit)
    {

        // Nodekits have their own built in policy for creating new transform
        // nodes. Allow them to contruct and return a path to it.
        SoBaseKit *kit = (SoBaseKit *)((SoNodeKitPath *)selectionPath)->getTail();

        // Before creating path, see if the transform part exists yet:
        if (SO_CHECK_PART(kit, "transform", SoTransform) != NULL)
            existedBefore = TRUE;

        if ((editXform = SO_GET_PART(kit, "transform", SoTransform)) != NULL)
        {
            pathToXform = kit->createPathToPart("transform", TRUE, selectionPath);
            pathToXform->ref();
        }
        else
        {
            // This nodekit has no transform part.
            // Treat the object as if it were not a nodekit.
            isTailKit = FALSE;
        }
    }

    if (!isTailGroup && !isTailKit)
    {
        //
        //    CASE 2: The path-tail is not a group.
        //    'group'      becomes the second to last node in the path.
        //    We search the path tail and its siblings from right to left for a
        //    transform node.
        //    We stop the search if we come to a 'movable' node
        //    to the left of the pathTail.  If we find a movable node, we
        //    will insert a transform just before the path-tail. This is
        //    because the manipulator should not affect objects that appear
        //    before selectionPath in the scene graph.
        //
        group = (SoGroup *)selectionPath->getNode(pathLength - 2);
        index = group->findChild(selectionPath->getTail());

        for (i = index; (i >= 0) && (editXform == NULL); i--)
        {
            node = group->getChild(i);
            // found an SoMaterial
            if (node->isOfType(SoTransform::getClassTypeId()))
                editXform = (SoTransform *)node;
            else if (i != index)
            {
                if (isAffectedByTransform(node))
                    break;
            }
        }

        if (editXform == NULL)
        {
            existedBefore = FALSE;
            editXform = new SoTransform;
            group->insertChild(editXform, index);
        }

        else
            existedBefore = TRUE;
    }

    else if (!isTailKit)
    {
        //    CASE 3: The path-tail is a group.
        //    'group'      becomes the path tail
        //      We search the children from left to right for transform nodes.
        //      We stop the search if we come to a movable node.
        //      If we find a movable node, we will insert a transform just
        //      before this node. This is because the editor
        //      for a group should affect ALL objects within that group.
        //
        group = (SoGroup *)selectionPath->getTail();
        for (i = 0; (i < group->getNumChildren()) && (editXform == NULL); i++)
        {
            node = group->getChild(i);
            if (node->isOfType(SoTransform::getClassTypeId()))
                editXform = (SoTransform *)node;
            else if (isAffectedByTransform(node))
                break;
        }

        if (editXform == NULL)
        {
            existedBefore = FALSE;
            editXform = new SoTransform;
            group->insertChild(editXform, i);
        }
        else
            existedBefore = TRUE;
    }

    // If we don't have a path yet (i.e., we weren't handed a nodekit path)
    // create the 'pathToXform'
    // by copying editPath and making the last node in the path be editXform
    if (pathToXform == NULL)
    {
        if (!isTailGroup)
            // CASE 2: path-tail was NOT 'group' -- copy all but last entry
            pathToXform = selectionPath->copy(0, pathLength - 1);
        else
            // CASE 3: path-tail was 'group' -- copy all of editPath
            pathToXform = selectionPath->copy(0, pathLength);
        pathToXform->ref();

        // add the transform to the end
        if (group)
        {
            int xfIndex = group->findChild(editXform);
            pathToXform->append(xfIndex);
        }

        else
            fprintf(stderr, "InvViewer::findTransformForAttach(): group used uninitialized\n");
    }

    // Now. If we created the transform node right here, right now, then
    // we will set the 'center' field based on the geometric center. We
    // don't do this if we didn't create the transform, because "maybe it
    // was that way for a reason."
    if (existedBefore == FALSE)
    {
        // First, find 'applyPath' by popping nodes off the path until you
        // reach a separator. This path will contain all nodes affected by
        // the transform at the end of 'pathToXform'
        SoFullPath *applyPath = (SoFullPath *)pathToXform->copy();
        applyPath->ref();
        for (int i = (applyPath->getLength() - 1); i > 0; i--)
        {
            if (applyPath->getNode(i)->isOfType(SoSeparator::getClassTypeId()))
                break;
            applyPath->pop();
        }

        // Next, apply a bounding box action to applyPath, and reset the
        // bounding box just before the tail of 'pathToXform' (which is just
        // the editXform). This will assure that the only things included in
        // the resulting bbox will be those affected by the editXform.
        SoGetBoundingBoxAction bboxAction(getViewportRegion());
        bboxAction.setResetPath(pathToXform, TRUE, SoGetBoundingBoxAction::BBOX);
        bboxAction.apply(applyPath);

        applyPath->unref();

        // Get the center of the bbox in world space...
        SbVec3f worldBoxCenter = bboxAction.getBoundingBox().getCenter();

        // Convert it into local space of the transform...
        SbVec3f localBoxCenter;
        SoGetMatrixAction ma(getViewportRegion());
        ma.apply(pathToXform);
        ma.getInverse().multVecMatrix(worldBoxCenter, localBoxCenter);

        // Finally, set the center value...
        editXform->center.setValue(localBoxCenter);
    }

    pathToXform->unrefNoDelete();
    return (pathToXform);
}

//======================================================================
//
// Description:
//    Determines whether a given node is affected by a transform.
//
//======================================================================
// node to be affected?
SbBool InvViewer::isAffectedByTransform(SoNode *theNode)
{
    if (theNode->isOfType(SoGroup::getClassTypeId())
        || theNode->isOfType(SoShape::getClassTypeId())
               //!!	    || theNode->isOfType( SoCamera::getClassTypeId() )
        || theNode->isOfType(SoPerspectiveCamera::getClassTypeId())
        || theNode->isOfType(SoLight::getClassTypeId()))
    {
        return true;
    }
    return false;
}

//======================================================================
//
// Description:
//    Determines whether a given node is affected by material node.
//
//======================================================================
// node to be affected?
SbBool InvViewer::isAffectedByMaterial(SoNode *theNode)
{
    if (theNode->isOfType(SoGroup::getClassTypeId())
        || theNode->isOfType(SoShape::getClassTypeId()))
    {
        return true;
    }
    return false;
}

int InvViewer::toggleHandleState()
{
    if (handleState_)
    {
        handleState_ = 0;
        if (pm_)
            pm_->setFreeMotion();
    }

    else
    {
        handleState_ = 1;
        if (pm_)
            pm_->setSnapToAxis();
    }
    return handleState_;
}

//======================================================================
//
// Description:
//   Find the appropriate material node in the scene graph to attach a material
//   editor to.
//
//   Two possible cases:
//        [1] The path-tail is NOT a group.  We search the siblings of the path
//            tail (including the tail itself) from right to left for a node
//	      that is affected by materials (shapes or groups).
//            We stop the search if we come to a material node to the left of the
//	      pathTail.  If we find a node that IS affected by material, we will
//	      insert a material node just before the path-tail. This is
//            because the editor should not affect nodes that appear
//            before attachPath in the scene graph.
//        [2] The path-tail IS a group.  We search the children from left to
//            right for material nodes.
//            We stop the search if we come to a material node.
//            If we find a node that is affected by materials, we will insert a
//	      material just before this node. This is because the editor for a
//	      group should affect ALL nodes within that group.
//
// NOTE: For the purposes of this routine, we consider SoSwitch as different
//       from other types of group. This is because we don't want to put
//       the new node underneath the switch, but next to it.
//
// Use: private
//
//
//======================================================================
// path to start search from
SoMaterial *InvViewer::findMaterialForAttach(const SoPath *target)
{
    int pathLength;
    SoPath *selectionPath;
    SoMaterial *editMtl = NULL;

    SbBool madeNewMtl = FALSE; // did we create a new material
    // node within this method?

    if ((selectionPath = (SoPath *)target) == NULL)
    {
        //
        //  If no selection path is specified, then use the LAST path in the
        //  current selection list.
        //
        // last guy
        if (selection->getNumSelected() > 0)
        {
            selectionPath = (*selection)[selection->getNumSelected() - 1];
        }
    }
    if (selectionPath)
        pathLength = selectionPath->getLength();
    else
        pathLength = 0;

    if (pathLength <= 0)
    {
        fprintf(stderr, "No objects currently selected...\n");
        return NULL;
    }

    // find 'group' and try to find 'editMtl'
    SoGroup *group = NULL;
    SoNode *node;
    int index, i;
    SbBool ignoreNodekit = FALSE;

    if (selectionPath->getTail()->isOfType(SoBaseKit::getClassTypeId()))
    {
        // Nodekits have their own built in policy for creating new material
        // nodes. Allow them to contruct and return it.
        // Get the last nodekit in the path:
        SoBaseKit *kit = (SoBaseKit *)((SoNodeKitPath *)selectionPath)->getTail();
        // SO_CHECK_PART returns NULL if the part doesn't exist yet...
        editMtl = SO_GET_PART(kit, "material", SoMaterial);
        if (editMtl == NULL)
        {
            // This nodekit does not have a material part.
            // Ignore the fact that this is a nodekit.
            ignoreNodekit = TRUE;
        }
    }

    SbBool isTailGroup = selectionPath->getTail()->isOfType(SoGroup::getClassTypeId()) && (!selectionPath->getTail()->isOfType(SoSwitch::getClassTypeId()));

    if ((editMtl == NULL) && (!isTailGroup))
    {
        //
        //    CASE 1: The path-tail is not a group.
        //    'group'      becomes the second to last node in the path.
        //    We search the path tail and its siblings from right to left for a
        //    mtl node.
        //    We stop the search if we come to a shape node or a group node
        //    to the left of the pathTail.  If we find a shape or group, we
        //    will insert a mtl just before the path-tail. This is
        //    because the manipulator should not affect objects that appear
        //    before selectionPath in the scene graph.
        //
        group = (SoGroup *)selectionPath->getNode(pathLength - 2);
        index = group->findChild(selectionPath->getTail());

        for (i = index; (i >= 0) && (editMtl == NULL); i--)
        {
            node = group->getChild(i);
            // found SoMaterial
            if (node->isOfType(SoMaterial::getClassTypeId()))
                editMtl = (SoMaterial *)node;
            else if (i != index)
            {
                if (isAffectedByMaterial(node))
                    break;
            }
        }

        if (editMtl == NULL)
        {
            editMtl = new SoMaterial;
            group->insertChild(editMtl, index);
            madeNewMtl = TRUE;
        }
    }
    else if (editMtl == NULL)
    {
        //    CASE 2: The path-tail is a group.
        //    'group'      becomes the path tail
        //      We search the children from left to right for mtl nodes.
        //      We stop the search if we come to a shape node or a group node.
        //      If we find a shape or group, we will insert a mtl just
        //      before this shape or group. This is because the editor
        //      for a group should affect ALL objects within that group.
        //
        group = (SoGroup *)selectionPath->getTail();
        for (i = 0; (i < group->getNumChildren()) && (editMtl == NULL); i++)
        {
            node = group->getChild(i);
            if (node->isOfType(SoMaterial::getClassTypeId()))
                editMtl = (SoMaterial *)node;
            else if (isAffectedByMaterial(node))
                break;
        }

        if (editMtl == NULL)
        {
            editMtl = new SoMaterial;
            group->insertChild(editMtl, i);
            madeNewMtl = TRUE;
        }
    }

    // If we just created the material node here, then set the ignore
    // flags for all fields in the node.  This will cause the fields
    // to be inherited from their ancestors. The material editor will
    // undo these flags whenever it changes the value of a field
    if (madeNewMtl == TRUE)
    {
        editMtl->ambientColor.setIgnored(TRUE);
        editMtl->diffuseColor.setIgnored(TRUE);
        editMtl->specularColor.setIgnored(TRUE);
        editMtl->emissiveColor.setIgnored(TRUE);
        editMtl->shininess.setIgnored(TRUE);
        editMtl->transparency.setIgnored(TRUE);
    }

    // If any of the fields is ignored, then fill the value with the value
    // inherited from the rest of the scene graph
    if (editMtl->ambientColor.isIgnored()
        || editMtl->diffuseColor.isIgnored()
        || editMtl->specularColor.isIgnored()
        || editMtl->emissiveColor.isIgnored()
        || editMtl->shininess.isIgnored()
        || editMtl->transparency.isIgnored())
    {

        // Create a path to the material
        SoPath *mtlPath;
        if ((!ignoreNodekit) && selectionPath->getTail()->isOfType(SoBaseKit::getClassTypeId()))
        {
            SoBaseKit *kit = (SoBaseKit *)((SoNodeKitPath *)selectionPath)->getTail();
            mtlPath = kit->createPathToPart("material", TRUE, selectionPath);
            mtlPath->ref();
        }
        else
        {
            if (!isTailGroup)
            {
                // CASE 1: path-tail was NOT 'group' -- copy all but last entry
                mtlPath = selectionPath->copy(0, pathLength - 1);
            }
            else
            {
                // CASE 2: path-tail was 'group' -- copy all of editPath
                mtlPath = selectionPath->copy(0, pathLength);
            }
            mtlPath->ref();
            // add the material to the end of the path
            if (group)
            {
                int mtlIndex = group->findChild(editMtl);
                mtlPath->append(mtlIndex);
            }
            else
                fprintf(stderr, "InvViewer.cpp::findMaterialForAttach(): group used uninitialized\n");
        }

        // Pass the material node to an accumulate state callback
        // that will load any 'ignored' values with their inherited values.

        SoCallbackAction cba;
        cba.addPreTailCallback(InvViewer::findMtlPreTailCB, editMtl);
        cba.apply(mtlPath);

        mtlPath->unref();
    }

    return (editMtl);
}

//======================================================================
//
// Description:
//   Callback used by 'findMaterialForAttach' as part of the accumulate state
//   action. Returns 'PRUNE', which tells the action not to draw the
//   shape as part of the accum state action.
//   editor to.
//
// Use: private
//
//
//======================================================================
SoCallbackAction::Response InvViewer::findMtlPreTailCB(void *data, SoCallbackAction *accum,
                                                       const SoNode *)
//
////////////////////////////////////////////////////////////////////////
{
    SoMaterial *mtl = (SoMaterial *)data;

    SbColor ambient, diffuse, specular, emissive;
    float shininess, transparency;

    accum->getMaterial(ambient, diffuse, specular, emissive,
                       shininess, transparency);

    float r, g, b;
    diffuse.getValue(r, g, b);

    // inherit the accumulated values only in those fields being ignored.
    if (mtl->ambientColor.isIgnored())
        mtl->ambientColor.setValue(ambient);
    if (mtl->diffuseColor.isIgnored())
    {
        mtl->diffuseColor.setValue(diffuse);
    }
    if (mtl->specularColor.isIgnored())
        mtl->specularColor.setValue(specular);
    if (mtl->emissiveColor.isIgnored())
        mtl->emissiveColor.setValue(emissive);
    if (mtl->shininess.isIgnored())
        mtl->shininess.setValue(shininess);
    if (mtl->transparency.isIgnored())
        mtl->transparency.setValue(transparency);

    return SoCallbackAction::ABORT;
}

//======================================================================
//
// Description:
//	send clipping plane message after it has changed here
//      and we are master
//
//======================================================================
void InvViewer::sendClippingPlane(int onoroff, SbVec3f &normal, SbVec3f &point)
{
    char message[100];

    if (renderer->isMaster() && renderer->getSyncMode() != InvMain::SYNC_LOOSE)
    {
        sprintf(message, "%d %f %f %f %f %f %f",
                onoroff,
                normal[0], normal[1], normal[2],
                point[0], point[1], point[2]);
        fprintf(stderr, "sending clip plane: %s\n", message);
#ifndef YAC
        renderer->cm->sendClippingPlaneMessage(message);
#endif
    }
}

//======================================================================
//
//  sets clipping on or off
//
//  Use: static private
//
//======================================================================
void InvViewer::setClipping(int onoroff)
{
    //fprintf(stderr, "setClipping(%d)\n", onoroff);
    clipState = onoroff;
    if (clipState == CO_ON)
    {
        clipSwitch->whichChild.setValue(0);
    }
    else
    {
        clipSwitch->whichChild.setValue(SO_SWITCH_NONE);
    }
    clipPlane->on.setValue(onoroff);
}

//======================================================================
//
// Description:
//      Invokes editor to set a clipping plane.
//
//======================================================================
void InvViewer::toggleClippingPlaneEditor()
{
    if (clippingPlaneEditor == NULL)
    {
        clippingPlaneEditor = new InvClipPlaneEditor;
    }

    if (clippingPlaneEditor->isVisible())
    {
        clippingPlaneEditor->hide();
    }
    else
    {
        clippingPlaneEditor->show();
    }
}

//======================================================================
//
// Description:
//      Set the clipping plane equation.
//
//======================================================================
void InvViewer::setClipPlaneEquation(SbVec3f &normal, SbVec3f &point)
{
    clipPlane->plane.setValue(SbPlane(normal, point));

    sendClippingPlane(clipState, normal, point);
}

//======================================================================
//
//  sets the clipping plane equation
//
//  Use: static private
//
//======================================================================
void InvViewer::setClippingPlane(SbVec3f &point, SbVec3f &normal)
{
    clipPlane->plane.setValue(SbPlane(point, normal));
}

//======================================================================
//
// Description:
//    Make access to getInteractiveCount public
//
//======================================================================
int InvViewer::getInteractiveCount() const
{
    return SoQtExaminerViewer::getInteractiveCount();
}

//======================================================================
//
// Description:
//   Find the appropriate transform node in the scene graph for a
//   given object name.
//
// Use: private
//
//
//======================================================================
void InvViewer::updateObjectView()
{

    SoTransform *transform;
    const char *name;
    char objName[255];
    SbName string;
    int i, j;

    SoSearchAction saLabel;
    SoPathList listLabel;
    SoLabel *label;

    saLabel.setFind(SoSearchAction::TYPE);
    saLabel.setInterest(SoSearchAction::ALL);
    saLabel.setType(SoLabel::getClassTypeId());

    saLabel.apply(selection);

    // get the list of paths
    listLabel = saLabel.getPaths();

    // cycle through the list and find a match
    if (listLabel.getLength() != 0)
    {
        for (i = 0; i < listLabel.getLength(); i++)
        {
            label = (SoLabel *)(listLabel[i]->getTail());
            string = label->label.getValue();
            name = string.getString();
            strcpy(&objName[0], name);

            SoGroup *group = (SoGroup *)(listLabel[i]->getNodeFromTail(1));

            for (j = 0; j < group->getNumChildren(); j++)
            {
                SoNode *n = group->getChild(j);
                if (n->isOfType(SoTransform::getClassTypeId()))
                {
                    transform = (SoTransform *)group->getChild(j);
                    // send Transformation to the slave renderers
                    sendTransformation(objName, transform);
                }
            }
        }
    }
    else
    {
#ifdef YAC
        LOGINFO("Currently no objects in renderer");
#else
        cerr << "Currently no objects in renderer" << endl;
#endif
    }

    /*
   if(c_first_time)
      {
      int sel = 0;
      cmapSelected_.get(c_oldname,sel);
      if(sel == 1)
         {
         char buffer[255];
         sprintf(buffer,"%s",colormap_manager->currentColormap());
#ifndef YAC
               renderer->cm->ssendColormapMessage(buffer);
#endif
         sleep(1);                                //  redirection problem
   }
   }*/

    InvViewer::cameraCallback(this, NULL);
}

//======================================================================
//
// Description:
//	Collect transformation change stuff
//
// Use: private
//
//======================================================================
void InvViewer::sendTransformation(const char *name, SoTransform *transform)
{

    if (renderer->isMaster() && renderer->getSyncMode() == InvMain::SYNC_LOOSE)
    {

        if (transform != NULL)
        { // do not change this routine and NOT the line above

            char message[255];
            SbVec3f scaleFactor;
            float scale[3];
            SbVec3f translation;
            float trans[3];
            SbVec3f center;
            float cen[3];
            SbRotation rotation;
            float rot[4];

            translation = transform->translation.getValue();
            center = transform->center.getValue();
            scaleFactor = transform->scaleFactor.getValue();
            rotation = transform->rotation.getValue();

            scaleFactor.getValue(scale[0], scale[1], scale[2]);
            translation.getValue(trans[0], trans[1], trans[2]);
            center.getValue(cen[0], cen[1], cen[2]);
            rotation.getValue(rot[0], rot[1], rot[2], rot[3]);

            //
            // pack into character string

            sprintf(message, "%s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f",
                    name, scale[0], scale[1], scale[2], trans[0], trans[1], trans[2], cen[0], cen[1], cen[2], rot[0], rot[1], rot[2], rot[3]);

#ifndef YAC
            renderer->cm->sendTransformMessage(message);
#endif
        }
    }
}

//======================================================================
//
// Description:
//	Collect transform change stuff
//
// Use: private
//
//======================================================================
void InvViewer::transformCallback(void *data, SoSensor *)
{
    InvViewer *r = (InvViewer *)data;
    char objName[255];
    int i;
    int num = r->selection->getNumSelected();
    SoTransform *transform = NULL;
    const SoPathList *selectedObjectsList = r->selection->getList();

    // transform all selected objects
    for (i = 0; i < num - 1; i++)
    {
        transform = (SoTransform *)(*(r->transformNode))[i];
        transform->translation.setValue(r->currTransformNode->translation.getValue());
        transform->center.setValue(r->currTransformNode->center.getValue());
        transform->scaleFactor.setValue(r->currTransformNode->scaleFactor.getValue());
        transform->rotation.setValue(r->currTransformNode->rotation.getValue());
    }

    if (renderer->isMaster())
    {
        // print_comment(__LINE__,__FILE__,"RENDERER: transformCallback transform has changed");
        // get the object names from the selected objects
        if (r->currTransformPath != NULL && r->currTransformNode != NULL)
        {
            for (i = 0; i < num; i++)
            {

                r->findObjectName(&objName[0], (*selectedObjectsList)[i]);
                r->sendTransformation(&objName[0], r->currTransformNode);
            }
        }
    }
}

//======================================================================
//
// Description:
//	Collect camera change stuff
//
// Use: private
//
//======================================================================
void InvViewer::cameraCallback(void *data, SoSensor *)
{
    InvViewer *r = (InvViewer *)data;

    if (renderer->isMaster() && renderer->getSyncMode() == InvMain::SYNC_LOOSE)
    {
        // print_comment(__LINE__,__FILE__,"cameraCallback : camera has changed");

        if (r->viewer_edit_state == CO_OFF) // zoom slider or home button...
        {
            //
            //
            // get the current camera and pass the values to the communication
            // manager
            //
            //      char message[100];
            // CAREFULL: this solves the problem only temporary!!!
            char message[255];
            float pos[3];
            float ori[4];
            int view;
            float aspect;
            float mynear;
            float myfar;
            float focal;
            float angleORheightangle; // depends on camera type !

            r->getTransformation(pos, ori, &view, &aspect, &mynear, &myfar, &focal, &angleORheightangle);

            //
            // pack into character string

            sprintf(message, "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %5d %7.3f %7.3f %7.3f %7.3f %7.3f",
                    pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3], view, aspect, mynear, myfar, focal, angleORheightangle);

#ifndef YAC
            renderer->cm->sendCameraMessage(message);
#endif
        }
    }
}

int InvViewer::getBillboardRenderingMethod()
{
    return m_iBillboardRenderingMethod;
}

void InvViewer::setBillboardRenderingMethod(int iMethod)
{
    m_iBillboardRenderingMethod = iMethod;
}

void InvViewer::setBillboardRenderingBlending(bool bBlending)
{
    m_bBillboardRenderingBlending = bBlending;
}

bool InvViewer::getBillboardRenderingBlending()
{
    return m_bBillboardRenderingBlending;
}

//======================================================================
//
// Description:
//    activates/deactivate the frames per second output
//
//======================================================================
void InvViewer::enableFramePerSecondOutputConsole(bool bOn)
{
    m_bFramePerSecondOutputConsole = bOn;
}

//======================================================================
//
// Description:
//    returns the frames per second output status
//
//======================================================================
bool InvViewer::isEnabledFramePerSecondOutputConsole()
{
    return m_bFramePerSecondOutputConsole;
}

//======================================================================
//
// Description:
//    activates/deactivate the process of capturing
//
//======================================================================
void InvViewer::enableRenderWindowCapture(bool bOn)
{
    m_bRenderWindowCapture = bOn;
}

//======================================================================
//
// Description:
//    returns the status of capturing
//    true means capturing is on, false otherwise
//
//======================================================================
bool InvViewer::isEnabledRenderWindowCapture()
{
    return m_bRenderWindowCapture;
}

//======================================================================
//
// Description:
//    sets the window size to capture
//
//======================================================================
void InvViewer::setRenderWindowCaptureSize(int width, int height)
{
    m_pCapture->SetWidth(width);
    m_pCapture->SetHeight(height);
}

//======================================================================
//
// Description:
//    Writes the current Render Aerea as a shnapshot to an image file
//
//======================================================================
void InvViewer::writeRenderWindowSnapshot()
{
    fprintf(stderr, "snap!\n");
    static int iFileNr = 0;
    std::string oldFileName = m_pCapture->GetFileName();
    bool bOldFrameSequence = m_pCapture->GetCaptureFrameSequence();
    char cTemp[10];
    sprintf(cTemp, "%d", iFileNr);
    m_pCapture->SetCaptureFrameSequence(false);
#ifdef _WIN32
    system("if not exist images mkdir images");
    m_pCapture->SetFileName(std::string(getenv("COVISEDIR")).append("\\images\\SnapshotQtRenderer").append(std::string(cTemp)));
#else
    m_pCapture->SetFileName(std::string("SnapshotQtRenderer").append(std::string(cTemp)));
#endif
    if (m_pCapture->Write() == true)
        iFileNr++;
    m_pCapture->SetFileName(oldFileName);
    m_pCapture->SetCaptureFrameSequence(bOldFrameSequence);
}

void InvViewer::snapshotCB(void *userData, SoAction *action)
{
    static float fElapsedTime;
    static double dCurTime;
    static double dLastTime;
    static bool bCurrInit = false;
    static float fFPS = 0.0f;
    static int iSample = 0;
    static timeval now;

    (void)userData;
    if (action->isOfType(SoGLRenderAction::getClassTypeId()))
    {
        if (m_pCapture != NULL)
        {
            if (m_bRenderWindowCapture == true)
            {
                m_pCapture->Write();
            }
            else if (m_bFramePerSecondOutputConsole == true)
            {
                if (bCurrInit == false)
                {
                    gettimeofday(&now, NULL);
                    dCurTime = ((float)now.tv_sec + (float)now.tv_usec) / 1000000.0;
                    dLastTime = dCurTime;
                    fElapsedTime = 0.0f;
                    bCurrInit = true;
                }
                gettimeofday(&now, NULL);
                dCurTime = ((float)now.tv_sec + (float)now.tv_usec) / 1000000.0;

                if ((float)((dCurTime - dLastTime) * 0.001 < 1))
                {
                    fElapsedTime += (float)((dCurTime - dLastTime) * 0.001);
                    iSample++;
                }
                else
                {
                    fElapsedTime = (float)((dCurTime - dLastTime) * 0.001);

                    dLastTime = dCurTime;
                    fFPS = 1.0 / (fElapsedTime / iSample);
                    fElapsedTime = 0.0f;
                    iSample = 0;

                    printf("fps: %04.2f\n", fFPS);
                }
            }
        }

        // Invalidate the state so that a cache is not made
        SoCacheElement::invalidate(action->getState());
    }
}
