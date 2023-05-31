/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: ParametricCurves plugin                                   **
 **              for Cyberclassroom mathematics                            **
 **                                                                        **
 ** cpp file                                                               **
 ** Author: A.Cyran                                                        **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include "cover/VRSceneGraph.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <osg/Switch>
#include <config/CoviseConfig.h>
#include <osg/Shape>
#include "HfT_osg_Parametric_Surface.h"
#include "HfT_osg_Sphere.h"
#include "HfT_osg_Plane.h"
#include "HfT_osg_Animation.h"
#include "HfT_osg_MobiusStrip.h"
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <grmsg/coGRKeyWordMsg.h>

#include "cover/coTranslator.h"

#include "ParametricCurves.h"

using namespace osg;
using namespace covise;
using namespace grmsg;

//---------------------------------------------------
//Implements ParametricCurves *plugin variable
//---------------------------------------------------
ParametricCurves *ParametricCurves::plugin = NULL;

//---------------------------------------------------
//Implements ParametricCurves::ParametricCurves()
//---------------------------------------------------
ParametricCurves::ParametricCurves()
: coVRPlugin(COVER_PLUGIN_NAME)
, m_presentationStepCounter(0)
, m_numPresentationSteps(0)
, m_presentationStep(0)
, m_sliderValueU(0.0)
, m_sliderValueV(0.0)
, m_animSphereRadian(0.0)
, m_rpRootNodeSwitch(NULL)
, m_pSphere(NULL)
, m_pSphereSecond(NULL)
, m_pPlane(NULL)
, m_pMobius(NULL)
, m_pAnimation(NULL)
, m_pAnimationRev(NULL)
, m_pObjectMenu(NULL)
, m_pSliderMenuU(NULL)
, m_pSliderMenuV(NULL)
{
}

//---------------------------------------------------
//Implements ParametricCurves::~ParametricCurves()
//---------------------------------------------------
ParametricCurves::~ParametricCurves()
{
    m_presentationStepCounter = 0;
    m_numPresentationSteps = 0;
    m_presentationStep = 0;
    m_sliderValueU = 0.0;
    m_sliderValueV = 0.0;
    m_animSphereRadian = 0.0;

    if (m_pSphere != NULL)
    {
        //Cleanup the sphere using the pointer, which points at the object
        //Calls destructor of the sphere
        delete m_pSphere;
    }

    if (m_pSphereSecond != NULL)
    {
        //Cleanup the second sphere using the pointer, which points at the object
        //Calls destructor of the second sphere
        delete m_pSphereSecond;
    }

    if (m_pPlane != NULL)
    {
        //Cleanup the plane using the pointer, which points at the object
        //Calls destructor of the plane
        delete m_pPlane;
    }

    if (m_pMobius != NULL)
    {
        //Cleanup the mobius strip using the pointer, which points at the object
        //Calls destructor of the mobius strip
        delete m_pMobius;
    }

    if (m_pAnimation != NULL)
    {
        //Cleanup the animation object using the pointer, which points
        //at the animated sphere object
        //Calls destructor of the animation object
        delete m_pAnimation;
    }

    if (m_pAnimationRev != NULL)
    {
        //Cleanup the animation object using the pointer, which points
        //at the animated sphere object
        //Calls destructor of the animation object
        delete m_pAnimationRev;
    }

    if (m_pObjectMenu != NULL)
    {
        //Cleanup the object menu using the pointer,
        //which points at the object menu object
        //Calls destructor of the coRowMenu object
        delete m_pObjectMenu;
    }

    if (m_pSliderMenuU != NULL)
    {
        //Cleanup the slider menu using the pointer,
        //which points at the slider menu object.
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuU;
    }

    if (m_pSliderMenuV != NULL)
    {
        //Cleanup the slider menu object using the pointer,
        //which points, at the slider menu object
        //Calls destructor of the coSliderMenuItem object
        delete m_pSliderMenuV;
    }
    //Removes the root node from the scene graph
    cover->getObjectsRoot()->removeChild(m_rpRootNodeSwitch.get());
}

//---------------------------------------------------
//Implements ParametricCurves::init()
//---------------------------------------------------
bool ParametricCurves::init()
{
    if (plugin)
        return false;

    //Set plugin
    ParametricCurves::plugin = this;

    //Sets the possible number of presentation steps
    m_numPresentationSteps = 9;
    m_sliderValueU = 0.0;
    m_sliderValueV = 0.0;

    //Initialization of the switch node as root
    m_rpRootNodeSwitch = new Switch;
    m_rpRootNodeSwitch->setName("SwitchNodeForSurfaces");

    //Add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(m_rpRootNodeSwitch.get());

    //Calls private method to initialize the surfaces
    this->initializeSurfaces();

    //Create tree
    //Set transform nodes as children of rootNodeSwitch
    //0   Sphere
    //1   Second sphere
    //2   Plane
    //3   MobiusStrip
    //4   Animated sphere
    //5   Animated sphere with reverse path
    //Third parameter of insertChild() is a bool value which coordinates
    //the visibility of the subtree
    m_rpRootNodeSwitch->insertChild(0, m_pSphere->m_rpTrafoNode.get(), false);
    m_rpRootNodeSwitch->insertChild(1, m_pSphereSecond->m_rpTrafoNode.get(), false);
    m_rpRootNodeSwitch->insertChild(2, m_pPlane->m_rpTrafoNode.get(), false);
    m_rpRootNodeSwitch->insertChild(3, m_pMobius->m_rpTrafoNode.get(), false);
    m_rpRootNodeSwitch->insertChild(4, m_pAnimation->m_rpAnimTrafoNode.get(), false);
    m_rpRootNodeSwitch->insertChild(5, m_pAnimationRev->m_rpAnimTrafoNode.get(), false);

    //as default a plane is visible
    m_rpRootNodeSwitch->setValue(2, true);

    //menu to move the u and v parameter lines
    createMenu();

    //zoom scene
    //attention: never use if the scene is empty!!
    VRSceneGraph::instance()->viewAll();
    return true;
}

//---------------------------------------------------
//Implements ParametricCurves::initializeSurfaces()
//---------------------------------------------------
void ParametricCurves::initializeSurfaces()
{
    //Main sphere
    m_pSphere = new HfT_osg_Sphere(1.0, 100, 100, -PI / 1, PI / 1, -(PI / 2), PI / 2, 0);
    m_pSphere->createSurface();

    //Wireframe sphere, depends on main sphere
    m_pSphereSecond = new HfT_osg_Sphere((m_pSphere->Radian()) + 0.003, 50,
                                         50, m_pSphere->LowerBoundU(),
                                         m_pSphere->UpperBoundU(), m_pSphere->LowerBoundV(),
                                         m_pSphere->UpperBoundV(), 2);
    m_pSphereSecond->createSurface();

    //Plane
    m_pPlane = new HfT_osg_Plane(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0),
                                 5, 5, 0.0, 1.0, 0.0, 1.0, 5);
    m_pPlane->createSurface();

    //Mobius strip
    //Creation in equator mode, in order that the animation recieves the equator edges
    //HAS TO BE the same Mobius strip as in presentation step 7 and 8!!!!!!!!!!!
    m_pMobius = new HfT_osg_MobiusStrip(1.0, 300, 300, 0.0, 2 * PI, -0.75, 0.75, 4);
    m_pMobius->createSurface();

    //Animation
    //A sphere around the equator of the mobius strip, color is yellow
    m_animSphereRadian = 0.2;
    m_pAnimation = new HfT_osg_Animation(*(new Sphere(Vec3(0.0, 0.0, 0.0), m_animSphereRadian)),
                                         *m_pMobius, Vec4(1.0, 1.0, 0.0, 1.0), 'E');

    //Animation
    //A second sphere around the equator of the mobius strip in the reverse direction, color is aquamarin
    m_pAnimationRev = new HfT_osg_Animation(*(new Sphere(Vec3(0.0, 0.0, 0.0), m_animSphereRadian)),
                                            *m_pMobius, Vec4(0.0, 1.0, 1.0, 1.0), 'E');
    m_pAnimationRev->setReverseAnimationPath();
}

//---------------------------------------------------
//Implements ParametricCurves::createMenu()
//---------------------------------------------------
void ParametricCurves::createMenu()
{
    m_pObjectMenu = new coRowMenu("U und V Parameterlinien");
    m_pObjectMenu->setVisible(false);
    m_pObjectMenu->setAttachment(coUIElement::RIGHT);

    //matrices to position the menu
    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;

    // position menu with values from config file
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600);
    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.ParametricCurves.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.ParametricCurves.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.ParametricCurves.MenuPosition", pz);
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("s", "COVER.Plugin.ParametricCurves.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    m_pObjectMenu->setTransformMatrix(&matrix);
    m_pObjectMenu->setScale(cover->getSceneSize() / 2500);

    //Add slider for u parameter to the menu
    m_pSliderMenuU = new coSliderMenuItem(coTranslator::coTranslate("Bereich des U Parameters"), -1.0, 1.0, 0.0);
    m_pSliderMenuU->setMenuListener(this);
    m_pObjectMenu->add(m_pSliderMenuU);
    //m_pSliderMenuU -> setInteger(true);
    //m_pSliderMenuU -> setDiscrete(m_pPlane -> PatchesU());

    //Add slider for v parameter to the menu
    m_pSliderMenuV = new coSliderMenuItem(coTranslator::coTranslate("Bereich des V Parameters"), -1.0, 1.0, 0.0);
    m_pSliderMenuV->setMenuListener(this);
    m_pObjectMenu->add(m_pSliderMenuV);
    //m_pSliderMenuV -> setInteger(true);
    //m_pSliderMenuV -> setDiscrete(m_pPlane -> PatchesV());
}

//---------------------------------------------------
//Implements ParametricCurves::preFrame()
//---------------------------------------------------
void ParametricCurves::preFrame()
{
    m_pPlane->surfacePreframe();
    m_pSphere->surfacePreframe();
    m_pSphereSecond->surfacePreframe();
    m_pMobius->surfacePreframe();
}

//---------------------------------------------------
//Implements ParametricCurves::changePresentationStep()
//---------------------------------------------------
void ParametricCurves::changePresentationStep()
{
    int tempStepCounter = m_presentationStepCounter;

    //If the counter has a negative value add the number of presentation steps as
    //long as the counter is positive.
    if (m_presentationStepCounter < 0)
    {
        do
        {
            tempStepCounter = tempStepCounter + m_numPresentationSteps;
        } while (tempStepCounter < 0);
    }
    //Computation of the presentation step by modulo calculation
    m_presentationStep = (int)mod((double)tempStepCounter, m_numPresentationSteps);

    //Definition of the presentation steps
    switch (m_presentationStep)
    {
    //Plane in wireframe as quads
    case 0:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, false);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, true);
        m_rpRootNodeSwitch->setValue(3, false);
        m_rpRootNodeSwitch->setValue(4, false);
        m_rpRootNodeSwitch->setValue(5, false);
        m_pPlane->setBoundriesPatchesAndMode(0.0, 1.0, 0.0, 1.0, 5, 5, 5);
        m_pPlane->setLabels(true);
        setMenuVisible(false);
        break;

    //Plane in wireframe as quads
    case 1:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, false);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, true);
        m_rpRootNodeSwitch->setValue(3, false);
        m_rpRootNodeSwitch->setValue(4, false);
        m_rpRootNodeSwitch->setValue(5, false);
        m_pPlane->setBoundriesPatchesAndMode(0.0, 1.0, 0.0, 1.0, 5, 5, 5);
        m_pPlane->setLabels(true);
        setMenuVisible(false);
        break;

    //Plane in wireframe as quads with moveable directrices
    case 2:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, false);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, true);
        m_rpRootNodeSwitch->setValue(3, false);
        m_rpRootNodeSwitch->setValue(4, false);
        m_rpRootNodeSwitch->setValue(5, false);
        m_pPlane->setBoundriesPatchesAndMode(0.0, 1.0, 0.0, 1.0, 10, 10, 3);
        //Settings for the slider menu
        m_pSliderMenuU->setMin(m_pPlane->LowerBoundU());
        m_pSliderMenuU->setMax(m_pPlane->UpperBoundU());
        m_pSliderMenuV->setMin(m_pPlane->LowerBoundV());
        m_pSliderMenuV->setMax(m_pPlane->UpperBoundV());
        //Two decimal places
        m_pSliderMenuU->setPrecision(2);
        m_pSliderMenuV->setPrecision(2);
        m_pSliderMenuU->setValue(m_pPlane->LowerBoundU());
        m_pSliderMenuV->setValue(m_pPlane->LowerBoundV());
        //Beginning value of the slider menu
        m_sliderValueU = m_pSliderMenuU->getValue();
        m_sliderValueV = m_pSliderMenuV->getValue();
        m_pPlane->setLabels(false);
        setMenuVisible(true);
        break;

    //Sphere with a globus as texture
    //Second sphere to show the wireframe as quads
    case 3:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, true);
        m_rpRootNodeSwitch->setValue(1, true);
        m_rpRootNodeSwitch->setValue(2, false);
        m_rpRootNodeSwitch->setValue(3, false);
        m_rpRootNodeSwitch->setValue(4, false);
        m_rpRootNodeSwitch->setValue(5, false);
        m_pSphere->setPatchesAndMode(100, 100, 5);
        m_pPlane->setLabels(false);
        setMenuVisible(false);
        break;

    //Sphere in wireframe as quads with moveable directrices
    case 4:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, true);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, false);
        m_rpRootNodeSwitch->setValue(3, false);
        m_rpRootNodeSwitch->setValue(4, false);
        m_rpRootNodeSwitch->setValue(5, false);
        m_pSphere->setPatchesAndMode(40, 40, 3);
        m_pSliderMenuU->setMin(m_pSphere->LowerBoundU());
        m_pSliderMenuU->setMax(m_pSphere->UpperBoundU());
        m_pSliderMenuV->setMin(m_pSphere->LowerBoundV());
        m_pSliderMenuV->setMax(m_pSphere->UpperBoundV());
        m_pSliderMenuU->setPrecision(2);
        m_pSliderMenuV->setPrecision(2);
        m_pSliderMenuU->setValue(m_pSphere->LowerBoundU());
        m_pSliderMenuV->setValue((m_pSphere->UpperBoundV() - m_pSphere->UpperBoundV()) / 2);
        //Beginning value of the slider menu
        m_sliderValueU = m_pSliderMenuU->getValue();
        m_sliderValueV = m_pSliderMenuV->getValue();
        m_pPlane->setLabels(false);
        setMenuVisible(true);
        break;

    //Mobius strip with VISENSO logo as texture
    case 5:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, false);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, false);
        m_rpRootNodeSwitch->setValue(3, true);
        m_rpRootNodeSwitch->setValue(4, false);
        m_rpRootNodeSwitch->setValue(5, false);
        m_pMobius->setPatchesAndMode(200, 200, 5);
        m_pPlane->setLabels(false);
        setMenuVisible(false);
        break;

    //Mobius strip in wireframe as quads with moveable directrices
    case 6:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, false);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, false);
        m_rpRootNodeSwitch->setValue(3, true);
        m_rpRootNodeSwitch->setValue(4, false);
        m_rpRootNodeSwitch->setValue(5, false);
        m_pMobius->setPatchesAndMode(50, 50, 3);
        m_pSliderMenuU->setMin(m_pMobius->LowerBoundU());
        m_pSliderMenuU->setMax(m_pMobius->UpperBoundU());
        m_pSliderMenuV->setMin(m_pMobius->LowerBoundV());
        m_pSliderMenuV->setMax(m_pMobius->UpperBoundV());
        m_pSliderMenuU->setPrecision(2);
        m_pSliderMenuV->setPrecision(2);
        m_pSliderMenuU->setValue(m_pMobius->LowerBoundU());
        m_pSliderMenuV->setValue(m_pMobius->LowerBoundV());
        //Beginning value of the slider menu
        m_sliderValueU = m_pSliderMenuU->getValue();
        m_sliderValueV = m_pSliderMenuV->getValue();
        m_pPlane->setLabels(false);
        setMenuVisible(true);
        break;

    //Mobius strip animation with osg::sphere
    case 7:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, false);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, false);
        m_rpRootNodeSwitch->setValue(3, true);
        m_rpRootNodeSwitch->setValue(4, true);
        m_rpRootNodeSwitch->setValue(5, true);
        m_pMobius->setPatchesAndMode(100, 100, 4);
        m_pPlane->setLabels(false);
        setMenuVisible(false);
        break;

    //Mobius strip animation with osg::sphere
    case 8:
        //fprintf(stderr,"PreFrame Schritt:\t %i \n", m_presentationStep);
        m_rpRootNodeSwitch->setValue(0, false);
        m_rpRootNodeSwitch->setValue(1, false);
        m_rpRootNodeSwitch->setValue(2, false);
        m_rpRootNodeSwitch->setValue(3, true);
        m_rpRootNodeSwitch->setValue(4, true);
        m_rpRootNodeSwitch->setValue(5, true);
        m_pPlane->setLabels(false);
        m_pMobius->setPatchesAndMode(200, 200, 5);
        setMenuVisible(false);
        break;
    }
}

//---------------------------------------------------
//Implements ParametricCurves::guiToRenderMsg(const grmsg::coGRMsg &msg) 
//---------------------------------------------------
void ParametricCurves::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- ParametricCurvesPlugin coVRGuiToRenderMsg %s\n", msg.getString().c_str());

    if (msg.isValid() && msg.getType() == coGRMsg::KEYWORD)
    {
        auto &keyWordMsg = msg.as<coGRKeyWordMsg>();
        const char *keyword = keyWordMsg.getKeyWord();

        // fprintf(stderr,"\tcoGRMsg::KEYWORD keyword=%s\n", keyword);

        // Forward button pressed
        if (strcmp(keyword, "presForward") == 0)
        {
            // fprintf(stderr, "presForward\n");
            m_presentationStepCounter++;
            this->changePresentationStep();
        }
        //Backward button pressed
        else if (strcmp(keyword, "presBackward") == 0)
        {
            //fprintf(stderr, "presBackward\n");
            m_presentationStepCounter--;
            this->changePresentationStep();
        }
        else if (strncmp(keyword, "goToStep", 8) == 0)
        {
            //fprintf(stderr,"\n--- ParametricCurvesPlugin coVRGuiToRenderMsg keyword %s\n", msg);
            stringstream sstream(keyword);
            string sub;
            int step;
            sstream >> sub;
            sstream >> step;
            //fprintf(stderr, "goToStep %d\n", step);
            //fprintf(stderr,"VRMoleculeViewer::message for %s\n", chbuf );
            m_presentationStepCounter = step;
            //fprintf(stderr, "PresStepCounter %d\n", m_presentationStepCounter);
            this->changePresentationStep();
        }
        else if (strcmp(keyword, "reload") == 0)
        {
            //fprintf(stderr, "reload\n");
            VRSceneGraph::instance()->viewAll();
        }
    }
}

////---------------------------------------------------
////Implements ParametricCurves::message(int , int , const void *data)
////---------------------------------------------------
//void ParametricCurves::message(int , int , const void *iData){
//   const char *message = (const char *)iData;
//
//   //fprintf(stderr,"ParametricCurves::message %s\n", message );
//
//   //string comparison
//   //Forward button pressed
//   if(strncmp(message,"presentationForward",strlen("presentationForward"))==0){
//      //fprintf(stderr,"ParametricCurves::message for %s\n", message );
//      m_presentationStepCounter++;
//      this -> changePresentationStep();
//    }
//
//   //Backward button pressed
//   else if(strncmp(message,"presentationBackward",strlen("presentationBackward"))==0){
//           //fprintf(stderr,"ParametricCurves::message for %s\n", message );
//           m_presentationStepCounter--;
//           this -> changePresentationStep();
//   }
//
//   //Reload button pressed
//   else if(strncmp(message,"presentationReload",strlen("presentationReload"))==0){
//           //fprintf(stderr,"ParametricCurves::message for %s\n", message );
//           VRSceneGraph::instance()->viewAll();
//   }
//}

//---------------------------------------------------
//Implements ParametricCurves::menuEvent(coMenuItem *iMenuItem)
//---------------------------------------------------
void ParametricCurves::menuEvent(coMenuItem *iMenuItem)
{
    //Variables to store the computed directrix number.
    int directrixUNumber = 0;
    int directrixVNumber = 0;

    if (iMenuItem == m_pSliderMenuU)
    {
        m_sliderValueU = m_pSliderMenuU->getValue();
        switch (m_presentationStep)
        {
        //plane
        case 2:
            //Interval with the u parameters, but the directrix is in v direction.
            //Remove primitive set to add a new one with new drawElements
            //to store the new edges.
            m_pPlane->m_rpDirectrixVGeom->removePrimitiveSet(0);
            m_pPlane->m_rpDirectrixVEdges = new DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
            m_pPlane->m_rpDirectrixVGeom->addPrimitiveSet(m_pPlane->m_rpDirectrixVEdges);
            //ab hier anders!!!
            directrixVNumber = m_pPlane->computeDirectrixNumber('U', m_sliderValueU);
            //
            m_pPlane->computeDirectrixV(directrixVNumber);
            break;
        //sphere
        case 4:
            m_pSphere->m_rpDirectrixVGeom->removePrimitiveSet(0);
            m_pSphere->m_rpDirectrixVEdges = new DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
            m_pSphere->m_rpDirectrixVGeom->addPrimitiveSet(m_pSphere->m_rpDirectrixVEdges);
            directrixVNumber = m_pSphere->computeDirectrixNumber('U', m_sliderValueU);
            m_pSphere->computeDirectrixV(directrixVNumber);
            break;
        //mobius strip
        case 6:
            m_pMobius->m_rpDirectrixVGeom->removePrimitiveSet(0);
            m_pMobius->m_rpDirectrixVEdges = new DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
            m_pMobius->m_rpDirectrixVGeom->addPrimitiveSet(m_pMobius->m_rpDirectrixVEdges);
            directrixVNumber = m_pMobius->computeDirectrixNumber('U', m_sliderValueU);
            m_pMobius->computeDirectrixV(directrixVNumber);
            break;
        }
    }

    if (iMenuItem == m_pSliderMenuV)
    {
        m_sliderValueV = m_pSliderMenuV->getValue();
        switch (m_presentationStep)
        {
        //plane
        case 2:
            //Interval with the v parameters, but the directrix is in u direction.
            //Remove primitive set to add a new one with new drawElements
            //to store the new edges.
            m_pPlane->m_rpDirectrixUGeom->removePrimitiveSet(0);
            m_pPlane->m_rpDirectrixUEdges = new DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
            m_pPlane->m_rpDirectrixUGeom->addPrimitiveSet(m_pPlane->m_rpDirectrixUEdges);
            //Interval with the u parameters, but the directrix is in u direction
            directrixUNumber = m_pPlane->computeDirectrixNumber('V', m_sliderValueV);
            m_pPlane->computeDirectrixU(directrixUNumber);
            break;
        //sphere
        case 4:
            m_pSphere->m_rpDirectrixUGeom->removePrimitiveSet(0);
            m_pSphere->m_rpDirectrixUEdges = new DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
            m_pSphere->m_rpDirectrixUGeom->addPrimitiveSet(m_pSphere->m_rpDirectrixUEdges);
            directrixUNumber = m_pSphere->computeDirectrixNumber('V', m_sliderValueV);
            m_pSphere->computeDirectrixU(directrixUNumber);
            break;
        //mobius strip
        case 6:
            //fprintf(stderr, "Directrix verändert \n");
            m_pMobius->m_rpDirectrixUGeom->removePrimitiveSet(0);
            m_pMobius->m_rpDirectrixUEdges = new DrawElementsUInt(osg::PrimitiveSet::LINE_STRIP, 0);
            m_pMobius->m_rpDirectrixUGeom->addPrimitiveSet(m_pMobius->m_rpDirectrixUEdges);
            directrixUNumber = m_pMobius->computeDirectrixNumber('V', m_sliderValueV);
            m_pMobius->computeDirectrixU(directrixUNumber);
            break;
        }
    }
}

void ParametricCurves::setMenuVisible(bool visible)
{
    m_pObjectMenu->setVisible(visible);
    VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
}

COVERPLUGIN(ParametricCurves)
