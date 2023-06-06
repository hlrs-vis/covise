/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: BezierCurvePluginv                                             **
 **              for Cyberclassroom mathematics                            **
 **                                                                        **
 ** Author: T.Milbich                                                      **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <config/CoviseConfig.h>
#include <grmsg/coGRKeyWordMsg.h>

#include "cover/coTranslator.h"

#include "BezierCurvePlugin.h"

const int MAX_POINTS = 10;

using namespace osg;
using namespace grmsg;
using covise::coCoviseConfig;

BezierCurvePlugin *BezierCurvePlugin::plugin = NULL;

//
// Constructor
//
BezierCurvePlugin::BezierCurvePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, node_(NULL)
, plane_(NULL)
, planeGeode_(NULL)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nBezierCurvePlugin::BezierCurvePlugin\n");
}

//
// Destructor
//
BezierCurvePlugin::~BezierCurvePlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nBezierCurvePlugin::~BezierCurvePlugin\n");
}

bool BezierCurvePlugin::destroy()
{
    // allen angelegten Speicher freimachen

    delete objectsMenu;
    delete menuItemObjectsAddPoint;
    delete menuItemObjectsRemovePoint;
    delete menuItemAnimateCasteljau;
    delete menuItemShowCasteljau;
    delete menuItemdegreeElevation;
    delete menuItemdegreeReductionForest;
    delete menuItemdegreeReductionFarin;
    delete menuItemshowTangents;
    delete menuItemParameterValue;

    while (controlPoints.size() > 0)
    {
        removePoint();
    }
    delete casteljauLabel;
    delete curve;

    cover->getObjectsRoot()->removeChild(node_.get());
    return true;
}

//
// INIT
//
bool BezierCurvePlugin::init()
{
    if (plugin)
        return false;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nBezierCurvePlugin::BezierCurvePlugin\n");

    // set plugin
    BezierCurvePlugin::plugin = this;

    // rotate scene so that x axis is in front
    //   MatrixTransform* trafo = VRSceneGraph::instance()->getTransform();
    //   Matrix m;
    //   m.makeRotate(inDegrees(-90.0), 0.0, 0.0, 1.0);
    //   trafo->setMatrix(m);

    // root node
    node_ = new MatrixTransform();
    node_->ref();
    node_->setName("BezierPlugin");

    //Initialize member variables
    parameterValueAnimation = 0;
    parameterValueStep = 0.5;
    computation = BezierCurveVisualizer::APROXIMATION;
    curve = new BezierCurveVisualizer(node_, computation);
    showCasteljauAnimation = false;
    showCasteljauStep = false;
    showTangents = false;
    firstLoad = true;
    scale = cover->getScale();
    presentationStepCounter = 0;
    objectsMenu = NULL;
    menuItemObjectsAddPoint = NULL;
    menuItemObjectsRemovePoint = NULL;
    menuItemAnimateCasteljau = NULL;
    menuItemShowCasteljau = NULL;
    menuItemdegreeElevation = NULL;
    menuItemdegreeReductionForest = NULL;
    menuItemdegreeReductionFarin = NULL;
    menuItemshowTangents = NULL;
    menuItemParameterValue = NULL;
    menuItemSeparator1 = NULL;
    menuItemSeparator2 = NULL;

    createMenu();
    removeMenuEntries();
    makeCurve();

    firstLoad = false;

    casteljauLabel = new coVRLabel("", 30, 100.0, osg::Vec4(0.5451, 0.7020, 0.2431, 1.0), osg::Vec4(0.0, 0.0, 0.0, 0.8));
    casteljauLabel->hide();

    // zoom scene
    //   VRSceneGraph::instance()->viewAll();

    // add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(node_.get());

    return true;
}

//----------------------------------------------------------------------
void BezierCurvePlugin::makeCurve()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "BezierCurvePlugin::makeCurve\n");

    controlPoints.push_back(new Point(Vec3(-600, 0, -400)));
    controlPoints.push_back(new Point(Vec3(-600, 0, 200)));
    controlPoints.push_back(new Point(Vec3(600, 0, 200)));
    controlPoints.push_back(new Point(Vec3(600, 0, -400)));

    initializeCurve();
}

//--------------------------------------------------------------------
void BezierCurvePlugin::initializeCurve()
{
    curve->showControlPolygon(true);
    curve->showCurve(true);
    //	curve->showCasteljau(true);
    //	curve->setT(0.5);

    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        curve->addControlPoint((controlPoints[i]->getPosition()));
    }
}

//--------------------------------------------------------------------
void BezierCurvePlugin::addNewPoint()
{
    Point *newPoint = new Point(Vec3(0, 0, 0));
    addNewPoint(newPoint);
}

//---------------------------------------------------------------------
void BezierCurvePlugin::addNewPoint(Point *newPoint)
{
    if (controlPoints.size() < MAX_POINTS)
    {
        controlPoints.push_back(newPoint);
        curve->addControlPoint(controlPoints[controlPoints.size() - 1]->getPosition());
        curve->updateGeometry();
    }
    else
    {
        delete newPoint;
        fprintf(stderr, "\n------------- Sie können nicht mehr als 10 Kontrollpunkte anlegen.\n");
    }
}

//---------------------------------------------------------------------
void BezierCurvePlugin::removePoint()
{
    //entfernt letzten Punkt in der Liste
    std::vector<Point *>::iterator lastPoint = controlPoints.end() - 1;
    curve->removeControlPoint();
    delete *lastPoint;
    controlPoints.erase(lastPoint);

    curve->updateGeometry();
}

//----------------------------------------------------------------------
void BezierCurvePlugin::preFrame()
{
    //fprintf(stderr,"BezierPlugin::preFrame\n");

    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        controlPoints[i]->preFrame();
        curve->changeControlPoint(controlPoints[i]->getPosition(), i);
    }

    curve->showTangentBegin(showTangents);
    curve->showTangentEnd(showTangents);

    if (showCasteljauAnimation)
    {
        casteljauAnimation();
        casteljauLabel->show();
    }
    else if (showCasteljauStep)
    {
        casteljauStep();
        casteljauLabel->show();
    }
    else
    {
        casteljauLabel->hide();
        setParameterValueAnimation(0);
        setParameterValueStep(0.5);
    }

    curve->updateGeometry();
}

//-----------------------------------------------------------------------
void BezierCurvePlugin::casteljauAnimation()
{
    parameterValueAnimation += 0.0025f;

    if (parameterValueAnimation > 1 || parameterValueAnimation < 0)
        parameterValueAnimation = 0;

    curve->setT(parameterValueAnimation);

    char buffer[50];
    sprintf(buffer, "t = %.2f", parameterValueAnimation);

    Vec3 casteljauPoint = curve->computePointOnCurve(parameterValueAnimation);
    casteljauLabel->setString(buffer);
    casteljauLabel->setPosition(casteljauPoint * cover->getBaseMat());
}

//-----------------------------------------------------------------------
void BezierCurvePlugin::casteljauStep()
{
    parameterValueStep = (menuItemParameterValue->getValue());

    if (parameterValueStep > 1 || parameterValueStep < 0)
        parameterValueStep = 0;

    curve->setT(parameterValueStep);

    char buffer[50];
    sprintf(buffer, "t = %.2f", parameterValueStep);

    Vec3 casteljauPoint = curve->computePointOnCurve(parameterValueStep);
    casteljauLabel->setString(buffer);
    casteljauLabel->setPosition(casteljauPoint * cover->getBaseMat());
}

//-----------------------------------------------------------------------
void BezierCurvePlugin::setParameterValueStep(float pv)
{
    if (pv >= 0 && pv <= 1)
    {
        parameterValueStep = pv;
    }
    else
    {
        fprintf(stderr, "\n---------------------- neuer Parameterwert liegt nicht zwischen 0 und 1, setze Wert auf 0!");
        parameterValueStep = 0;
    }
}

//-----------------------------------------------------------------------
void BezierCurvePlugin::setParameterValueAnimation(float pv)
{
    if (pv >= 0 && pv <= 1)
    {
        parameterValueAnimation = pv;
    }
    else
    {
        fprintf(stderr, "\n---------------------- neuer Parameterwert liegt nicht zwischen 0 und 1, setze Wert auf 0!");
        parameterValueAnimation = 0;
    }
}

//------------------------------------------------------------------------
void BezierCurvePlugin::elevateDegreeOfCurve()
{
    if (controlPoints.size() < 3)
    {
        fprintf(stderr, "\n------------ Die Graderhöhung von Bezierkurven funktioniert erst ab Grad 2 \n");
        return;
    }

    if (controlPoints.size() < MAX_POINTS)
    {
        //speichere alte Kontrollpunkte
        int size = controlPoints.size();
        std::vector<Vec3> tmpVec;
        for (int i = 0; i < size; i++)
        {
            tmpVec.push_back(controlPoints[i]->getPosition());
        }

        //lösche Kontrollpunkte.
        //Achtung: es werden auch die Kontrollpunkte des BezierCurveVisualizers gelöscht!
        for (int i = 0; i < size; i++)
        {
            removePoint();
        }

        //füge alte Kontrollpunkte wieder in den BezierCurveVisualizer ein
        for (int i = 0; i < size; i++)
        {
            curve->addControlPoint(tmpVec[i]);
        }

        //Führe Graderhöhung der Kurve durch
        curve->degreeElevation();

        //Füge neue Punkte ein
        std::vector<Vec3> newPoints = curve->getAllControlPoints();
        for (size_t i = 0; i < newPoints.size(); i++)
        {
            controlPoints.push_back(new Point(newPoints[i]));
        }

        //aktualisiere die Kurve
        curve->updateGeometry();
    }
}

//-----------------------------------------------------------------------
void BezierCurvePlugin::reduceDegreeOfCurveForest()
{
    if (controlPoints.size() < 3)
    {
        fprintf(stderr, "\n------------ Der Grad kann nur bis Grad 1 verringert werden \n");
        return;
    }

    //speichere alte Kontrollpunkte
    int size = controlPoints.size();
    std::vector<Vec3> tmpVec;
    for (int i = 0; i < size; i++)
    {
        tmpVec.push_back(controlPoints[i]->getPosition());
    }

    //lösche Kontrollpunkte.
    //Achtung: es werden auch die Kontrollpunkte des BezierCurveVisualizers gelöscht!
    for (int i = 0; i < size; i++)
    {
        removePoint();
    }

    //füge alte Kontrollpunkte wieder in den BezierCurveVIsualizer ein
    for (int i = 0; i < size; i++)
    {
        curve->addControlPoint(tmpVec[i]);
    }

    //Führe Graderhöhung der Kurve durch
    curve->degreeReductionForest();

    //Füge neue Punkte ein
    std::vector<Vec3> newPoints = curve->getAllControlPoints();
    for (size_t i = 0; i < newPoints.size(); i++)
    {
        controlPoints.push_back(new Point(newPoints[i]));
    }

    //aktualisiere die Kurve
    curve->updateGeometry();
}

//-----------------------------------------------------------------------
void BezierCurvePlugin::reduceDegreeOfCurveFarin()
{
    if (controlPoints.size() < 3)
    {
        fprintf(stderr, "\n------------ Der Grad kann nur bis Grad 1 verringert werden \n");
        return;
    }

    //speichere alte Kontrollpunkte
    int size = controlPoints.size();
    std::vector<Vec3> tmpVec;
    for (int i = 0; i < size; i++)
    {
        tmpVec.push_back(controlPoints[i]->getPosition());
    }

    //lösche Kontrollpunkte.
    //Achtung: es werden auch die Kontrollpunkte des BezierCurveVisualizers gelöscht!
    for (int i = 0; i < size; i++)
    {
        removePoint();
    }

    //füge alte Kontrollpunkte wieder in den BezierCurveVIsualizer ein
    for (int i = 0; i < size; i++)
    {
        curve->addControlPoint(tmpVec[i]);
    }

    //Führe Graderhöhung der Kurve durch
    curve->degreeReductionFarin();

    //Füge neue Punkte ein
    std::vector<Vec3> newPoints = curve->getAllControlPoints();
    for (size_t i = 0; i < newPoints.size(); i++)
    {
        controlPoints.push_back(new Point(newPoints[i]));
    }

    //aktualisiere die Kurve
    curve->updateGeometry();
}

//------------------------------------------------------------------------
void BezierCurvePlugin::createMenu()
{
    if (firstLoad)
    {
        objectsMenu = new coRowMenu("Bezier Menue");
        objectsMenu->setVisible(false);
        objectsMenu->setAttachment(coUIElement::RIGHT);

        // position Menu
        OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;
        double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000);
        double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0);
        double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600);

        px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.BezierCurve.MenuPosition", px);
        py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.BezierCurve.MenuPosition", py);
        pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.BezierCurve.MenuPosition", pz);

        // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
        float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
        s = coCoviseConfig::getFloat("value", "COVER.Plugin.BezierCurve.MenuSize", s);

        transMatrix.makeTranslate(px, py, pz);
        rotateMatrix.makeEuler(0, 90, 0);
        scaleMatrix.makeScale(s, s, s);

        matrix.makeIdentity();
        matrix.mult(&scaleMatrix);
        matrix.mult(&rotateMatrix);
        matrix.mult(&transMatrix);

        objectsMenu->setTransformMatrix(&matrix);
        objectsMenu->setScale(cover->getSceneSize() / 2500);
    }

    if (presentationStepCounter == 0 || presentationStepCounter == 7 || firstLoad)
    {
        setMenuVisible(false);
    }
    else
    {
        setMenuVisible(true);
    }

    if (presentationStepCounter > 0 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemObjectsAddPoint = new coButtonMenuItem(coTranslator::coTranslate("erstelle Kontrollpunkt"));
            menuItemObjectsAddPoint->setMenuListener(this);
        }

        objectsMenu->add(menuItemObjectsAddPoint);

        if (firstLoad)
        {
            menuItemObjectsRemovePoint = new coButtonMenuItem(coTranslator::coTranslate("loesche Kontrollpunkt"));
            menuItemObjectsRemovePoint->setMenuListener(this);
        }

        objectsMenu->add(menuItemObjectsRemovePoint);
    }

    if (presentationStepCounter == 2 || presentationStepCounter == 6 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemshowTangents = new coCheckboxMenuItem(coTranslator::coTranslate("Anf.-/Endtangente"), showTangents);
            menuItemshowTangents->setMenuListener(this);
        }
        menuItemshowTangents->setState(showTangents);

        objectsMenu->add(menuItemshowTangents);
    }

    if (presentationStepCounter == 3 || presentationStepCounter == 6 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemSeparator1 = new coLabelMenuItem("_____________________");
            menuItemSeparator1->setMenuListener(this);
        }

        objectsMenu->add(menuItemSeparator1);

        if (firstLoad)
        {
            menuItemAnimateCasteljau = new coCheckboxMenuItem(coTranslator::coTranslate("Casteljau Animation"), showCasteljauAnimation);
            menuItemAnimateCasteljau->setMenuListener(this);
        }
        menuItemAnimateCasteljau->setState(showCasteljauAnimation);
        objectsMenu->add(menuItemAnimateCasteljau);

        if (firstLoad)
        {
            menuItemShowCasteljau = new coCheckboxMenuItem(coTranslator::coTranslate("Show Casteljau"), showCasteljauStep || showCasteljauAnimation);
            menuItemShowCasteljau->setMenuListener(this);
        }
        menuItemShowCasteljau->setState(showCasteljauStep || showCasteljauAnimation);
        objectsMenu->add(menuItemShowCasteljau);

        if (firstLoad)
        {
            menuItemParameterValue = new coSliderMenuItem(coTranslator::coTranslate("Param. t"), 0.0, 1.0, 0.5);
            menuItemParameterValue->setMenuListener(this);
        }

        objectsMenu->add(menuItemParameterValue);
    }

    if (presentationStepCounter == 6 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemSeparator2 = new coLabelMenuItem("_____________________");
            menuItemSeparator2->setMenuListener(this);
        }

        objectsMenu->add(menuItemSeparator2);
    }

    if (presentationStepCounter == 4 || presentationStepCounter == 5 || presentationStepCounter == 6 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemdegreeElevation = new coButtonMenuItem(coTranslator::coTranslate("Graderhoehung"));
            menuItemdegreeElevation->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeElevation);
    }

    if (presentationStepCounter == 5 || presentationStepCounter == 6 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemdegreeReductionForest = new coButtonMenuItem(coTranslator::coTranslate("Gradereduktion Forest"));
            menuItemdegreeReductionForest->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeReductionForest);

        if (firstLoad)
        {
            menuItemdegreeReductionFarin = new coButtonMenuItem(coTranslator::coTranslate("Gradereduktion Farin"));
            menuItemdegreeReductionFarin->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeReductionFarin);
    }
}

void BezierCurvePlugin::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == menuItemObjectsAddPoint)
    {
        addNewPoint();
    }

    else if (menuItem == menuItemObjectsRemovePoint)
    {
        if (controlPoints.size() > 2)
        {
            removePoint();
        }
        else
        {
            fprintf(stderr, "\n--------------- Eine Bezierkurve benötigt min. 2 Punkte!\n");
        }
    }

    else if (menuItem == menuItemAnimateCasteljau)
    {
        showCasteljauAnimation = menuItemAnimateCasteljau->getState();
        showCasteljauStep = false;
        curve->showCasteljau(showCasteljauAnimation);
        menuItemShowCasteljau->setState(showCasteljauStep || showCasteljauAnimation);
    }

    else if (menuItem == menuItemParameterValue)
    {
        if (showCasteljauStep == false)
        {
            showCasteljauStep = true;
            showCasteljauAnimation = false;
            menuItemAnimateCasteljau->setState(showCasteljauAnimation);
            parameterValueAnimation = 0;
            curve->showCasteljau(true);
        }
        menuItemShowCasteljau->setState(showCasteljauStep || showCasteljauAnimation);
    }

    else if (menuItem == menuItemShowCasteljau)
    {
        if (menuItemShowCasteljau->getState())
        {
            showCasteljauAnimation = false;
            showCasteljauStep = true;
            menuItemAnimateCasteljau->setState(showCasteljauAnimation);
            curve->showCasteljau(showCasteljauStep);
        }
        else
        {
            showCasteljauAnimation = false;
            showCasteljauStep = false;
            menuItemAnimateCasteljau->setState(showCasteljauAnimation);
            curve->showCasteljau(showCasteljauStep);
        }
    }

    else if (menuItem == menuItemdegreeElevation)
    {
        elevateDegreeOfCurve();
    }

    else if (menuItem == menuItemdegreeReductionForest)
    {
        reduceDegreeOfCurveForest();
    }

    else if (menuItem == menuItemdegreeReductionFarin)
    {
        reduceDegreeOfCurveFarin();
    }

    else if (menuItem == menuItemshowTangents)
    {
        showTangents = menuItemshowTangents->getState();
    }
}

//void BezierCurvePlugin::message(int , int , const void *iData){
//   const char *message = (const char *)iData;
//
//   //fprintf(stderr,"ParametricCurves::message %s\n", message );
//
//   //string comparison
//   //Forward button pressed
//   if(strncmp(message,"presentationForward",strlen("presentationForward"))==0){
//      //fprintf(stderr,"ParametricCurves::message for %s\n", message );
//
//	   presentationStepCounter++;
//	   if(presentationStepCounter > 7)
//		   presentationStepCounter = 0;
//
//	   changeMenu();
//	   changeStatus();
//    }
//
//   //Backward button pressed
//   else if(strncmp(message,"presentationBackward",strlen("presentationBackward"))==0){
//           //fprintf(stderr,"ParametricCurves::message for %s\n", message );
//
//	   presentationStepCounter--;
//	   if(presentationStepCounter < 0)
//	  		   presentationStepCounter = 0;
//
//	   changeMenu();
//	   changeStatus();
//   }
//
////   //Reload button pressed
////   else if(strncmp(message,"presentationReload",strlen("presentationReload"))==0){
////           //fprintf(stderr,"ParametricCurves::message for %s\n", message );
////           VRSceneGraph::instance()->viewAll();
////   }
//}

void BezierCurvePlugin::removeMenuEntries()
{
    if (menuItemObjectsAddPoint)
    {
        objectsMenu->remove(menuItemObjectsAddPoint);
    }

    if (menuItemObjectsRemovePoint)
    {
        objectsMenu->remove(menuItemObjectsRemovePoint);
    }

    if (menuItemshowTangents)
    {
        objectsMenu->remove(menuItemshowTangents);
    }

    if (menuItemSeparator1)
    {
        objectsMenu->remove(menuItemSeparator1);
    }

    if (menuItemAnimateCasteljau)
    {
        objectsMenu->remove(menuItemAnimateCasteljau);
    }

    if (menuItemShowCasteljau)
    {
        objectsMenu->remove(menuItemShowCasteljau);
    }

    if (menuItemParameterValue)
    {
        objectsMenu->remove(menuItemParameterValue);
    }

    if (menuItemSeparator2)
    {
        objectsMenu->remove(menuItemSeparator2);
    }

    if (menuItemdegreeElevation)
    {
        objectsMenu->remove(menuItemdegreeElevation);
    }

    if (menuItemdegreeReductionForest)
    {
        objectsMenu->remove(menuItemdegreeReductionForest);
    }

    if (menuItemdegreeReductionFarin)
    {
        objectsMenu->remove(menuItemdegreeReductionFarin);
    }

    if (!firstLoad)
        createMenu();
}

void BezierCurvePlugin::changeStatus()
{
    int size = controlPoints.size();
    for (int i = 0; i < size; i++)
    {
        removePoint();
    }

    makeCurve();

    MatrixTransform *trafo = VRSceneGraph::instance()->getTransform();
    Matrix m;
    m.makeIdentity();
    trafo->setMatrix(m);

    switch (presentationStepCounter)
    {
    case 0:
        curve->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showTangents = false;
        break;

    case 1:
        curve->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showTangents = false;
        break;

    case 2:
        curve->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showTangents = true;
        break;

    case 3:
        setParameterValueAnimation(0);
        setParameterValueStep(0.5);
        curve->showCasteljau(true);
        showCasteljauAnimation = true;
        showCasteljauStep = false;
        showTangents = false;
        break;

    case 4:
        showCasteljauAnimation = false;
        curve->showCasteljau(false);
        showCasteljauStep = false;
        showTangents = false;
        break;

    case 5:
        curve->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showTangents = false;
        break;

    case 6:
        setParameterValueAnimation(0);
        setParameterValueStep(0.5);
        curve->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showTangents = false;
        break;

    case 7:
        curve->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showTangents = false;
        break;
    }
}

void BezierCurvePlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- BezierCurvePlugin coVRGuiToRenderMsg %s\n", msg.getString().c_str());

    if (msg.isValid())
    {
        if (msg.getType() == grmsg::coGRMsg::KEYWORD)
        {
            auto &keyWordMsg = msg.as<grmsg::coGRKeyWordMsg>();
            const char *keyword = keyWordMsg.getKeyWord();
            if (cover->debugLevel(3))
                fprintf(stderr, "\tcoGRMsg::KEYWORD keyword=%s\n", keyword);
            if (strcmp(keyword, "presForward") == 0)
            {

                presentationStepCounter++;
                if (presentationStepCounter > 7)
                    presentationStepCounter = 0;

                changeStatus();
                removeMenuEntries();
            }
            else if (strcmp(keyword, "presBackward") == 0)
            {
                presentationStepCounter--;
                if (presentationStepCounter < 0)
                    presentationStepCounter = 0;

                changeStatus();
                removeMenuEntries();
            }
            else if (strncmp(keyword, "goToStep", 8) == 0)
            {
                stringstream sstream(keyword);
                string sub;
                int step;
                sstream >> sub;
                sstream >> step;
                presentationStepCounter = step;
                changeStatus();
                removeMenuEntries();
            }
        }
    }
}

void BezierCurvePlugin::setMenuVisible(bool visible)
{
    objectsMenu->setVisible(visible);
    VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
}

COVERPLUGIN(BezierCurvePlugin)
