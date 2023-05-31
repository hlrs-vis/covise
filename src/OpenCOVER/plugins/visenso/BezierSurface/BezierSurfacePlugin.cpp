/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: BezierSurfacePluginv                                             **
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
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRNavigationManager.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <config/CoviseConfig.h>
#include <string.h>

#include <grmsg/coGRKeyWordMsg.h>

#include "cover/coTranslator.h"

#include "BezierSurfacePlugin.h"

using namespace osg;
using covise::coCoviseConfig;
using namespace grmsg;

BezierSurfacePlugin *BezierSurfacePlugin::plugin = NULL;

const int MAX_POINTS = 16;

//
// Constructor
//
BezierSurfacePlugin::BezierSurfacePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, node_(NULL)
, plane_(NULL)
, planeGeode_(NULL)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nBezierSurfacePlugin::BezierSurfacePlugin\n");
}

//
// Destructor
//
BezierSurfacePlugin::~BezierSurfacePlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nBezierSurfacePlugin::~BezierSurfacePlugin\n");
    // allen angelegten Speicher freimachen
}
bool BezierSurfacePlugin::destroy()
{
    delete objectsMenu;
    delete menuItemdegreeElevationU;
    delete menuItemdegreeElevationV;
    delete menuItemShowControlPolygon;
    delete menuItemShowInteractors;
    delete menuItemAnimateCasteljau;
    delete menuItemShowCasteljau;
    delete menuItemdegreeReductionForestU;
    delete menuItemdegreeReductionForestV;
    delete menuItemdegreeReductionFarinU;
    delete menuItemdegreeReductionFarinV;
    delete menuItemParameterValueU;
    delete menuItemParameterValueV;
    delete menuItemSeparator1;
    delete menuItemSeparator2;
    delete menuItemSeparator3;

    delete labelU;
    delete labelV;
    delete labelStatus;
    delete casteljauLabel;

    for (std::vector<Point *>::iterator iter = controlPoints.begin(); iter != controlPoints.end(); iter++)
    {
        delete *iter;
    }

    controlPoints.clear();
    surface->removeAllControlPoints();

    delete surface;

    cover->getObjectsRoot()->removeChild(node_.get());

    return true;
}

//
// INIT
//
bool BezierSurfacePlugin::init()
{
    if (plugin)
        return false;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nBezierSurfacePlugin::BezierSurfacePlugin\n");

    // set plugin
    BezierSurfacePlugin::plugin = this;

    // rotate scene so that x axis is in front
    //   MatrixTransform* trafo = VRSceneGraph::instance()->getTransform();
    //   Matrix m;
    //   m.makeRotate(inDegrees(-90.0), 0.0, 0.0, 1.0);
    //   trafo->setMatrix(m);

    // root node
    node_ = new MatrixTransform();
    node_->ref();
    node_->setName("ParametricCurve");
    node_->setNodeMask(node_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    // make the curve
    surface = new BezierSurfaceVisualizer(node_);
    showCasteljauStep = false;
    showCasteljauAnimation = false;
    showControlPolygon = true;
    showInteractors = true;
    firstLoad = true;
    parameterValueAnimationU = 0;
    parameterValueAnimationV = 0;
    presentationStepCounter = 0;
    scale = cover->getScale();
    objectsMenu = NULL;
    menuItemdegreeElevationU = NULL;
    menuItemdegreeElevationV = NULL;
    menuItemShowControlPolygon = NULL;
    menuItemShowInteractors = NULL;
    menuItemAnimateCasteljau = NULL;
    menuItemShowCasteljau = NULL;
    menuItemdegreeReductionForestU = NULL;
    menuItemdegreeReductionForestV = NULL;
    menuItemdegreeReductionFarinU = NULL;
    menuItemdegreeReductionFarinV = NULL;
    menuItemParameterValueU = NULL;
    menuItemParameterValueV = NULL;
    menuItemSeparator1 = NULL;
    menuItemSeparator2 = NULL;

    createMenu();
    removeMenuEntries();
    makeSurface();

    firstLoad = false;

    // labels
    labelU = new coVRLabel("u = 0", 30, 100.0, osg::Vec4(0.5451, 0.7020, 0.2431, 1.0), osg::Vec4(0.0, 0.0, 0.0, 0.8));
    labelV = new coVRLabel("v = 0", 30, 100.0, osg::Vec4(0.5451, 0.7020, 0.2431, 1.0), osg::Vec4(0.0, 0.0, 0.0, 0.8));
    labelStatus = new coVRLabel("", 24, 100.0, osg::Vec4(0.5451, 0.7020, 0.2431, 1.0), osg::Vec4(0.0, 0.0, 0.0, 0.8));
    casteljauLabel = new coVRLabel("", 24, 100.0, osg::Vec4(0.5451, 0.7020, 0.2431, 1.0), osg::Vec4(0.0, 0.0, 0.0, 0.8));
    casteljauLabel->hide();

    // zoom scene
    //   VRSceneGraph::instance()->viewAll();

    // add root node to cover scenegraph
    cover->getObjectsRoot()->addChild(node_.get());

    return true;
}

//----------------------------------------------------------------------
void BezierSurfacePlugin::makeSurface()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "BezierSurfacePlugin::makeCurve\n");

    n = 3;
    m = 3;

    surface->setN(n);
    surface->setM(m);

    controlPoints.push_back(new Point(Vec3(-300, 300, -200)));
    controlPoints.push_back(new Point(Vec3(-300, 0, 0)));
    controlPoints.push_back(new Point(Vec3(-300, -300, -200)));

    controlPoints.push_back(new Point(Vec3(0, 300, 0)));
    controlPoints.push_back(new Point(Vec3(0, 0, 200)));
    controlPoints.push_back(new Point(Vec3(0, -300, 0)));

    controlPoints.push_back(new Point(Vec3(300, 300, -200)));
    controlPoints.push_back(new Point(Vec3(300, 0, 0)));
    controlPoints.push_back(new Point(Vec3(300, -300, -200)));

    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        surface->addControlPoint((controlPoints[i]->getPosition()));
        controlPoints[i]->showInteractor(true);
    }

    surface->showControlPolygon(true);
    surface->showSurface(true);
}

//----------------------------------------------------------------------
void BezierSurfacePlugin::preFrame()
{
    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        controlPoints[i]->preFrame();
        controlPoints[i]->showInteractor(showInteractors);
        surface->changeControlPoint(controlPoints[i]->getPosition(), i);
    }

    surface->showControlPolygon(showControlPolygon);

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
        parameterValueAnimationU = 0;
        parameterValueAnimationV = 0;
        parameterValueStepV = 0.5;
        parameterValueStepU = 0.5;
        casteljauLabel->hide();
    }

    surface->updateGeometry();

    Vec3 labelUPos = surface->computePointOnCurve(0, 0.5);
    labelU->setPosition(labelUPos * cover->getBaseMat());

    Vec3 labelVPos = surface->computePointOnCurve(0.5, 0);
    labelV->setPosition(labelVPos * cover->getBaseMat());

    string status = "";
    ostringstream textStream;
    textStream << coTranslator::coTranslate("Grad v = ");
    textStream << (n - 1);
    textStream << "\n";
    textStream << coTranslator::coTranslate("Grad u = ");
    textStream << (m - 1);
    status.append(textStream.str());

    labelStatus->setString(status.c_str());
    labelStatus->setPosition(controlPoints[0]->getPosition() * cover->getBaseMat());
}

//-----------------------------------------------------------------------
void BezierSurfacePlugin::casteljauStep()
{
    parameterValueStepU = (menuItemParameterValueU->getValue());
    parameterValueStepV = (menuItemParameterValueV->getValue());

    if (parameterValueStepU > 1 || parameterValueStepU < 0)
        parameterValueStepU = 0;

    if (parameterValueStepV > 1 || parameterValueStepV < 0)
        parameterValueStepV = 0;

    surface->setParameterU(parameterValueStepU);
    surface->setParameterV(parameterValueStepV);

    string status = "";
    ostringstream textStream;
    textStream << "u = ";
    textStream << parameterValueStepU;
    textStream << "\n";
    textStream << "v = ";
    textStream << parameterValueStepV;
    status.append(textStream.str());

    Vec3 casteljauPoint = surface->computePointOnCurve(parameterValueStepU, parameterValueStepV);
    casteljauLabel->setString(status.c_str());
    casteljauLabel->setPosition(casteljauPoint * cover->getBaseMat());
}

//-----------------------------------------------------------------------
void BezierSurfacePlugin::casteljauAnimation()
{
    parameterValueAnimationU += 0.0025f;
    parameterValueAnimationV += 0.0025f;

    if (parameterValueAnimationU > 1 || parameterValueAnimationU < 0)
        parameterValueAnimationU = 0;

    if (parameterValueAnimationV > 1 || parameterValueAnimationV < 0)
        parameterValueAnimationV = 0;

    surface->setParameterU(parameterValueAnimationU);
    surface->setParameterV(parameterValueAnimationV);

    string status = "";
    ostringstream textStream;
    textStream << "u = ";
    textStream << parameterValueAnimationU;
    textStream << "\n";
    textStream << "v = ";
    textStream << parameterValueAnimationV;
    status.append(textStream.str());

    Vec3 casteljauPoint = surface->computePointOnCurve(parameterValueAnimationU, parameterValueAnimationV);
    casteljauLabel->setString(status.c_str());
    casteljauLabel->setPosition(casteljauPoint * cover->getBaseMat());
}

//-----------------------------------------------------------------------
void BezierSurfacePlugin::elevateDegreeOfSurface(char direction)
{

    switch (direction)
    {
    case 'u':
    {
        if (m > 4)
        {
            fprintf(stderr, "\n------------ Die Flaeche ist wegen der Performance auf Grad 4 beschraenkt\n");
            return;
        }

        //lösche Kontrollpunkte.
        for (size_t i = 0; i < controlPoints.size(); i++)
        {
            delete controlPoints[i];
        }
        controlPoints.clear();

        //Führe Graderhöhung der Flaeche durch
        surface->degreeElevation('u');

        //Füge neue Punkte ein
        std::vector<Vec3> newPoints = surface->getAllControlPoints();
        for (size_t i = 0; i < newPoints.size(); i++)
        {
            controlPoints.push_back(new Point(newPoints[i]));
        }

        //Grad in u  Richtung aktualisieren
        m = surface->getM();

        break;
    }

    case 'v':
    {
        if (n > 4)
        {
            fprintf(stderr, "\n------------ Die Flaeche ist wegen der Performance auf Grad 4 beschraenkt\n");
            return;
        }

        //lösche Kontrollpunkte.
        for (size_t i = 0; i < controlPoints.size(); i++)
        {
            delete controlPoints[i];
        }
        controlPoints.clear();

        //Führe Graderhöhung der Flaeche durch
        surface->degreeElevation('v');

        //Füge neue Punkte ein
        std::vector<Vec3> newPoints = surface->getAllControlPoints();
        for (size_t i = 0; i < newPoints.size(); i++)
        {
            controlPoints.push_back(new Point(newPoints[i]));
        }

        //Grad in u und v Richtung aktualisieren
        n = surface->getN();

        break;
    }

    default:
    {
    }
    }

    //aktualisiere die Kurve
    surface->updateGeometry();
}

//------------------------------------------------------------------------
void BezierSurfacePlugin::reduceDegreeOfSurface(char direction, Reduction reduction)
{
    switch (direction)
    {
    case 'u':
    {
        if (m < 4)
        {
            fprintf(stderr, "\n------------ Der Grad kann nur bis Grad 3 verringert werden \n");
            return;
        }

        //lösche Kontrollpunkte.
        for (std::vector<Point *>::iterator iter = controlPoints.begin(); iter != controlPoints.end(); iter++)
        {
            delete *iter;
        }
        controlPoints.clear();

        //Führe Graderhöhung der Flaeche durch
        if (reduction == FOREST)
            surface->degreeReductionForest('u');

        else if (reduction == FARIN)
            surface->degreeReductionFarin('u');

        //Füge neue Punkte ein
        std::vector<Vec3> newPoints = surface->getAllControlPoints();
        for (size_t i = 0; i < newPoints.size(); i++)
        {
            controlPoints.push_back(new Point(newPoints[i]));
        }

        //Grad in u und v Richtung aktualisieren
        m = surface->getM();

        break;
    }

    case 'v':
    {
        if (n < 4)
        {
            fprintf(stderr, "\n------------ Der Grad kann nur bis Grad 3 verringert werden \n");
            return;
        }

        //lösche Kontrollpunkte.
        for (std::vector<Point *>::iterator iter = controlPoints.begin(); iter != controlPoints.end(); iter++)
        {
            delete *iter;
        }
        controlPoints.clear();

        //Führe Graderhöhung der Flaeche durch
        if (reduction == FOREST)
            surface->degreeReductionForest('v');

        else if (reduction == FARIN)
            surface->degreeReductionFarin('v');

        //Füge neue Punkte ein
        std::vector<Vec3> newPoints = surface->getAllControlPoints();
        for (size_t i = 0; i < newPoints.size(); i++)
        {
            controlPoints.push_back(new Point(newPoints[i]));
        }

        //Grad in u und v Richtung aktualisieren
        n = surface->getN();

        break;
    }
    }

    //aktualisiere die Kurve
    surface->updateGeometry();
}

//------------------------------------------------------------------------
void BezierSurfacePlugin::createMenu()
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

        px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.BezierSurface.MenuPosition", px);
        py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.BezierSurface.MenuPosition", py);
        pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.BezierSurface.MenuPosition", pz);

        // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
        float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
        s = coCoviseConfig::getFloat("value", "COVER.Plugin.BezierSurface.MenuSize", s);

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

    if (presentationStepCounter == 0 || presentationStepCounter == 6)
    {
        setMenuVisible(false);
    }
    else
    {
        setMenuVisible(true);
    }

    if (presentationStepCounter != 0 || presentationStepCounter != 6 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemShowControlPolygon = new coCheckboxMenuItem(coTranslator::coTranslate("Kontrollpolygon"), true);
            menuItemShowControlPolygon->setMenuListener(this);
        }
        menuItemShowControlPolygon->setState(showControlPolygon);

        objectsMenu->add(menuItemShowControlPolygon);

        if (firstLoad)
        {
            menuItemShowInteractors = new coCheckboxMenuItem(coTranslator::coTranslate("Interaktoren"), true);
            menuItemShowInteractors->setMenuListener(this);
        }
        menuItemShowInteractors->setState(showInteractors);

        objectsMenu->add(menuItemShowInteractors);
    }

    if (presentationStepCounter == 2 || presentationStepCounter == 3 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemSeparator1 = new coLabelMenuItem("______________________");
            menuItemSeparator1->setMenuListener(this);
        }

        objectsMenu->add(menuItemSeparator1);
    }

    if (presentationStepCounter == 4 || presentationStepCounter == 5 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemSeparator2 = new coLabelMenuItem("______________________");
            menuItemSeparator2->setMenuListener(this);
        }
        objectsMenu->add(menuItemSeparator2);
    }

    if (presentationStepCounter == 3 || presentationStepCounter == 4 || presentationStepCounter == 5 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemdegreeElevationU = new coButtonMenuItem(coTranslator::coTranslate("Graderhoehung u Richtung"));
            menuItemdegreeElevationU->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeElevationU);

        if (firstLoad)
        {
            menuItemdegreeElevationV = new coButtonMenuItem(coTranslator::coTranslate("Graderhoehung v Richtung"));
            menuItemdegreeElevationV->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeElevationV);
    }

    if (presentationStepCounter == 4 || presentationStepCounter == 5 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemdegreeReductionForestU = new coButtonMenuItem(coTranslator::coTranslate("Gradreduzieren u Richtung (Forest)"));
            menuItemdegreeReductionForestU->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeReductionForestU);

        if (firstLoad)
        {
            menuItemdegreeReductionForestV = new coButtonMenuItem(coTranslator::coTranslate("Gradreduzieren v Richtung (Forest)"));
            menuItemdegreeReductionForestV->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeReductionForestV);

        if (firstLoad)
        {
            menuItemdegreeReductionFarinU = new coButtonMenuItem(coTranslator::coTranslate("Gradreduzieren u Richtung (Farin)"));
            menuItemdegreeReductionFarinU->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeReductionFarinU);

        if (firstLoad)
        {
            menuItemdegreeReductionFarinV = new coButtonMenuItem(coTranslator::coTranslate("Gradreduzieren v Richtung (Farin)"));
            menuItemdegreeReductionFarinV->setMenuListener(this);
        }

        objectsMenu->add(menuItemdegreeReductionFarinV);
    }

    if (presentationStepCounter == 5 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemSeparator3 = new coLabelMenuItem("______________________");
            menuItemSeparator3->setMenuListener(this);
        }

        objectsMenu->add(menuItemSeparator3);
    }

    if (presentationStepCounter == 2 || presentationStepCounter == 5 || firstLoad)
    {
        if (firstLoad)
        {
            menuItemParameterValueU = new coSliderMenuItem(coTranslator::coTranslate("Parameter u"), 0.0, 1.0, 0.5);
            menuItemParameterValueU->setMenuListener(this);
        }

        objectsMenu->add(menuItemParameterValueU);

        if (firstLoad)
        {
            menuItemParameterValueV = new coSliderMenuItem(coTranslator::coTranslate("Parameter v"), 0.0, 1.0, 0.5);
            menuItemParameterValueV->setMenuListener(this);
        }

        objectsMenu->add(menuItemParameterValueV);

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
    }
}

//---------------------------------------------------------------------------------
void BezierSurfacePlugin::menuEvent(coMenuItem *menuItem)
{

    if (menuItem == menuItemdegreeElevationU)
    {
        elevateDegreeOfSurface('u');
    }

    else if (menuItem == menuItemdegreeElevationV)
    {
        elevateDegreeOfSurface('v');
    }

    else if (menuItem == menuItemdegreeReductionForestU)
    {
        reduceDegreeOfSurface('u', FOREST);
    }

    else if (menuItem == menuItemdegreeReductionForestV)
    {
        reduceDegreeOfSurface('v', FOREST);
    }

    else if (menuItem == menuItemdegreeReductionFarinU)
    {
        reduceDegreeOfSurface('u', FARIN);
    }

    else if (menuItem == menuItemdegreeReductionFarinV)
    {
        reduceDegreeOfSurface('v', FARIN);
    }

    else if (menuItem == menuItemParameterValueU)
    {
        if (showCasteljauStep == false)
        {
            showCasteljauStep = true;
            showCasteljauAnimation = false;
            menuItemAnimateCasteljau->setState(showCasteljauAnimation);
            menuItemShowCasteljau->setState(showCasteljauAnimation || showCasteljauStep);
            parameterValueAnimationU = 0;
            parameterValueAnimationV = 0;
            surface->showCasteljau(true);
        }
    }

    else if (menuItem == menuItemParameterValueV)
    {
        if (showCasteljauStep == false)
        {
            showCasteljauStep = true;
            showCasteljauAnimation = false;
            menuItemAnimateCasteljau->setState(showCasteljauAnimation);
            menuItemShowCasteljau->setState(showCasteljauAnimation || showCasteljauStep);
            parameterValueAnimationU = 0;
            parameterValueAnimationV = 0;
            surface->showCasteljau(true);
        }
    }

    else if (menuItem == menuItemShowControlPolygon)
    {
        showControlPolygon = menuItemShowControlPolygon->getState();
    }

    else if (menuItem == menuItemShowInteractors)
    {
        showInteractors = menuItemShowInteractors->getState();
    }

    else if (menuItem == menuItemAnimateCasteljau)
    {
        showCasteljauAnimation = menuItemAnimateCasteljau->getState();
        showCasteljauStep = false;
        surface->showCasteljau(showCasteljauAnimation);
        menuItemShowCasteljau->setState(showCasteljauAnimation || showCasteljauStep);
    }

    else if (menuItem == menuItemShowCasteljau)
    {
        if (menuItemShowCasteljau->getState())
        {
            showCasteljauAnimation = false;
            showCasteljauStep = true;
            menuItemAnimateCasteljau->setState(showCasteljauAnimation);
            surface->showCasteljau(showCasteljauStep);
        }
        else
        {
            showCasteljauAnimation = false;
            showCasteljauStep = false;
            menuItemAnimateCasteljau->setState(showCasteljauAnimation);
            surface->showCasteljau(showCasteljauStep);
        }
    }
}

void BezierSurfacePlugin::removeMenuEntries()
{
    if (menuItemdegreeElevationU)
    {
        objectsMenu->remove(menuItemdegreeElevationU);
    }

    if (menuItemdegreeElevationV)
    {
        objectsMenu->remove(menuItemdegreeElevationV);
    }

    if (menuItemShowControlPolygon)
    {
        objectsMenu->remove(menuItemShowControlPolygon);
    }

    if (menuItemShowInteractors)
    {
        objectsMenu->remove(menuItemShowInteractors);
    }

    if (menuItemAnimateCasteljau)
    {
        objectsMenu->remove(menuItemAnimateCasteljau);
    }

    if (menuItemShowCasteljau)
    {
        objectsMenu->remove(menuItemShowCasteljau);
    }

    if (menuItemdegreeReductionForestU)
    {
        objectsMenu->remove(menuItemdegreeReductionForestU);
    }

    if (menuItemdegreeReductionForestV)
    {
        objectsMenu->remove(menuItemdegreeReductionForestV);
    }

    if (menuItemdegreeReductionFarinU)
    {
        objectsMenu->remove(menuItemdegreeReductionFarinU);
    }

    if (menuItemdegreeReductionFarinV)
    {
        objectsMenu->remove(menuItemdegreeReductionFarinV);
    }

    if (menuItemParameterValueU)
    {
        objectsMenu->remove(menuItemParameterValueU);
    }

    if (menuItemParameterValueV)
    {
        objectsMenu->remove(menuItemParameterValueV);
    }

    if (menuItemSeparator1)
    {
        objectsMenu->remove(menuItemSeparator1);
    }

    if (menuItemSeparator2)
    {
        objectsMenu->remove(menuItemSeparator2);
    }

    if (menuItemSeparator3)
    {
        objectsMenu->remove(menuItemSeparator3);
    }

    if (!firstLoad)
        createMenu();
}

void BezierSurfacePlugin::changeStatus()
{
    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        delete controlPoints[i];
    }
    controlPoints.clear();
    surface->removeAllControlPoints();

    makeSurface();

    MatrixTransform *trafo = VRSceneGraph::instance()->getTransform();
    Matrix m;
    m.makeIdentity();
    trafo->setMatrix(m);

    switch (presentationStepCounter)
    {
    case 0:
        surface->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showControlPolygon = true;
        showInteractors = true;
        break;

    case 1:
        surface->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showInteractors = true;
        showControlPolygon = true;
        break;

    case 2:
        parameterValueAnimationU = 0;
        parameterValueAnimationV = 0;
        parameterValueStepV = 0.5;
        parameterValueStepU = 0.5;
        elevateDegreeOfSurface('v');
        elevateDegreeOfSurface('u');
        surface->showCasteljau(true);
        showCasteljauAnimation = true;
        showCasteljauStep = false;
        showControlPolygon = true;
        showInteractors = false;
        break;

    case 3:
        showCasteljauAnimation = false;
        surface->showCasteljau(false);
        showCasteljauStep = false;
        showInteractors = true;
        showControlPolygon = true;
        showInteractors = true;
        break;

    case 4:
        surface->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showControlPolygon = true;
        showInteractors = true;
        break;

    case 5:
        surface->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showControlPolygon = true;
        showInteractors = true;
        break;

    case 6:
        parameterValueAnimationU = 0;
        parameterValueAnimationV = 0;
        parameterValueStepV = 0.5;
        parameterValueStepU = 0.5;
        surface->showCasteljau(false);
        showCasteljauAnimation = false;
        showCasteljauStep = false;
        showControlPolygon = true;
        showInteractors = true;
        break;
    }
}

void BezierSurfacePlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- BezierSurfacePlugin coVRGuiToRenderMsg %s\n", msg.getString().c_str());

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
                if (presentationStepCounter > 6)
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

void BezierSurfacePlugin::setMenuVisible(bool visible)
{
    objectsMenu->setVisible(visible);
    VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
}

COVERPLUGIN(BezierSurfacePlugin)
