/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TwoHandInteractionPlugin.h"

#include <config/CoviseConfig.h>
#include <cover/OpenCOVER.h>
#include <cover/input/input.h>
#include <iostream>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>

#include "TestScaleInteractionHandler.h"
#include "TestRotateInteractionHandler.h"

using namespace std;
using namespace opencover;
using covise::coCoviseConfig;

//-----------------------------------------------------------------------------

namespace
{
ostream &operator<<(ostream &os, const osg::Matrixf &transform)
{
    os << "[osg::Matrixf: " << endl;
    os << "[ " << transform(0, 0) << ", " << transform(0, 1) << ", " << transform(0, 2) << ", " << transform(0, 3)
       << "]" << endl;
    os << "[ " << transform(1, 0) << ", " << transform(1, 1) << ", " << transform(1, 2) << ", " << transform(1, 3)
       << "]" << endl;
    os << "[ " << transform(2, 0) << ", " << transform(2, 1) << ", " << transform(2, 2) << ", " << transform(2, 3)
       << "]" << endl;
    os << "[ " << transform(3, 0) << ", " << transform(3, 1) << ", " << transform(3, 2) << ", " << transform(3, 3)
       << "]" << endl;
    os << "]" << endl;
    return os;
}
}

namespace TwoHandInteraction
{

TwoHandInteractionPlugin::TwoHandInteractionPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, m_HasIndicators(false)
, m_InteractionHandler(new TestRotateInteractionHandler)
{
}

TwoHandInteractionPlugin::~TwoHandInteractionPlugin()
{
}

bool TwoHandInteractionPlugin::init()
{
    fprintf(stderr, "\nTwoHandedInteractionPlugin::init\n");

    bool showIndicator = coCoviseConfig::isOn("COVER.Plugin.TwoHandInteraction.ShowIndicator", false);
    float indicatorSize = coCoviseConfig::getFloat("COVER.Plugin.TwoHandInteraction.IndicatorSize", 5);

    if (showIndicator)
    {
        m_HasIndicators = true;
        createIndicators(indicatorSize);
    }

    return true;
}

void TwoHandInteractionPlugin::createIndicators(float indicatorSize)
{
    m_HandIndicator = new osg::MatrixTransform();
    m_HandIndicator->setName("TwoHandInteraction_HandIndicator");
    osg::Geode *handGeode = new osg::Geode();
    cover->setNodesIsectable(handGeode, false);
    m_HandIndicator->addChild(handGeode);

    handGeode->addDrawable(new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0.0, 0.0, 0.0), indicatorSize)));
    cover->getScene()->addChild(m_HandIndicator.get());

    m_SecondHandIndicator = new osg::MatrixTransform();
    m_SecondHandIndicator->setName("TwoHandInteraction_SecondHandIndicator");

    osg::Geode *secondHandGeode = new osg::Geode();
    cover->setNodesIsectable(secondHandGeode, false);
    m_SecondHandIndicator->addChild(secondHandGeode);
    secondHandGeode->addDrawable(new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0.0, 0.0, 0.0), indicatorSize)));
    cover->getScene()->addChild(m_SecondHandIndicator.get());
}

void TwoHandInteractionPlugin::preFrame()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\nTwoHandInteractionPlugin::preFrame\n");

    //cerr << "Num Stations = " << cover->getNumStations() << endl;

    if (Input::instance()->hasHead() && Input::instance()->isHeadValid())
    {
        osg::Matrix mat = Input::instance()->getHeadMat();
        //cerr << "head Matrix = " << mat << endl;
    }
    else
    {
        //cerr << "no head device" << endl;
    }

    if (m_HasIndicators)
    {
        //cerr << "Has Indicators!" << endl;
    }

    osg::Matrix handMat;
    if (Input::instance()->hasHand(0) && Input::instance()->isHandValid(0))
    {
        handMat = Input::instance()->getHandMat(0);
        if (m_HasIndicators)
        {
            m_HandIndicator->setMatrix(handMat);
        }
    }
    else
    {
        //cerr << "no hand device" << endl;
    }

    osg::Matrix secondHandMat;
    if (Input::instance()->hasHand(1) && Input::instance()->isHandValid(1))
    {
        secondHandMat = Input::instance()->getHandMat(1);
        if (m_HasIndicators)
        {
            m_SecondHandIndicator->setMatrix(secondHandMat);
        }
    }
    else
    {
        //cerr << "no second hand device" << endl;
    }

    bool buttonPressed = false;
    coPointerButton *pointerButton = cover->getPointerButton();
    if (pointerButton)
    {
        if (pointerButton->getState() == 1)
        {
            buttonPressed = true;
            //cerr << "Button pressed !" << endl;
        }
        else
        {
            //cerr << "Button not pressed." << endl;
        }
    }

    InteractionStart interactionStart;
    interactionStart.ScalingMatrix = cover->getObjectsScale()->getMatrix();
    osg::Matrix rotTransMat = cover->getObjectsXform()->getMatrix();
    interactionStart.RotationMatrix.makeRotate(rotTransMat.getRotate());
    interactionStart.TranslationMatrix.makeTranslate(rotTransMat.getTrans());

    if (m_InteractionHandler)
    {
        InteractionResult interactionResult = m_InteractionHandler->CalculateInteraction(cover->frameTime(), interactionStart, buttonPressed, handMat,
                                                                                         secondHandMat);
        applyInteractionResult(interactionResult);
    }
}

void TwoHandInteractionPlugin::applyInteractionResult(const InteractionResult &interactionResult)
{
    osg::MatrixTransform *scaling = cover->getObjectsScale();
    osg::MatrixTransform *rotTrans = cover->getObjectsXform();

    scaling->setMatrix(interactionResult.ScalingMatrix);
    rotTrans->setMatrix(interactionResult.RotationMatrix * interactionResult.TranslationMatrix);
}
}

COVERPLUGIN(TwoHandInteraction::TwoHandInteractionPlugin)
