/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif

#include <OpenVRUI/coToolboxMenuHandle.h>

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coFrame.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coToggleButtonGeometry.h>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/coCombinedButtonInteraction.h>

#include <config/CoviseConfig.h>

#define HANDLESCALE 20

namespace vrui
{

/** Creates a menu item style handle for RowMenus.
    @param name text displayed in title bar
*/
coToolboxMenuHandle::coToolboxMenuHandle(const std::string &name, coToolboxMenu *menu)
    : coRowContainer(coRowContainer::HORIZONTAL)
    , localPickPosition(3)
    , pickPosition(3)
{
    // structure: HandleContainer{ titleFrame( titleBackground( titleContainer( Operations... ) ) ) FURTHER_ITEMS ... }
    localPickPosition[0] = localPickPosition[1] = localPickPosition[2] = 0.0f;
    pickPosition[0] = pickPosition[1] = pickPosition[2] = 0.0f;
    myMenu = menu;

    myScale = 1;

    // create background
    titleBackground = new coColoredBackground(coUIElement::HANDLE_BACKGROUND_NORMAL, coUIElement::HANDLE_BACKGROUND_HIGHLIGHTED, coUIElement::HANDLE_BACKGROUND_DISABLED);

    // create operation buttons
    closeButton = new coPushButton(new coFlatButtonGeometry("UI/close"), this);
    minmaxButton = new coToggleButton(new coToggleButtonGeometry("UI/minmax"), this);
    cwrotButton = new coToggleButton(new coToggleButtonGeometry("UI/cwrot"), this);

    // set up operations container
    titleContainer = new coMenuContainer(coRowContainer::VERTICAL);
    titleContainer->addElement(minmaxButton);
    titleContainer->addElement(cwrotButton);
    titleContainer->setNumAlignedMin(1);

    // set up
    titleFrame = new coFrame("UI/Frame");
    titleFrame->addElement(titleBackground);
    titleBackground->addElement(titleContainer);

    titleFrame->fitToParent();

    unregister = false;
    vruiIntersection::getIntersectorForAction("coAction")->add(titleBackground->getDCS(), this);

    std::string entryName = "COVER.VRUI." + name + ".Menu.Position";
    float x = covise::coCoviseConfig::getFloat("x", entryName.c_str(), 0.0f);
    float y = covise::coCoviseConfig::getFloat("y", entryName.c_str(), 0.0f);
    float z = covise::coCoviseConfig::getFloat("z", entryName.c_str(), 0.0f);

    entryName = "COVER.VRUI." + name + ".Menu.Orientation";
    float h = covise::coCoviseConfig::getFloat("h", entryName.c_str(), 0.0f);
    float p = covise::coCoviseConfig::getFloat("p", entryName.c_str(), 0.0f);
    float r = covise::coCoviseConfig::getFloat("r", entryName.c_str(), 0.0f);

    entryName = "COVER.VRUI." + name + ".Menu.Size";
    myScale = covise::coCoviseConfig::getFloat(entryName.c_str(), myScale);

    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rotateMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *matrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rxMatrix = vruiRendererInterface::the()->createMatrix();
    /*
	float theight = 0;
	float twidth = 0;
	if (!covise::coCoviseConfig::isOn("COVER.Plugin.AKToolbar.HandleOn", true) && covise::coCoviseConfig::isOn("oldPosition","COVER.Plugin.AKToolbar.HandleOn", false, 0) ) 
	{
		theight = getScale()*120;
		if (matrix->getTranslate()[0] < 0)
			twidth = getScale()*(titleFrame->getWidth() +titleFrame->getBorderWidth());
	}*/

    transMatrix->makeTranslate(x, y, z);
    rotateMatrix->makeEuler(h, p, r);
    scaleMatrix->makeScale(myScale, myScale, myScale);
    rxMatrix->makeEuler(0.0, 90.0, 0.0);

    matrix->makeIdentity();
    matrix->mult(rxMatrix);
    matrix->mult(scaleMatrix);
    matrix->mult(rotateMatrix);
    matrix->mult(transMatrix);

    getDCS()->setMatrix(matrix);

    // check if the handle is hidden by config
    if (covise::coCoviseConfig::isOn("COVER.Plugin.AKToolbar.HandleOn", true))
        addElement(titleFrame);

    minimized = false;
    fixedPos = false;

    vruiRendererInterface::the()->deleteMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(rotateMatrix);
    vruiRendererInterface::the()->deleteMatrix(scaleMatrix);
    vruiRendererInterface::the()->deleteMatrix(matrix);
    vruiRendererInterface::the()->deleteMatrix(rxMatrix);

    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "MenuHandle", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonB, "MenuHandle", coInteraction::Menu);
    interactionC = new coCombinedButtonInteraction(coInteraction::ButtonC, "MenuHandle", coInteraction::Menu);

    invStartHandTrans = vruiRendererInterface::the()->createMatrix();
    startPosition = vruiRendererInterface::the()->createMatrix();

    invStartHandTrans->makeIdentity();
    startPosition->makeIdentity();
}

coToolboxMenuHandle::~coToolboxMenuHandle()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    vruiRendererInterface::the()->deleteMatrix(invStartHandTrans);
    vruiRendererInterface::the()->deleteMatrix(startPosition);

    delete interactionA;
    delete interactionB;
    delete interactionC;

    delete titleBackground;
    delete closeButton;
    delete minmaxButton;
    delete cwrotButton;
    delete titleContainer;
    delete titleFrame;
}

void coToolboxMenuHandle::addCloseButton()
{
    titleContainer->insertElement(closeButton, 0);
}

void coToolboxMenuHandle::buttonEvent(coButton *button)
{
    if (button == closeButton)
    {
        myMenu->closeMenu();
    }

    else if (button == minmaxButton)
    {
        if (minimized)
            myMenu->show();
        else
            myMenu->hide();

        minimized = !minimized;
    }
    else if (button == cwrotButton)
    {
        // tell Menu to rotate clockwise
        switch (myMenu->getAttachment())
        {
        case TOP:
            myMenu->setAttachment(RIGHT);
            break;

        case RIGHT:
            myMenu->setAttachment(BOTTOM);
            break;

        case BOTTOM:
            myMenu->setAttachment(LEFT);
            break;

        case LEFT:
            myMenu->setAttachment(TOP);
            break;
        }
        startPosition = getDCS()->getMatrix();
        // FIXME: get hand or mouse matrix
        invStartHandTrans->makeInverse(vruiRendererInterface::the()->getHandMatrix());
    }
}

/*
void coToolboxMenuHandle::childResized()
{
    float oldHeight=getHeight();

    resize();

    pfMatrix trans,mat;
    myDCS->getMat(mat);

    trans.preTrans(0,oldHeight-getHeight(),0,mat);

myDCS->setMat(trans);
}*/

void coToolboxMenuHandle::setScale(float s)
{
    vruiMatrix *currentMatrix = 0;
    vruiMatrix *newMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *itMatrix = vruiRendererInterface::the()->createMatrix();

    float scale = s / myScale;
    itMatrix->makeTranslate(-getWidth() / 2.0, -getHeight(), 0.0);
    if (!covise::coCoviseConfig::isOn("COVER.Plugin.AKToolbar.HandleOn", true) && covise::coCoviseConfig::isOn("oldPosition", "COVER.Plugin.AKToolbar.HandleOn", false, 0))
        transMatrix->makeTranslate(getWidth() / 2.0 * scale + titleFrame->getWidth() + 2 * titleFrame->getBorderWidth(), getHeight(), 0.0);
    else
        transMatrix->makeTranslate(getWidth() * scale / 2.0, getHeight(), 0.0);

    currentMatrix = getDCS()->getMatrix();
    scaleMatrix->makeScale(scale, scale, scale);

    newMatrix->makeIdentity();
    newMatrix->mult(itMatrix);
    newMatrix->mult(scaleMatrix);
    newMatrix->mult(transMatrix);
    newMatrix->mult(currentMatrix);

    getDCS()->setMatrix(newMatrix);
    myScale = s;

    vruiRendererInterface::the()->deleteMatrix(newMatrix);
    vruiRendererInterface::the()->deleteMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(scaleMatrix);
    vruiRendererInterface::the()->deleteMatrix(itMatrix);
}

float coToolboxMenuHandle::getScale() const
{
    return myScale;
}

void coToolboxMenuHandle::setTransformMatrix(vruiMatrix *matrix)
{

    vruiMatrix *translatedMatrix = vruiRendererInterface::the()->createMatrix();

    //VRUILOG("coRowMenuHandle::setTransformMatrix info: heights: title = " <<
    //        titleFrame->getHeight() << " all = " << getHeight())

    //VRUILOG("coRowMenuHandle::setTransformMatrix info: widths: all = " << getWidth())
    //VRUILOG("coRowMenuHandle::setTransformMatrix info: scale = " << getScale())
    /*
	float theight = 0;
	float twidth = 0;
	if (!covise::coCoviseConfig::isOn("COVER.Plugin.AKToolbar.HandleOn", true) && covise::coCoviseConfig::isOn("oldPosition","COVER.Plugin.AKToolbar.HandleOn", false, 0) ) 
	{
		theight = getScale()*120;
		if (matrix->getTranslate()[0] < 0)
			twidth = getScale()*(titleFrame->getWidth() +titleFrame->getBorderWidth());
	}*/

    translatedMatrix->preTranslated(0.0, -getHeight(), 0.0, matrix);

    vruiMatrix *newMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *itMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rxMatrix = vruiRendererInterface::the()->createMatrix();
    rxMatrix->makeEuler(0.0, 90.0, 0.0);

    itMatrix->makeTranslate(-getWidth() / 2.0, -getHeight(), 0.0);
    transMatrix->makeTranslate((getWidth() * myScale) / 2.0, getHeight() * myScale, 0.0);
    scaleMatrix->makeScale(myScale, myScale, myScale);

    newMatrix->makeIdentity();
    newMatrix->mult(rxMatrix);
    newMatrix->mult(itMatrix);
    newMatrix->mult(scaleMatrix);
    newMatrix->mult(transMatrix);
    newMatrix->mult(translatedMatrix);

    getDCS()->setMatrix(newMatrix);

    vruiRendererInterface::the()->deleteMatrix(newMatrix);
    vruiRendererInterface::the()->deleteMatrix(scaleMatrix);
    vruiRendererInterface::the()->deleteMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(itMatrix);
    vruiRendererInterface::the()->deleteMatrix(rxMatrix);
}

/** set a new menu position orientation and scale matrix.
    @param mat transformation matrix which includes scale
    @param scale scale factor included in the above matrix
  */
void coToolboxMenuHandle::setTransformMatrix(vruiMatrix *matrix, float scale)
{

    //VRUILOG("coRowMenuHandle::setTransformMatrix info: heights: all = " << getHeight())
    /*
	float theight = 0;
	float twidth = 0;
	if (!covise::coCoviseConfig::isOn("COVER.Plugin.AKToolbar.HandleOn", true) && covise::coCoviseConfig::isOn("oldPosition","COVER.Plugin.AKToolbar.HandleOn", false, 0) ) 
	{
		theight = getScale()*120;
		if (matrix->getTranslate()[0] < 0)
			twidth = getScale()*(titleFrame->getWidth() +titleFrame->getBorderWidth());
	}*/

    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    transMatrix->preTranslated(0.0, titleFrame->getHeight() - getHeight(), 0.0, matrix);
    myScale = scale;
    getDCS()->setMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(transMatrix);
}

bool coToolboxMenuHandle::update()
{

    if (interactionA->isRunning() && !fixedPos)
    {
        // left Button is moved

        vruiMatrix *newMat = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *ppTransform = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *transMat = vruiRendererInterface::the()->createMatrix();
        ppTransform->makeIdentity();
        ppTransform->mult(invStartHandTrans);
        ppTransform->mult(interactionA->getHandMatrix());
        coVector newPos = ppTransform->getFullXformPt(pickPosition);
        if (vruiRendererInterface::the()->isMultiTouchDevice())
        {
            newMat->makeIdentity();
            newMat->mult(startPosition);
            transMat->makeTranslate(newPos[0] - pickPosition[0], newPos[1] - pickPosition[1], newPos[2] - pickPosition[2]);
            newMat->mult(transMat);
            getDCS()->setMatrix(newMat);
        }
        else
        {

            coVector viewerPos = vruiRendererInterface::the()->getViewerMatrix()->getTranslate();
            coVector viewerDir;
            coVector tmp;

            coVector normal(0.0, -1.0, 0.0);
            coVector axis(0.0, 0.0, 1.0);
            // ViewerMat in world coordinatesystem
            // direction to the viewer

            viewerDir = viewerPos - newPos;
            viewerDir[2] = 0.0;
            viewerDir.normalize();

            vruiMatrix *rMat = vruiRendererInterface::the()->createMatrix();
            vruiMatrix *rxMat = vruiRendererInterface::the()->createMatrix();
            vruiMatrix *sMat = vruiRendererInterface::the()->createMatrix();
            vruiMatrix *invTransMat = vruiRendererInterface::the()->createMatrix();
            vruiMatrix *current = vruiRendererInterface::the()->createMatrix();

            float angle; // rotationswinkel ausrechnen

            //VRUILOG("coRowMenuHandle::update info: viewerPos = " << viewerPos << ", viewerDir = " << viewerDir)

            angle = (float)(acos(viewerDir.dot(normal)) / M_PI * 180.0);
            tmp = viewerDir.cross(normal);
            if (tmp.dot(axis) > 0.0)
                angle = -angle;

            rMat->makeRotate(angle, 0.0, 0.0, 1.0);
            rxMat->makeEuler(0.0, 90.0, 0.0);
            current->makeTranslate(newPos[0], newPos[1], newPos[2]);
            transMat->makeTranslate(localPickPosition[0], localPickPosition[1], localPickPosition[2]);
            invTransMat->makeTranslate(-localPickPosition[0], -localPickPosition[1], -localPickPosition[2]);
            sMat->makeScale(myScale, myScale, myScale);

            newMat->makeIdentity();
            if (vruiRendererInterface::the()->isMultiTouchDevice())
            {
                rxMat->makeEuler(0.0, 90.0, 0.0); // don't rotate, only translate
            }
            newMat->mult(invTransMat)->mult(rxMat)->mult(sMat)->mult(rMat)->mult(current);
            getDCS()->setMatrix(newMat);

            vruiRendererInterface::the()->deleteMatrix(rMat);
            vruiRendererInterface::the()->deleteMatrix(rxMat);
            vruiRendererInterface::the()->deleteMatrix(sMat);
            vruiRendererInterface::the()->deleteMatrix(invTransMat);
            vruiRendererInterface::the()->deleteMatrix(current);
        }
        vruiRendererInterface::the()->deleteMatrix(transMat);
        vruiRendererInterface::the()->deleteMatrix(ppTransform);
        vruiRendererInterface::the()->deleteMatrix(newMat);
    }
    else if (interactionB->isRunning() && !fixedPos)
    {
        // middle Button is moved
        vruiCoord mouseCoord = interactionB->getHandMatrix();
        if (lastRoll != mouseCoord.hpr[2])
        {
            float lastSize = myScale;
            if ((lastRoll - mouseCoord.hpr[2]) > 180.0)
                lastRoll -= 360.0;
            if ((lastRoll - mouseCoord.hpr[2]) < -180.0)
                lastRoll += 360.0;
            myScale += (lastRoll - (float)mouseCoord.hpr[2]) / 9.0f;
            if (myScale < 0.01f)
                myScale = 0.01f;
            lastRoll = (float)mouseCoord.hpr[2];

            vruiMatrix *newMat;
            vruiMatrix *sMat = vruiRendererInterface::the()->createMatrix();
            vruiMatrix *tMat = vruiRendererInterface::the()->createMatrix();
            vruiMatrix *itMat = vruiRendererInterface::the()->createMatrix();

            itMat->makeTranslate(-getWidth() / 2.0, -getHeight(), 0.0);
            tMat->makeTranslate(getWidth() / 2.0, getHeight(), 0.0);
            sMat->makeScale(myScale / lastSize, myScale / lastSize, myScale / lastSize);
            newMat = itMat->mult(sMat)->mult(tMat)->mult(getDCS()->getMatrix());

            getDCS()->setMatrix(newMat);

            vruiRendererInterface::the()->deleteMatrix(sMat);
            vruiRendererInterface::the()->deleteMatrix(tMat);
            vruiRendererInterface::the()->deleteMatrix(itMat);
        }
    }
    else if (interactionC->isRunning() && !fixedPos)
    {
        // right Button is moved
        vruiMatrix *current = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *hand = vruiRendererInterface::the()->createMatrix();

        current->makeIdentity();
        current->mult(startPosition);
        current->mult(invStartHandTrans);
        current->mult(interactionC->getHandMatrix());

        getDCS()->setMatrix(current);

        vruiRendererInterface::the()->deleteMatrix(current);
        vruiRendererInterface::the()->deleteMatrix(hand);
    }

    if (unregister)
    {
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionB->isRegistered() && (interactionB->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionB);
        }
        if (interactionC->isRegistered() && (interactionC->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionC);
        }
        if ((!interactionA->isRegistered()) && (!interactionB->isRegistered()) && (!interactionC->isRegistered()))
        {
            unregister = false;
        }
    }

    return true;
}

void coToolboxMenuHandle::miss()
{
    vruiRendererInterface::the()->miss(this);
    titleContainer->setHighlighted(false);
    unregister = true;
}

int coToolboxMenuHandle::hit(vruiHit *hit)
{

    //VRUILOG("coRowMenuHandle::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    titleContainer->setHighlighted(true);

    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
        interactionA->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionB->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionB);
        interactionB->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionC->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionC);
        interactionC->setHitByMouse(hit->isMouseHit());
    }

    if (interactionA->wasStarted() || interactionB->wasStarted() || interactionC->wasStarted())
    {
        coCombinedButtonInteraction *inter = interactionA->wasStarted() ? interactionA
                                                                        : (interactionB->wasStarted() ? interactionB : interactionC);
        startPosition->makeIdentity();
        startPosition->mult(getDCS()->getMatrix());

        invStartHandTrans->makeInverse(inter->getHandMatrix());

        vruiCoord mouseCoord = inter->getHandMatrix();
        lastRoll = (float)mouseCoord.hpr[2];

        vruiMatrix *myTrans = vruiRendererInterface::the()->createMatrix();
        myTrans->makeInverse(getTransformMatrix());

        localPickPosition = myTrans->getFullXformPt(hit->getWorldIntersectionPoint());
        //localPickPosition = hit->getLocalIntersectionPoint();
        pickPosition = hit->getWorldIntersectionPoint();
        vruiRendererInterface::the()->deleteMatrix(myTrans);
    }
    return ACTION_CALL_ON_MISS;
}

void coToolboxMenuHandle::highlight(bool highlight)
{
    titleContainer->setHighlighted(highlight);
}

void coToolboxMenuHandle::setOrientation(coRowContainer::Orientation ori)
{
    orientation = ori;
    childResized();

    if (ori == HORIZONTAL)
        titleContainer->setOrientation(VERTICAL);
    else
        titleContainer->setOrientation(HORIZONTAL);
}

const char *coToolboxMenuHandle::getClassName() const
{
    return "coToolboxMenuHandle";
}

bool coToolboxMenuHandle::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coRowContainer::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void coToolboxMenuHandle::fixPos(bool doFix)
{
    fixedPos = doFix;
}
}
