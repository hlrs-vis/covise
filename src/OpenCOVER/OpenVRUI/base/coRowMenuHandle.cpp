/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif

#include <OpenVRUI/coRowMenuHandle.h>

#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <util/coVector.h>
#include <config/CoviseConfig.h>

#include <string>

using namespace std;

namespace vrui
{

#define HANDLESCALE 20

/** Creates a titlebar for RowMenus.
    @param name text displayed in title bar
    @param menu the menu to handle
*/
coRowMenuHandle::coRowMenuHandle(const string &name, coMenu *menu)
    : coRowContainer(coRowContainer::VERTICAL)
{
    // myDCS is created by coRowContainer c'tor
    string nodeName = (string("coRowMenu(") + name) + ")";
    getDCS()->setName(nodeName);
    localPickPosition[0] = localPickPosition[1] = localPickPosition[2] = 0.0f;
    pickPosition[0] = pickPosition[1] = pickPosition[2] = 0.0f;

    title = name;

    myMenu = menu;
    myScale = 1.0f;

    // Create title bar:
    titleBackground = new coColoredBackground(coUIElement::HANDLE_BACKGROUND_NORMAL, coUIElement::HANDLE_BACKGROUND_HIGHLIGHTED, coUIElement::HANDLE_BACKGROUND_DISABLED);
    titleLabel = new coLabel();
    titleLabel->setString(title);
    closeButton = new coPushButton(new coFlatButtonGeometry("UI/close"), this);
    minmaxButton = new coToggleButton(new coToggleButtonGeometry("UI/minmax"), this);

    titleContainer = new coMenuContainer(coRowContainer::HORIZONTAL);

    titleContainer->addElement(titleLabel);
    titleContainer->addElement(minmaxButton);
    titleContainer->setNumAlignedMin(1);
    titleContainer->addElement(closeButton);

    titleBackground->addElement(titleContainer);

    titleFrame = new coFrame("UI/Frame");

    titleFrame->addElement(titleBackground);
    titleFrame->fitToParent();

    unregister = false;

    vruiIntersection::getIntersectorForAction("coAction")->add(titleBackground->getDCS(), this);

    string entryName = "COVER.VRUI." + name + ".Menu.Position";
    float x = covise::coCoviseConfig::getFloat("x", entryName.c_str(), 0.0f);
    float y = covise::coCoviseConfig::getFloat("y", entryName.c_str(), 0.0f);
    float z = covise::coCoviseConfig::getFloat("z", entryName.c_str(), 0.0f);

    entryName = "COVER.VRUI." + name + ".Menu.Orientation";
    float h = covise::coCoviseConfig::getFloat("h", entryName.c_str(), 0.0f);
    float p = covise::coCoviseConfig::getFloat("p", entryName.c_str(), 0.0f);
    float r = covise::coCoviseConfig::getFloat("r", entryName.c_str(), 0.0f);

    entryName = "COVER.VRUI." + name + ".Menu.Size";
    myScale = covise::coCoviseConfig::getFloat(entryName.c_str(), myScale);

    //fprintf(stderr, "menu %s pos (%f %f %f) scale %f\n", name.c_str(), x,y,z, myScale);

    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rotateMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *matrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rxMatrix = vruiRendererInterface::the()->createMatrix();

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
    if (covise::coCoviseConfig::isOn("COVER.Menu.HandleOn", true))
        addElement(titleFrame);

    minimized = false;

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

/// Destructor
coRowMenuHandle::~coRowMenuHandle()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    vruiRendererInterface::the()->deleteMatrix(invStartHandTrans);
    vruiRendererInterface::the()->deleteMatrix(startPosition);

    delete interactionA;
    delete interactionB;
    delete interactionC;

    delete titleBackground;
    delete titleLabel;
    delete closeButton;
    delete minmaxButton;

    delete titleContainer;
    delete titleFrame;
}

void coRowMenuHandle::resizeToParent(float newWidth, float newHeight, float newDepth, bool shrink)
{

    //VRUILOG("coRowMenuHandle::resizeToParent info: called")

    float oldHeight = getHeight();

    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *matrix = 0;
    // activate resize mechanism of own container
    coRowContainer::resizeToParent(newWidth, newHeight, newDepth, shrink);

    if (transMatrix)
    {
        // move container up? down?
        matrix = getDCS()->getMatrix();

        transMatrix->preTranslated(0.0, oldHeight - getHeight(), 0.0, matrix);
        getDCS()->setMatrix(transMatrix);

        vruiRendererInterface::the()->deleteMatrix(transMatrix);
    }
}

void coRowMenuHandle::shrinkToMin()
{

    //VRUILOG("coRowMenuHandle::shrinkToMin info: called")

    float oldHeight = getHeight();

    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *matrix = 0;

    if (transMatrix)
    {
        // do the real shrinkToMin
        coRowContainer::shrinkToMin();

        // move container up? down?
        matrix = getDCS()->getMatrix();
        transMatrix->preTranslated(0.0, oldHeight - getHeight(), 0.0, matrix);
        getDCS()->setMatrix(transMatrix);

        vruiRendererInterface::the()->deleteMatrix(transMatrix);
    }
}

void coRowMenuHandle::buttonEvent(coButton *button)
{

    //VRUILOG("coRowMenuHandle::buttonEvent info: called")

    if (button == closeButton)
    {
        myMenu->closeMenu();
    }
    else
    {
        // minmax Button was pushed
        if (button == minmaxButton)
        {
            // if already minimised: show it
            if (minimized)
            {
                myMenu->show();
            }
            // if shown: minimise
            else
            {
                myMenu->hide();
            }

            // toggle state
            minimized = !minimized;
        }
    }
}

/** scales the menu.
   Default scale is 1, the menu is scaled around the center of the titlebar
  */
void coRowMenuHandle::setScale(float s)
{
    //VRUILOG("coRowMenuHandle::setScale info: called with s = " << s)

    vruiMatrix *currentMatrix = 0;
    vruiMatrix *newMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *itMatrix = vruiRendererInterface::the()->createMatrix();

    float scale = s / myScale;
    itMatrix->makeTranslate(-getWidth() / 2.0, -getHeight(), 0.0);
    if (!covise::coCoviseConfig::isOn("COVER.Menu.HandleOn", true) && covise::coCoviseConfig::isOn("oldPosition", "COVER.Menu.HandleOn", false, 0))
        transMatrix->makeTranslate(getWidth() * scale / 2.0, getHeight() - (titleFrame->getHeight() + titleFrame->getBorderHeight()) * scale, 0.0);
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

/** Get Scale factor.
  @return current scale factor (default is 1.0)
  */
float coRowMenuHandle::getScale() const
{
    return myScale;
}

/** set a new menu position and orientation matrix.
   the Menu is scaled to the correct size after applying this matrix
  */
void coRowMenuHandle::setTransformMatrix(vruiMatrix *matrix)
{
    vruiMatrix *translatedMatrix = vruiRendererInterface::the()->createMatrix();

    //VRUILOG("coRowMenuHandle::setTransformMatrix info: heights: title = " <<
    //        titleFrame->getHeight() << " all = " << getHeight())

    //VRUILOG("coRowMenuHandle::setTransformMatrix info: widths: all = " << getWidth())
    //VRUILOG("coRowMenuHandle::setTransformMatrix info: scale = " << getScale())

    translatedMatrix->preTranslated(0.0, -getHeight(), 0.0, matrix);

    vruiMatrix *newMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *itMatrix = vruiRendererInterface::the()->createMatrix();

    itMatrix->makeTranslate(-getWidth() / 2.0, -getHeight(), 0.0);
    transMatrix->makeTranslate((getWidth() * myScale) / 2.0, getHeight() * myScale, 0.0);
    scaleMatrix->makeScale(myScale, myScale, myScale);

    newMatrix->makeIdentity();
    newMatrix->mult(itMatrix);
    newMatrix->mult(scaleMatrix);
    newMatrix->mult(transMatrix);
    newMatrix->mult(translatedMatrix);

    getDCS()->setMatrix(newMatrix);

    vruiRendererInterface::the()->deleteMatrix(newMatrix);
    vruiRendererInterface::the()->deleteMatrix(scaleMatrix);
    vruiRendererInterface::the()->deleteMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(itMatrix);
}

/** set a new menu position orientation and scale matrix.
    @param mat transformation matrix which includes scale
    @param scale scale factor included in the above matrix
  */
void coRowMenuHandle::setTransformMatrix(vruiMatrix *matrix, float scale)
{
    //VRUILOG("coRowMenuHandle::setTransformMatrix info: heights: all = " << getHeight())

    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    transMatrix->preTranslated(0.0, -getHeight(), 0.0, matrix);
    myScale = scale;
    getDCS()->setMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(transMatrix);
}

/** In this method, the menu is moved and scaled during userinteractions.
  @return true, so that it is called during each frame
*/
bool coRowMenuHandle::update()
{

    if (interactionA->isRunning())
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
    else if (interactionB->isRunning())
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
    else if (interactionC->isRunning() || myMenu->getMoveInteraction()->isRunning())
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

/// Called when input device leaves the element.
void coRowMenuHandle::miss()
{
    vruiRendererInterface::the()->miss(this);
    titleContainer->setHighlighted(false);
    unregister = true;
}

/** This method is called on intersections of the input device with the
  titlebar.
  @return ACTION_CALL_ON_MISS
*/

int coRowMenuHandle::hit(vruiHit *hit)
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

    if (interactionA->wasStarted() || interactionB->wasStarted() || interactionC->wasStarted() || myMenu->getMoveInteraction()->wasStarted())
    {
        coCombinedButtonInteraction *inter = interactionA->wasStarted() ? interactionA
                                                                        : interactionB->wasStarted() ? interactionB
                                                                                                     : interactionC->wasStarted() ? interactionC
                                                                                                                                  : myMenu->getMoveInteraction();

        myMenu->setMoved(true);

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

void coRowMenuHandle::highlight(bool highlight)
{
    titleContainer->setHighlighted(highlight);
}

const char *coRowMenuHandle::getClassName() const
{
    return "coRowMenuHandle";
}

bool coRowMenuHandle::isOfClassName(const char *classname) const
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

void coRowMenuHandle::updateTitle(const char *newTitle)
{
    titleLabel->setString(newTitle);
}
}
