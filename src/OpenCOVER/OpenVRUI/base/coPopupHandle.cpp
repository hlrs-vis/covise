/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif

#include <OpenVRUI/coPopupHandle.h>

#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <config/CoviseConfig.h>

#define HANDLESCALE 20

using namespace std;

namespace vrui
{

/** Creates a titlebar for PopupDialogs.
    @param name text displayed in title bar
*/
coPopupHandle::coPopupHandle(const string &name)
    : coRowContainer(coRowContainer::VERTICAL)
{

    // FIXME: Common base class
    localPickPosition[0] = localPickPosition[1] = localPickPosition[2] = 0.0f;
    pickPosition[0] = pickPosition[1] = pickPosition[2] = 0.0f;
    title = name;
    myScale = 1;
    // Create title bar:
    titleBackground = new coColoredBackground(coUIElement::HANDLE_BACKGROUND_NORMAL, coUIElement::HANDLE_BACKGROUND_HIGHLIGHTED, coUIElement::HANDLE_BACKGROUND_DISABLED);
    titleLabel = new coLabel();
    titleLabel->setString(title);
    closeButton = new coPushButton(new coFlatButtonGeometry("UI/close"), this);
    minmaxButton = new coToggleButton(new coToggleButtonGeometry("UI/minmax"), this);
    titleContainer = new coMenuContainer(coRowContainer::HORIZONTAL);
    titleContainer->addElement(titleLabel);
    titleContainer->addElement(minmaxButton);
    titleContainer->addElement(closeButton);
    titleContainer->setNumAlignedMin(1);
    titleFrame = new coFrame("UI/Frame");
    titleFrame->addElement(titleBackground);
    titleBackground->addElement(titleContainer);

    titleFrame->fitToParent();

    vruiRendererInterface::the()->getUpdateManager()->add(this);

    invStartHandTrans = vruiRendererInterface::the()->createMatrix();
    startPosition = vruiRendererInterface::the()->createMatrix();

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

    vruiRendererInterface::the()->deleteMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(rotateMatrix);
    vruiRendererInterface::the()->deleteMatrix(scaleMatrix);
    vruiRendererInterface::the()->deleteMatrix(matrix);
    vruiRendererInterface::the()->deleteMatrix(rxMatrix);

    addElement(titleFrame);

    minimized = false;

    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "MenuHandle", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonB, "MenuHandle", coInteraction::Menu);
    interactionC = new coCombinedButtonInteraction(coInteraction::ButtonC, "MenuHandle", coInteraction::Menu);
}

/// Destructor
coPopupHandle::~coPopupHandle()
{

    // FIXME: Common base class

    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    vruiRendererInterface::the()->deleteMatrix(invStartHandTrans);
    vruiRendererInterface::the()->deleteMatrix(startPosition);

    delete interactionA;
    delete interactionB;
    delete interactionC;
}

void coPopupHandle::resizeToParent(float newWidth, float newHeight, float newDepth, bool shrink)
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

void coPopupHandle::shrinkToMin()
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

void coPopupHandle::buttonEvent(coButton *button)
{
    if (button == closeButton)
    {
        // TODO call a menu listener to inform the application to close this popup window
        setVisible(false);
    }
    else
    {
        if (button == minmaxButton)
        {
            list<coUIElement *>::iterator element = elements.begin();
            ++element; // skip the titlebar

            for (; element != elements.end(); ++element)
            {
                if (minimized)
                {
                    show(*element);
                }
                else
                {
                    hide(*element);
                }
            }
            minimized = !minimized;
        }
    }
}

/** scales the menu.
   Default scale is 1, the menu is scaled around the center of the titlebar
  */
void coPopupHandle::setScale(float s)
{
    // FIXME: Common base class
    vruiMatrix *currentMatrix = 0;
    vruiMatrix *newMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *itMatrix = vruiRendererInterface::the()->createMatrix();

    itMatrix->makeTranslate(-getWidth() / 2.0, -getHeight(), 0.0);
    transMatrix->makeTranslate(getWidth() / 2.0, getHeight(), 0.0);
    currentMatrix = getDCS()->getMatrix();
    scaleMatrix->makeScale(s / myScale, s / myScale, s / myScale);

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
float coPopupHandle::getScale() const
{
    // FIXME: Common base class
    return myScale;
}

/** set a new menu position and orientation matrix.
   the Menu is scaled to the correct size after applying this matrix
  */
void coPopupHandle::setTransformMatrix(vruiMatrix *matrix)
{

    // FIXME: Common base class
    vruiMatrix *translatedMatrix = vruiRendererInterface::the()->createMatrix();

    translatedMatrix->preTranslated(0.0, titleFrame->getHeight() - getHeight(), 0.0, matrix);

    vruiMatrix *newMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *itMatrix = vruiRendererInterface::the()->createMatrix();

    itMatrix->makeTranslate(-getWidth() / 2.0, -getHeight(), 0.0);
    transMatrix->makeTranslate(getWidth() / 2.0, getHeight(), 0.0);
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
void coPopupHandle::setTransformMatrix(vruiMatrix *matrix, float scale)
{
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    transMatrix->preTranslated(0.0, titleFrame->getHeight() - getHeight(), 0.0, matrix);
    myScale = scale;
    getDCS()->setMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(transMatrix);
}

/** In this method, the menu is moved and scaled during userinteractions.
  @return true, so that it is called during each frame
*/
bool coPopupHandle::update()
{
    // FIXME: Common base class

    if (interactionA && interactionA->isRunning())
    {

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
    else if (interactionB && interactionB->isRunning())
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
    else if (interactionC && interactionC->isRunning())
    {
        // right Button is moved
        vruiMatrix *current = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *hand = vruiRendererInterface::the()->createMatrix();

        current->makeIdentity();
        hand->makeIdentity();

        current->mult(startPosition)->mult(hand->mult(invStartHandTrans)->mult(interactionC->getHandMatrix()));

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
void coPopupHandle::miss()
{
    // FIXME: Common base class
    titleContainer->setHighlighted(false);
    unregister = true;
}

/** This method is called on intersections of the input device with the
  titlebar.
  @return ACTION_CALL_ON_MISS
*/
int coPopupHandle::hit(vruiHit *hit)
{

    // FIXME: Common base class

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
                                                                        : interactionB->wasStarted() ? interactionB
                                                                                                     : interactionC;
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

const char *coPopupHandle::getClassName() const
{
    return "coPopupHandle";
}

bool coPopupHandle::isOfClassName(const char *classname) const
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
}
