/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>

#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/sginterface/vruiUIElementProvider.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiGroupNode.h>

#include <string.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

/// Constructor.
vruiMatrix *coUIElement::getMatrixFromPositionHprScale(float x, float y, float z, float h, float p, float r, float scale)
{
    vruiMatrix *transMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rotateMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *scaleMatrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *matrix = vruiRendererInterface::the()->createMatrix();
    vruiMatrix *rxMatrix = vruiRendererInterface::the()->createMatrix();

    transMatrix->makeTranslate(x, y, z);
    rotateMatrix->makeEuler(h, p, r);
    scaleMatrix->makeScale(scale, scale, scale);
    rxMatrix->makeEuler(0.0, 90.0, 0.0);

    matrix->makeIdentity();
    matrix->mult(rxMatrix);
    matrix->mult(scaleMatrix);
    matrix->mult(rotateMatrix);
    matrix->mult(transMatrix);

    vruiRendererInterface::the()->deleteMatrix(transMatrix);
    vruiRendererInterface::the()->deleteMatrix(rotateMatrix);
    vruiRendererInterface::the()->deleteMatrix(scaleMatrix);
    vruiRendererInterface::the()->deleteMatrix(rxMatrix);

    return matrix;
}

coUIElement::coUIElement()
{

    parentContainer = 0;
    userData = 0;
    enabled = true;
    visible = false;
    highlighted = false;

    xScaleFactor = yScaleFactor = zScaleFactor = 1;

    Unique_Name = "";

    uiElementProvider = 0;
    transformMatrix = 0;
}

/// Destructor
coUIElement::~coUIElement()
{
    if (parentContainer)
    {
        parentContainer->removeElement(this);
    }

    delete uiElementProvider;

    if (vruiRendererInterface::the())
        vruiRendererInterface::the()->deleteMatrix(transformMatrix);
}

/** Set the current userdata object.
    @param user new userdata object
*/
void coUIElement::setUserData(coUIUserData *user)
{
    userData = user;
}

/** This method is called by children whenever they change their size
    Implementations of this method should call the childResized() of their
    parent container.
    This method calls shrinkToMin if shrink is true (the default)
    and resizeToParent to trigger a resize on all children if
    this is the topmost element in the tree
*/
void coUIElement::childResized(bool shrink)
{
    if (getParent())
    {
        getParent()->childResized(shrink);
    }
    else
    {
        if (shrink)
        {
            shrinkToMin();
        }
        resizeToParent(getWidth(), getHeight(), getDepth(), false);
    }
}

/** Returns the current userdata object.
  @return userdata object, default NULL
*/
coUIUserData *coUIElement::getUserData() const
{
    return userData;
}

/** Get z position.
  @return z position
*/
float coUIElement::getZpos() const
{
    return 0.0f;
}

/** Get z axis object size.
  @return z axis object size
*/
float coUIElement::getDepth() const
{
    return 0.0f;
}

/** Set parent container.
  @param c parent container
*/
void coUIElement::setParent(coUIContainer *c)
{
    parentContainer = c;
    visible = true;
}

/** Set UI element size, use equal values for all dimensions.
  @param s size = scaling factor (1 is default)
*/
void coUIElement::setSize(float s)
{
    xScaleFactor = yScaleFactor = zScaleFactor = s;
    getDCS()->setScale(s);
}

/** Set UI element size. Use different values for all dimensions.
  @param xs,ys,zs size = scaling factor for respective dimension (1 is default)
*/
void coUIElement::setSize(float xs, float ys, float zs)
{
    xScaleFactor = xs;
    yScaleFactor = ys;
    zScaleFactor = zs;
    getDCS()->setScale(xScaleFactor, yScaleFactor, zScaleFactor);
}

/** Get parent container.
  @return parent container
*/
coUIContainer *coUIElement::getParent()
{
    return parentContainer;
}

/**
  This method is called by containers after they resized to
  allow children to adjust their geometry to the new parents' size
  Children must not call childResized() of their parent here,
  as this could lead to an infinite loop.
  Derived elements do not have to implement this method.
  @param newWidth, newHeight, newDepth desired size
*/
void coUIElement::resizeToParent(float, float, float, bool)
{
}

/**
 This method is called to shrink the element to its smallest
 size. Usually this is done before resizing containers.
 */
void coUIElement::shrinkToMin()
{
}

/** Set activation state.
  @param en true = element enabled
*/
void coUIElement::setEnabled(bool en)
{
    enabled = en;
}

/** Set highlight state.
  @param hl true = element highlighted
*/
void coUIElement::setHighlighted(bool hl)
{
    highlighted = hl;
}

/** Set element visibility.
  @param newState true = element visible
*/
void coUIElement::setVisible(bool newState)
{
    if (visible == newState)
        return; // state is already ok
    visible = newState;
    if (visible)
    {
        if (parentContainer)
        {
            parentContainer->showElement(this);
        }
        else
        {
            if (vruiRendererInterface::the())
                vruiRendererInterface::the()->getMenuGroup()->addChild(getDCS());
        }
    }
    else
    {
        while (getDCS() && getDCS()->getParent())
        {
            getDCS()->getParent()->removeChild(getDCS());
        }
    }
}

/** Get activation state.
  @return activation state (true = enabled)
*/
bool coUIElement::isEnabled() const
{
    return enabled;
}

/** Get highlight state.
  @return highlight state (true = highlighted)
*/
bool coUIElement::isHighlighted() const
{
    return highlighted;
}

/** Get visibility state.
  @return visibility state (true = visible)
*/
bool coUIElement::isVisible() const
{
    return visible;
}

void coUIElement::setUniqueName(const char *newname)
{
    Unique_Name = newname;
}

const char *coUIElement::getUniqueName() const
{
    if (Unique_Name == "")
        return 0;
    else
        return Unique_Name.c_str();
}

const char *coUIElement::getClassName() const
{
    return "coUIElement";
}

bool coUIElement::isOfClassName(const char *classname) const
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
            return false;
            // coUIElement already is the root class. Else:
            // return co::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void coUIElement::createGeometry()
{

    if (!uiElementProvider && vruiRendererInterface::the())
    {
        //VRUILOG("coUIElement::createGeometry info creating uiElementProvider for " << getClassName())
        uiElementProvider = vruiRendererInterface::the()->createUIElementProvider(this);
        if (uiElementProvider)
            uiElementProvider->createGeometry();
    }
}

vruiTransformNode *coUIElement::getDCS()
{

    createGeometry();
    if (uiElementProvider)
        return uiElementProvider->getDCS();
    return NULL;
}

const vruiMatrix *coUIElement::getTransformMatrix()
{

    createGeometry();

    if (!transformMatrix)
    {
        if (!vruiRendererInterface::the())
            return NULL;

        transformMatrix = vruiRendererInterface::the()->createMatrix();
    }

    transformMatrix->makeIdentity();
    getDCS()->convertToWorld(transformMatrix);

    return transformMatrix;
}

void coUIElement::resizeGeometry()
{

    createGeometry();
    uiElementProvider->resizeGeometry();
}
}
