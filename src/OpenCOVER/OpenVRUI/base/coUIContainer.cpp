/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>

#include <OpenVRUI/util/vruiLog.h>

using std::list;

namespace vrui
{

/// Constructor
coUIContainer::coUIContainer()
{
    xAlignment = CENTER;
    yAlignment = CENTER;
    zAlignment = CENTER;
}

/// Destructor
coUIContainer::~coUIContainer()
{

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        (*i)->setParent(0);
    }
}

/** Appends a child to this container.
  @param el element to add
*/
void coUIContainer::addElement(coUIElement *el)
{
    coUIContainer *test = this;

    // Make sure this element is not its own parent:
    while ((test = test->getParent()))
    {
        if (test == this)
        {
            VRUILOG("coUIContainer::addElement warn: circular parentship prevented")
            return;
        }
    }

    elements.push_back(el);
    el->setParent(this);
}

/** Adds the specified element to the scenegraph
    if it has previously been removed.
  @param el element to add
*/
void coUIContainer::showElement(coUIElement *el)
{
    if (getDCS() && el->getDCS())
    {
        getDCS()->addChild(el->getDCS());
    }
}

/** 
  @param el element to remove
*/
void coUIContainer::removeElement(coUIElement *el)
{
    elements.remove(el);
    el->setParent(0);
}

/** 
  @param el element to remove
*/
void coUIContainer::removeLastElement()
{
    fprintf(stderr, "coUIContainer::removeLastElement\n");
    list<coUIElement *>::iterator it = elements.end();
    it--;
    it--;
    fprintf(stderr, "last element is %s\n", (*it)->getUniqueName());
    elements.remove(*it);
    (*it)->setParent(0);
}

/** set the alignment in X direction of the children
    default is CENTER
*/
void coUIContainer::setXAlignment(int a)
{
    xAlignment = a;
}

/** set the alignment in Y direction of the children
    default is CENTER
*/
void coUIContainer::setYAlignment(int a)
{
    yAlignment = a;
}

/** set the alignment in Z direction of the children
    default is CENTER
*/
void coUIContainer::setZAlignment(int a)
{
    zAlignment = a;
}

// documentet in coUIElement.cpp
void coUIContainer::resizeToParent(float, float, float, bool /*shrink*/)
{
}

void coUIContainer::shrinkToMin()
{
    // walk through elements
    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        (*i)->shrinkToMin();
    }
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void coUIContainer::setEnabled(bool en)
{

    coUIElement::setEnabled(en);

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        (*i)->setEnabled(en);
    }
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void coUIContainer::setHighlighted(bool hl)
{

    coUIElement::setHighlighted(hl);

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        (*i)->setHighlighted(hl);
    }
}

const char *coUIContainer::getClassName() const
{
    return "coUIContainer";
}

bool coUIContainer::isOfClassName(const char *classname) const
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
            return coUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

// +++++ calculate maximal sizes

// @@@ calculate size in "add" ??

float coUIContainer::getMaxH() const
{

    float maxH = 0.0f;

    for (list<coUIElement *>::const_iterator i = elements.begin(); i != elements.end(); ++i)
    {
        float currH = (*i)->getHeight();
        if (currH > maxH)
            maxH = currH;
    }

    return maxH;
}

float coUIContainer::getMaxW() const
{

    float maxW = 0.0f;

    for (list<coUIElement *>::const_iterator i = elements.begin(); i != elements.end(); ++i)
    {
        float currW = (*i)->getWidth();
        if (currW > maxW)
            maxW = currW;
    }

    return maxW;
}

float coUIContainer::getMaxD() const
{

    float maxD = 0.0f;

    for (list<coUIElement *>::const_iterator i = elements.begin(); i != elements.end(); ++i)
    {
        float currD = (*i)->getDepth();
        if (currD > maxD)
            maxD = currD;
    }

    return maxD;
}

// +++++ calculate sum sizes

float coUIContainer::getSumH() const
{

    float sumH = 0.0f;

    for (list<coUIElement *>::const_iterator i = elements.begin(); i != elements.end(); ++i)
    {
        sumH += (*i)->getHeight();
    }

    return sumH;
}

float coUIContainer::getSumW() const
{

    float sumW = 0.0f;

    for (list<coUIElement *>::const_iterator i = elements.begin(); i != elements.end(); ++i)
    {
        sumW += (*i)->getWidth();
    }

    return sumW;
}

float coUIContainer::getSumD() const
{

    float sumD = 0.0f;

    for (list<coUIElement *>::const_iterator i = elements.begin(); i != elements.end(); ++i)
    {
        sumD += (*i)->getDepth();
    }

    return sumD;
}

void coUIContainer::resizeGeometry()
{

    if (!uiElementProvider)
    {
        VRUILOG("coUIElement::resizeGeometry info creating uiElementProvider for " << getClassName())
        uiElementProvider = vruiRendererInterface::the()->createUIElementProvider(this);
    }

    uiElementProvider->resizeGeometry();
}
}
