/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coBackground.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>

#include <list>
using std::list;

namespace vrui
{

/// Constructor
coBackground::coBackground()
{
    myWidth = myHeight = 0.001f;
    myZOffset = 5;
    zAlignment = MIN;
    myDepth = 0;
    minDepth = 0;
    minWidth = 0;
    minHeight = 0;
}

/// Destructor
coBackground::~coBackground()
{

    vruiTransformNode *myDCS = getDCS();

    myDCS->removeAllParents();
    myDCS->removeAllChildren();

    myDCS = 0;
}

void coBackground::resizeToParent(float x, float y, float z, bool shrink)
{
    //  fprintf(stderr,"coBackground::resizeToParent(%f,%f,%f,%s)\n",
    //                 x,y,z,((shrink)?"true":"false"));

    // only shrink when required
    // (quite stupid here because it is not used for own resizing,
    //  but maybe children need it...)
    if (shrink)
    {
        shrinkToMin();
    }

    myWidth = coMax(x, minWidth);
    myHeight = coMax(y, minHeight);
    myDepth = coMax(z, minDepth);

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        (*i)->resizeToParent(x, y, z);
    }

    float mw = 0;
    float mh = 0;
    float md = 0;

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        if ((*i)->getWidth() > mw)
        {
            mw = (*i)->getWidth();
        }
        if ((*i)->getHeight() > mh)
        {
            mh = (*i)->getHeight();
        }
        if ((*i)->getDepth() > md)
        {
            md = (*i)->getDepth();
        }
    }

    myWidth = coMax(mw, x);
    myHeight = coMax(mh, y);
    myDepth = coMax(md, z);
    myWidth = coMax(myWidth, minWidth);
    myHeight = coMax(myHeight, minHeight);
    myDepth = coMax(myDepth, minDepth);
    resizeGeometry();
    realign();

    // fprintf(stderr,"W/H/D = %8f,%8f,%8f\n",myWidth,myHeight,myDepth);
}

/// Shrinks itself! This includes shrinking all children.
void coBackground::shrinkToMin()
{

    float mw = 0;
    float mh = 0;
    float md = 0;

    // walk through elements

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        // shrink them
        (*i)->shrinkToMin();

        // get max
        if ((*i)->getWidth() > mw)
        {
            mw = (*i)->getWidth();
        }

        if ((*i)->getHeight() > mh)
        {
            mh = (*i)->getHeight();
        }

        if ((*i)->getDepth() > md)
        {
            md = (*i)->getDepth();
        }
    }

    // set own size to max
    myWidth = coMax(minWidth, mw);
    myHeight = coMax(minHeight, mh);
    myDepth = coMax(minDepth, md);
    resizeGeometry();
}

/** set the width of the background element explicitely
    @param f new width
*/
void coBackground::setWidth(float f)
{
    myWidth = coMax(minWidth, f);
    if (getParent())
    {
        getParent()->childResized();
    }
}

/** set the height of the background element explicitely
    @param f new height
*/
void coBackground::setHeight(float f)
{
    myHeight = coMax(minHeight, f);
    if (getParent())
    {
        getParent()->childResized();
    }
}

/** set the depth of the background element explicitely
    @param f new depth
*/
void coBackground::setDepth(float f)
{
    myDepth = coMax(minDepth, f);
    if (getParent())
    {
        getParent()->childResized();
    }
}

/** set the minimum width of the background element
    @param f minimum width
*/
void coBackground::setMinWidth(float f)
{
    myWidth = coMax(myWidth, f);
    minWidth = f;
    if (getParent())
    {
        getParent()->childResized();
    }
}

/** set the minimum height of the background element
    @param f minimum height
*/
void coBackground::setMinHeight(float f)
{
    myHeight = coMax(myHeight, f);
    minHeight = f;
    if (getParent())
    {
        getParent()->childResized();
    }
}

/** set the minimum depth of the background element
    @param f minimum depth
*/
void coBackground::setMinDepth(float f)
{
    myDepth = coMax(myDepth, f);
    minDepth = f;
    if (getParent())
    {
        getParent()->childResized();
    }
}

/** Change the Z offset of the child, default is 5mm
 @param o new Z offset
*/
void coBackground::setZOffset(float offset)
{
    myZOffset = offset;
    realign();
}

void coBackground::setSize(float nw, float nh, float nd)
{
    myWidth = coMax(minWidth, nw);
    myHeight = coMax(minHeight, nh);
    myDepth = coMax(minDepth, nd);
    resizeGeometry();
    if (getParent())
    {
        getParent()->childResized();
    }
}

void coBackground::setSize(float s)
{
    myWidth = s;
    myHeight = s;
    myDepth = s;
    resizeGeometry();
    if (getParent())
    {
        getParent()->childResized();
    }
}

/** this method is called whenever the geometry of the
    background has to be recomputed
*/
void coBackground::resizeGeometry()
{
    createGeometry();
    uiElementProvider->resizeGeometry();
}

/// centers the child
void coBackground::realign()
{

    float xp;
    float yp;
    float zp;

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        xp = 0.0f;
        yp = 0.0f;
        zp = 0.0f;

        if (xAlignment == CENTER)
        {
            xp = (getWidth() - (*i)->getWidth()) / 2.0f;
        }
        else if (xAlignment == MAX)
        {
            xp = getWidth() - (*i)->getWidth();
        }

        if (yAlignment == CENTER)
        {
            yp = (getHeight() - (*i)->getHeight()) / 2.0f;
        }
        else if (yAlignment == MAX)
        {
            yp = getHeight() - (*i)->getHeight();
        }

        if (zAlignment == CENTER)
        {
            zp = (getDepth() - (*i)->getDepth()) / 2.0f;
        }
        else if (zAlignment == MAX)
        {
            zp = getDepth() - (*i)->getDepth();
        }

        (*i)->setPos(xp, yp, zp + myZOffset);
    }
}

void coBackground::removeElement(coUIElement *el)
{
    if (el->getDCS() && getDCS())
    {
        getDCS()->removeChild(el->getDCS());
    }
    coUIContainer::removeElement(el);
    childResized();
}

void coBackground::addElement(coUIElement *el)
{
    coUIContainer::addElement(el);
    if (el->getDCS())
    {
        getDCS()->addChild(el->getDCS());
    }
    childResized(false);
}

void coBackground::setPos(float x, float y, float z)
{
    myX = x;
    myY = y;
    myZ = z;
    getDCS()->setTranslation(myX, myY, myZ);
}

void coBackground::setEnabled(bool en)
{
    coUIContainer::setEnabled(en);
}

void coBackground::setHighlighted(bool hl)
{
    coUIContainer::setHighlighted(hl);
}

const char *coBackground::getClassName() const
{
    return "coBackground";
}

bool coBackground::isOfClassName(const char *classname) const
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
            return coUIContainer::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

/// There is no sense to make virtual functions inline...
float coBackground::getWidth() const
{
    return myWidth;
}

float coBackground::getHeight() const
{
    return myHeight;
}

float coBackground::getDepth() const
{
    return myDepth;
}

float coBackground::getXpos() const
{
    return myX;
}

float coBackground::getYpos() const
{
    return myY;
}

float coBackground::getZpos() const
{
    return myZ;
}
}
