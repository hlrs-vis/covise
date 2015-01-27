/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coFrame.h>

#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <list>
using std::list;
using std::string;

namespace vrui
{

/** Constructor
 @param name Texture name, default is "UI/Frame",
 a white frame with round edges
*/
coFrame::coFrame(const string &name)
{
    textureName = name;
    myWidth = 0.001f;
    myHeight = 0.001f;
    myDepth = 0.0f;
    bw = bh = 10.0f;
    bd = 0.0f;
    fitParent = false;
}

/** Destructor
 */
coFrame::~coFrame()
{

    vruiNode *myDCS = getDCS();

    myDCS->removeAllChildren();
    myDCS->removeAllParents();
}

/// maximize its size to always fit into its parent container
void coFrame::fitToParent()
{
    fitParent = true;
}

/// fit tight around its child, this is the default behaviour
void coFrame::fitToChild()
{
    fitParent = false;
}

/** set the width of the frame explicitely
    @param f outer width
*/
void coFrame::setWidth(float f)
{
    myWidth = f;
    resizeGeometry();
}

/** set the height of the frame explicitely
    @param f outer height
*/
void coFrame::setHeight(float f)
{
    myHeight = f;
    resizeGeometry();
}

/** set the depth of the frame explicitely
    @param f outer depth
*/
void coFrame::setDepth(float f)
{
    myDepth = f;
    resizeGeometry();
}

/** Change the border width
    @param f new border width
*/
void coFrame::setBorderWidth(float f)
{
    bw = f;
    resizeGeometry();
}

/** Change the border height
    @param f new border height
*/
void coFrame::setBorderHeight(float f)
{
    bh = f;
    resizeGeometry();
}

/** Change the border depth
    @param f new border depth
*/
void coFrame::setBorderDepth(float f)
{
    bw = f;
    resizeGeometry();
}

void coFrame::setSize(float nw, float nh, float nd)
{
    myWidth = nw;
    myHeight = nh;
    myDepth = nd;
    resizeGeometry();
    if (getParent())
    {
        getParent()->childResized();
    }
}

void coFrame::setSize(float s)
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

/** Change the border width, hight and depth
    @param nw new border width
    @param nh new border hight
    @param nd new border depth
*/
void coFrame::setBorder(float nw, float nh, float nd)
{
    float iw = myWidth - 2 * bw;
    float ih = myHeight - 2 * bh;
    float id = myDepth - 2 * bd;
    bw = nw;
    bh = nh;
    bd = nd;
    setSize(iw + 2 * bw, ih + 2 * bh, id + 2 * bd);
}

void coFrame::resizeToParent(float x, float y, float z, bool shrink)
{

    // only shrink when required
    // (quite stupid here because it is not used for own resizing,
    //  but maybe children need it...)
    if (shrink)
    {
        shrinkToMin();
    }

    if (fitParent)
    {

        myWidth = x;
        myHeight = y;
        myDepth = z;

        for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
        {
            (*i)->resizeToParent(x - 2 * bw, y - 2 * bh, z - 2 * bd, false);
        }

        float mw = 0.0f;
        float mh = 0.0f;
        float md = 0.0f;

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
        myWidth = coMax(mw + 2 * bw, x);
        myHeight = coMax(mh + 2 * bh, y);
        myDepth = coMax(md + 2 * bd, z);
        resizeGeometry();
        realign();
    }
    else
    { /// change the geometry to fit around the child

        float mw = 0.0f;
        float mh = 0.0f;
        float md = 0.0f;

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

        myWidth = mw + 2 * bw;
        myHeight = mh + 2 * bh;
        myDepth = md + 2 * bd;

        resizeGeometry();
    }
}

/// Shrinks itself! This includes shrinking all children.
void coFrame::shrinkToMin()
{

    float mw = 0.0f;
    float mh = 0.0f;
    float md = 0.0f;

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
    myWidth = mw;
    myHeight = mh;
    myDepth = md;
}

/// reposition the children to fit into the frame
void coFrame::realign()
{
    float xp = 0.0f;
    float yp = 0.0f;
    float zp = 0.0f;

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {

        xp = 0.0f;
        yp = 0.0f;
        zp = 0.0f;
        if (xAlignment == CENTER)
        {
            xp = ((getWidth() - (2 * bw)) - (*i)->getWidth()) / 2.0f;
        }
        else if (xAlignment == MAX)
        {
            xp = (getWidth() - (2 * bw)) - (*i)->getWidth();
        }
        if (yAlignment == CENTER)
        {
            yp = ((getHeight() - (2 * bh)) - (*i)->getHeight()) / 2.0f;
        }
        else if (yAlignment == MAX)
        {
            yp = (getHeight() - (2 * bh)) - (*i)->getHeight();
        }
        if (zAlignment == CENTER)
        {
            zp = ((getDepth() - (2 * bd)) - (*i)->getDepth()) / 2.0f;
        }
        else if (zAlignment == MAX)
        {
            zp = (getDepth() - (2 * bd)) - (*i)->getDepth();
        }
        (*i)->setPos(xp + bw, yp + bh, zp + bd);
    }
}

void coFrame::removeElement(coUIElement *el)
{
    if (el->getDCS() && getDCS())
    {
        getDCS()->removeChild(el->getDCS());
    }

    coUIContainer::removeElement(el);
    childResized();
}

void coFrame::addElement(coUIElement *el)
{

    coUIContainer::addElement(el);
    if (el->getDCS())
    {
        getDCS()->insertChild(0, el->getDCS());
    }
    el->setPos(bw, bh, bd);
    childResized(false);
}

void coFrame::setPos(float x, float y, float z)
{

    myX = x;
    myY = y;
    myZ = z;
    getDCS()->setTranslation(myX, myY, myZ);
}

const char *coFrame::getClassName() const
{
    return "coFrame";
}

bool coFrame::isOfClassName(const char *classname) const
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
}
