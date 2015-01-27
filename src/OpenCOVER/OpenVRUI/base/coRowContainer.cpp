/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRowContainer.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>

using std::list;

namespace vrui
{

/// Constructor
coRowContainer::coRowContainer(Orientation orientation)
{
    myDCS = vruiRendererInterface::the()->createTransformNode();
    myDCS->setName("coRowContainer");
    this->orientation = orientation;
    alignment = CENTER;
    Hgap = Vgap = 5.0f;
    Dgap = 0.0f;
    myWidth = 0.001f;
    myHeight = 0.001f;
    myDepth = 0.0f;
    myX = 0.0f;
    myY = 0.0f;
    myZ = 0.0f;

    if (orientation == HORIZONTAL)
        attachment = TOP;
    else
        attachment = LEFT;
}

/// Destructor, does not delete its children
coRowContainer::~coRowContainer()
{
    myDCS->removeAllParents();
    myDCS->removeAllChildren();
    vruiRendererInterface::the()->deleteNode(myDCS);
    myDCS = 0;
}

// Helper function: sum up internal size plus Gaps
float coRowContainer::getExtW() const
{
    // total width, height, depth with surrounding gaps of THIS ELEMENT
    // this one is later set on myself.
    float sumW = 0.0f;

    // now maximum fit the container on desired values
    if (orientation == HORIZONTAL)
    {
        int numElem = (int)elements.size();
        // number of Gaps: 2 for empty
        int numGaps = (numElem) ? (numElem + 1) : 2;
        sumW = getSumW() + numGaps * Hgap; // blank before after
    }
    else if (orientation == VERTICAL)
    {
        sumW = getMaxW() + 2 * Hgap;
    }

    return sumW;
}

float coRowContainer::getExtH() const
{
    // total width, height, depth with surrounding gaps of THIS ELEMENT
    // this one is later set on myself.
    float sumH = 0.0f;

    // now maximum fit the container on desired values
    if (orientation == HORIZONTAL)
    {
        sumH = getMaxH() + 2 * Vgap; // and between all elements
    }
    else if (orientation == VERTICAL)
    {
        int numElem = (int)elements.size();
        // number of Gaps: 2 for empty
        int numGaps = (numElem) ? (numElem + 1) : 2;
        sumH = getSumH() + numGaps * Vgap;
    }
    else
    {
        VRUILOG("coRowContainer::getExtH: strange orientation, sumH uninitialized")
    }

    return sumH;
}

// this function walkes through all children and gets the maximum
// of desired and internal child sizes (which means that a container
// can only 'grow' not 'shrink' over time).
// This geometry is set as own new geometry.
// Then all children are resized (using their resizeToParent() function) and
// positioned inside the container according to the alignment.
// (It is strange that the children are resized _after_ their maximum was
// determined)

//  newWidth   =   new MAXIMAL size of the container,
//  newHEIGHT  =   new MAXIMAL height of the container,

void coRowContainer::resizeToParent(float newWidth, float newHeight, float, bool shrink)
{

    float old_sumW = 0.0f;
    float old_sumH = 0.0f;
    float old_sumD = 0.0f;

    // primary SHRINK TO MIN loop
    if (shrink)
    {
        shrinkToMin();
    }

    // total width, height, depth with surrounding gaps of THIS ELEMENT
    // this one is later set on myself.
    float sumW;
    float sumH;
    float sumD;

    sumD = 2 * Dgap; // Menu is assumes to have no extension in depth

    // if requested larger, take larger size
    sumH = coMax(getExtH(), newHeight);
    sumW = coMax(getExtW(), newWidth);

    // ---------------------------------------------
    // second loop to resize all children
    // it is almost sure that heights will change!

    for (int ctr = 0; ctr < 3; ++ctr)
    {
        old_sumW = sumW;
        old_sumH = sumH;
        old_sumD = sumD;

        // resize children to new bounds
        for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
        {
            if (orientation == HORIZONTAL)
            {
                (*i)->resizeToParent((*i)->getWidth(), sumH - 2 * Vgap, 0.0f, false);
            }
            else if (orientation == VERTICAL)
            {
                (*i)->resizeToParent(sumW - 2 * Hgap, (*i)->getHeight(), 0.0f, false);
            }
        }

        // make sure we use at least the requested space
        sumW = coMax(getExtW(), newWidth);
        sumH = coMax(getExtH(), newHeight);

        // if nothing changed, we can exit here
        if ((sumH == old_sumH) && (sumW == old_sumW) && (sumD == old_sumD))
            break;
    }

    // ---------------------------------------------
    // third loop positions elements inside the container

    if (orientation == HORIZONTAL)
    {

        float currW = Hgap; // leading Gap

        // loop over all elements
        for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
        {
            float elemW = (*i)->getWidth();
            float elemH = (*i)->getHeight();
            float offsetH;

            // modify the height in Horizontal mode
            if (alignment == CENTER)
            {
                offsetH = (sumH - elemH) / 2.0f;
            }
            else if (alignment == MAX) // MAX is not the proper name -> BOTTOM !
            {
                offsetH = sumH - elemH - Vgap;
            }
            else // TOP
            {
                offsetH = Vgap;
            }

            // set position to element and add the leading gap
            (*i)->setPos(currW, offsetH, 0);

            // increment x-position
            currW += elemW + Hgap;
        }
    }
    else if (orientation == VERTICAL)
    {
        float currH = sumH; // H runs from bottom to top, so count backward
        // 1st Gap is created later
        float offsW;

        // loop over all elements
        for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
        {
            float elemW = (*i)->getWidth();
            float elemH = (*i)->getHeight();

            // modify the width in Vertical mode
            if (alignment == CENTER) // centered - offset is half the rest
            {
                offsW = (sumW - elemW) / 2.0f;
            }
            else if (alignment == BOTTOM) // LEFT-align
            {
                offsW = sumW - Hgap - elemW;
            }
            else // LEFT-aligned
            {
                offsW = Hgap;
            }

            // decrement y-position by element height and gap
            currH -= elemH + Vgap;

            // set position to element
            (*i)->setPos(offsW + Hgap, currH, 0);
        }
    }
    else
    {
        VRUILOG("coRowContainer with illegal orientation value")
    }

    // now set own geometry
    myWidth = sumW;
    myHeight = sumH;
    myDepth = sumD;
}

/// shrink all children.
/// adapt own size on maximum of them?

void coRowContainer::shrinkToMin()
{

    // walk through elements
    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        (*i)->shrinkToMin();
    }

    myWidth = getExtW();
    myHeight = getExtH();
    myDepth = 0.0f;
}

/** Appends a child to this container.
    the size of the container is adjusted to accommodate all chidren
  @param el element to add
*/
void coRowContainer::addElement(coUIElement *element)
{
    coUIContainer::addElement(element);
    myDCS->addChild(element->getDCS());
    childResized(false);
}

/** Inserts a child into this container.
    the size of the container is adjusted to accommodate all chidren
  @param el element to add
  @param pos position in the list of elements
    if pos is larger than the number of elements in the list, it
    is appended as last element, if it is less or equal 0, it is inserted
    as first element.
*/
void coRowContainer::insertElement(coUIElement *element, int pos)
{
    int num = (int)elements.size();
    list<coUIElement *>::iterator elementsIterator = elements.begin();

    if (pos < num)
    {
        for (int ctr = 0; ctr < pos; ++ctr)
            ++elementsIterator;

        elements.insert(elementsIterator, element);
        element->setParent(this);
    }
    else
    {
        coUIContainer::addElement(element);
    }

    if (element->getDCS())
        myDCS->addChild(element->getDCS());
    childResized(false);
}

/** Removes a child from this container.
    the size of the container is adjusted after removing the element
  @param el element to remove
*/
void coRowContainer::removeElement(coUIElement *element)
{
    if (element->getDCS() && myDCS)
        myDCS->removeChild(element->getDCS());
    coUIContainer::removeElement(element);
    childResized();
}

/** Hide one of the children.
  @param el element to hide
*/
void coRowContainer::hide(coUIElement *element)
{
    if (element->getDCS() && myDCS)
        myDCS->removeChild(element->getDCS());
}

/** Show one of the children.
  @param el element to show
*/
void coRowContainer::show(coUIElement *element)
{
    if (element->getDCS())
        myDCS->addChild(element->getDCS());
}

/** Change the orientation of the container.
  @param ori new orientation
   default orientation is HORIZONTAL
*/
void coRowContainer::setOrientation(Orientation orientation)
{
    this->orientation = orientation;
    childResized();
}

/** Retrieve the orientation of the container.
 */
int coRowContainer::getOrientation() const
{
    return orientation;
}

/** Change the alignment of the children.
  @param a new alignment
*/
void coRowContainer::setAlignment(int alignment)
{
    this->alignment = alignment;
    childResized();
}

/** Change the horizontal gap between children.
  @param g gap size
*/
void coRowContainer::setHgap(float g)
{
    Hgap = g;
}

/** Change the vertical gap between children.
  @param g gap size
*/
void coRowContainer::setVgap(float g)
{
    Vgap = g;
}

/** Change the depth gap between children.
  @param g gap size
*/
void coRowContainer::setDgap(float g)
{
    Dgap = g;
}

void coRowContainer::setPos(float x, float y, float z)
{
    myX = x;
    myY = y;
    myZ = z;
    myDCS->setTranslation(myX, myY, myZ);
}

vruiTransformNode *coRowContainer::getDCS()
{
    return myDCS;
}

float coRowContainer::getHgap() const
{
    return Hgap;
}

float coRowContainer::getVgap() const
{
    return Vgap;
}

float coRowContainer::getDgap() const
{
    return Dgap;
}

void coRowContainer::setAttachment(int newatt)
{
    int childatt;

    switch (newatt)
    {
    case TOP:
        setOrientation(HORIZONTAL);
        if ((alignment == MIN) || (alignment == MAX))
            setAlignment(MAX);
        // doesn't work
        break;

    case BOTTOM:
        setOrientation(HORIZONTAL);
        if ((alignment == MIN) || (alignment == MAX))
            setAlignment(MIN);
        break;

    case LEFT:
        setOrientation(VERTICAL);
        if ((alignment == MIN) || (alignment == MAX))
            setAlignment(MAX);
        // doesn't work
        break;

    case RIGHT:
        setOrientation(VERTICAL);
        if ((alignment == MIN) || (alignment == MAX))
            setAlignment(MIN);
        break;

    case REPLACE:
        setOrientation(VERTICAL);
        if ((alignment == MIN) || (alignment == MAX))
            setAlignment(MIN);
        break;
    }

    // simply copy attachment to children
    childatt = newatt;

    // set all children attachments
    // browse all items
    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        // update and proceed (internal increment of element number)
        (*i)->setAttachment(childatt);
    }

    // set own attachment, forget about Orientation!
    attachment = newatt;
}

int coRowContainer::getAttachment() const
{
    return attachment;
}

const char *coRowContainer::getClassName() const
{
    return "coRowContainer";
}

bool coRowContainer::isOfClassName(const char *classname) const
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
