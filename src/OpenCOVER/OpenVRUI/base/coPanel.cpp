/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPanelGeometry.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <list>

using namespace std;

namespace vrui
{

#define BORDERWIDTH 0.5f
/** Constructor
   @param geom Panel geometry
*/
coPanel::coPanel(coPanelGeometry *geometry)
{

    myDCS = vruiRendererInterface::the()->createTransformNode();
    myDCS->setName("coPanel");

    myChildDCS = vruiRendererInterface::the()->createTransformNode();
    myPosDCS = vruiRendererInterface::the()->createTransformNode();

    vruiIntersection::getIntersectorForAction("coAction")->add(myDCS, this);

    myGeometry = geometry;

    if (geometry)
        geometry->attachGeode(myPosDCS);

    myDCS->addChild(myPosDCS);
    myDCS->addChild(myChildDCS);

    myX = myY = myZ = 0.0f;

    myHeight = 10.0f;
    myWidth = 10.0f;

    contentHeight = myHeight - 2 * BORDERWIDTH;
    contentWidth = myWidth - 2 * BORDERWIDTH;

    scale = 1.0f;
}

/** Destructor
 */
coPanel::~coPanel()
{

    vruiIntersection::getIntersectorForAction("coAction")->remove(this);

    myDCS->removeAllParents();
    myDCS->removeAllChildren();

    myChildDCS->removeAllParents();
    myChildDCS->removeAllChildren();

    vruiRendererInterface::the()->deleteNode(myDCS);
    vruiRendererInterface::the()->deleteNode(myPosDCS);
    vruiRendererInterface::the()->deleteNode(myChildDCS);
    myDCS = 0;
}

/** set the scale factor of this panel and all its children
 */
void coPanel::setScale(float s)
{
    scale = s;
    myChildDCS->setScale(scale, scale, scale);
}

void coPanel::addElement(coUIElement *el)
{
    coUIContainer::addElement(el);
    if (el->getDCS())
    {
        myChildDCS->addChild(el->getDCS());
    }
    childResized(false);
}

/** hide one of the children
 */
void coPanel::hide(coUIElement *el)
{
    if (el->getDCS())
    {
        myChildDCS->removeChild(el->getDCS());
    }
}

/** show one of the children
 */
void coPanel::show(coUIElement *el)
{
    if (el->getDCS())
    {
        myChildDCS->addChild(el->getDCS());
    }
}

void coPanel::showElement(coUIElement *el)
{
    if (el->getDCS())
    {
        myChildDCS->addChild(el->getDCS());
    }
}

/** resizes the panel to accommodate all children
 */
void coPanel::resize()
{
    float maxX = -100000;
    float maxY = -100000;
    float minX = 100000;
    float minY = 100000;
    float minZ = 100000;

    float xOff, yOff = 0.f;
    //float zOff = 0.f;

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        if (maxX < (*i)->getXpos() + (*i)->getWidth())
            maxX = (*i)->getXpos() + (*i)->getWidth();

        if (maxY < (*i)->getYpos() + (*i)->getHeight())
            maxY = (*i)->getYpos() + (*i)->getHeight();

        if (minX > (*i)->getXpos())
            minX = (*i)->getXpos();

        if (minY > (*i)->getYpos())
            minY = (*i)->getYpos();

        if (minZ > (*i)->getZpos())
            minZ = (*i)->getZpos();
    }

    contentHeight = (maxY - minY);
    myHeight = contentHeight + (float)(2 * BORDERWIDTH);
    xOff = minX - (float)BORDERWIDTH;
    yOff = minY - (float)BORDERWIDTH;

    //   if(myGeometry)
    //      zOff = minZ - myGeometry->getDepth();

    contentWidth = (maxX - minX);
    myWidth = contentWidth + (float)(2 * BORDERWIDTH);

    //myChildDCS->setTranslation(-xOff * scale, -yOff * scale, -zOff * scale);
    myChildDCS->setTranslation(-xOff * scale, -yOff * scale, 0.0);
    myChildDCS->setScale(scale, scale, scale);

    if (myGeometry)
        myPosDCS->setScale(getWidth() / myGeometry->getWidth(), getHeight() / myGeometry->getHeight(), 1.0);
}

void coPanel::setPos(float x, float y, float)
{
    resize();
    myX = x;
    myY = y;
    myDCS->setTranslation(myX, myY, myZ);
}

/**hit is called whenever the panel is intersected
 @param hitPoint point of intersection in world coordinates
 @param hit hit structure to query other information like normal
 @return ACTION_CALL_ON_MISS if you want miss to be called
otherwise return ACTION_DONE
*/
int coPanel::hit(vruiHit *hit)
{

    //VRUILOG("coPanel::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    return ACTION_CALL_ON_MISS;
}

/**miss is called once after a hit, if the panel is not intersected
 anymore*/
void coPanel::miss()
{
    vruiRendererInterface::the()->miss(this);
}

const char *coPanel::getClassName() const
{
    return "coPanel";
}

bool coPanel::isOfClassName(const char *classname) const
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

vruiTransformNode *coPanel::getDCS()
{
    return myDCS;
}
}
