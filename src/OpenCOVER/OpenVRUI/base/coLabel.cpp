/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coUIContainer.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#define BORDERWIDTH 5.0

using std::string;

namespace vrui
{

/// Constructor.
coLabel::coLabel(const std::string &labelText)
{
    labelString = labelText;
    justify = LEFT;
    myX = 0.0f;
    myY = 0.0f;
    myZ = 0.0f;
    myHeight = 20.0f;
    myWidth = 20.0f;
    myDepth = 0.0f;
    fontSize = 50.0f;
    direction = HORIZONTAL;

    resize();
}

/// Destructor.
coLabel::~coLabel()
{
}

/** Set label size using one size for all dimensions.
  @param s size in mm
*/
void coLabel::setSize(float s)
{
    myWidth = s;
    myHeight = s;
    myDepth = s;

    resizeGeometry();
}

/** Set label size using different sizes for all dimensions.
  @param xs,ys,zs  x/y/z size in mm
*/
void coLabel::setSize(float xs, float ys, float zs)
{
    myWidth = xs;
    myHeight = ys;
    myDepth = zs;

    resizeGeometry();
}

/** Set label string.
  @param s zero terminated label string
*/
void coLabel::setString(const std::string &s)
{

    createGeometry();

    if (s != "")
    {
        labelString = s;
        string nodeName = (string("coLabel(") + s) + ")";
        getDCS()->setName(nodeName);
    }
    else
    {
        labelString = "";
    }

    resize();
}

/** Set label font size.
  @param s label text font size [mm]
*/
void coLabel::setFontSize(float s)
{
    fontSize = s;
    resize();
}

/** Set label position.
  @param x,y,z  label position in 3D space
*/
void coLabel::setPos(float x, float y, float z)
{
    //resize(); - infinite loop
    myX = x;
    myY = y;
    myZ = z;
    getDCS()->setTranslation(myX, myY, myZ);
}

void coLabel::setHighlighted(bool hl)
{
    createGeometry();
    uiElementProvider->setHighlighted(hl);
}

const char *coLabel::getClassName() const
{
    return "coLabel";
}

bool coLabel::isOfClassName(const char *classname) const
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

void coLabel::resizeGeometry()
{
    createGeometry();
    uiElementProvider->resizeGeometry();
}

/// Resize label to string size.
void coLabel::resize()
{

    createGeometry();

    uiElementProvider->update();

    bool changed = false;
    if (myWidth != uiElementProvider->getWidth())
    {
        myWidth = uiElementProvider->getWidth();
        changed = true;
    }

    if (myHeight != uiElementProvider->getHeight())
    {
        myHeight = uiElementProvider->getHeight();
        changed = true;
    }

    if (myDepth != uiElementProvider->getDepth())
    {
        myDepth = uiElementProvider->getDepth();
        changed = true;
    }

    if (changed)
    {
        if (getParent())
        {
            getParent()->childResized();
        }
    }
}
}
