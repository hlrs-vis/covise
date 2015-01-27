/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/sginterface/vruiUIElementProvider.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

namespace vrui
{

/** Constructor
  @param backgroundMaterial normal color
  @param highlightMaterial highlighted color
  @param disableMaterial disabled color
  @see coUIElement for color definition
*/
coColoredBackground::coColoredBackground(coUIElement::Material backgroundMaterial,
                                         coUIElement::Material highlightMaterial,
                                         coUIElement::Material disableMaterial)
{

    this->backgroundMaterial = backgroundMaterial;
    this->highlightMaterial = highlightMaterial;
    this->disableMaterial = disableMaterial;
}

/** Destructor
 */
coColoredBackground::~coColoredBackground()
{
}

/** Set activation state of this background and all its children.
  if this background is disabled, the color is always the
  disabled color, regardless of the highlighted state
  @param en true = elements enabled
*/
void coColoredBackground::setEnabled(bool en)
{
    createGeometry();
    coUIContainer::setEnabled(en);
    uiElementProvider->setEnabled(en);
}

void coColoredBackground::setHighlighted(bool hl)
{
    createGeometry();
    coUIContainer::setHighlighted(hl);
    uiElementProvider->setHighlighted(hl);
}

coUIElement::Material coColoredBackground::getBackgroundMaterial() const
{
    return backgroundMaterial;
}

coUIElement::Material coColoredBackground::getHighlightMaterial() const
{
    return highlightMaterial;
}

coUIElement::Material coColoredBackground::getDisableMaterial() const
{
    return disableMaterial;
}

const char *coColoredBackground::getClassName() const
{
    return "coColoredBackground";
}

bool coColoredBackground::isOfClassName(const char *classname) const
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
            return coBackground::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
