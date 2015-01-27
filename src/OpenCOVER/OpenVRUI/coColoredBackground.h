/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COLORED_BACKGROUND_H
#define CO_COLORED_BACKGROUND_H

#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/coBackground.h>

/** This class provides background for GUI elements.
  The color of this background changes according to the elements state
  (normal/highlighted/disabled)
  A background should contain only one child, use another container to layout
  multiple chlidren inside the frame.
*/
namespace vrui
{

class OPENVRUIEXPORT coColoredBackground : public coBackground
{
public:
    coColoredBackground(coUIElement::Material backgroundMaterial,
                        coUIElement::Material highlightMaterial,
                        coUIElement::Material disableMaterial);
    virtual ~coColoredBackground();

    virtual void setEnabled(bool enabled);
    virtual void setHighlighted(bool highlighted);

    coUIElement::Material getBackgroundMaterial() const;
    coUIElement::Material getHighlightMaterial() const;
    coUIElement::Material getDisableMaterial() const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

private:
    coUIElement::Material backgroundMaterial;
    coUIElement::Material highlightMaterial;
    coUIElement::Material disableMaterial;
};
}
#endif
