/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_FLAT_PANEL_GEOMETRY_H
#define CO_FLAT_PANEL_GEOMETRY_H

#include <OpenVRUI/coPanelGeometry.h>
#include <OpenVRUI/coUIElement.h>

namespace vrui
{

class OPENVRUIEXPORT coFlatPanelGeometry : public coPanelGeometry
{
public:
    coFlatPanelGeometry(coUIElement::Material backgroundMaterial);
    virtual ~coFlatPanelGeometry();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    coUIElement::Material getBackgroundMaterial() const;

private:
    coUIElement::Material backgroundMaterial;
};
}
#endif
