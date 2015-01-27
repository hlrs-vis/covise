/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TOGGLE_BUTTON_GEOMETRY_H
#define CO_TOGGLE_BUTTON_GEOMETRY_H

#include <OpenVRUI/coButtonGeometry.h>

#include <string>

namespace vrui
{

class OPENVRUIEXPORT coToggleButtonGeometry : public coButtonGeometry
{
public:
    coToggleButtonGeometry(const std::string &name = "UI/haken");
    virtual ~coToggleButtonGeometry();
    virtual float getWidth() const;
    virtual float getHeight() const;

    virtual void createGeometry();
    virtual void resizeGeometry();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
};
}
#endif
