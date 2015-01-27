/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_RECT_BUTTON_GEOMETRY_H
#define CO_RECT_BUTTON_GEOMETRY_H

#include "coButtonGeometry.h"

#include <string>

namespace vrui
{

class OPENVRUIEXPORT coRectButtonGeometry : public coButtonGeometry
{
public:
    coRectButtonGeometry(float width, float height, const std::string &name);
    virtual ~coRectButtonGeometry();

    float getInnerWidth() const
    {
        return width;
    }
    float getInnerHeight() const
    {
        return height;
    }

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

private:
    float width;
    float height;
};
}
#endif
