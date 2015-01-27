/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TEXT_BUTTON_GEOMETRY_H
#define CO_TEXT_BUTTON_GEOMETRY_H

#include "coButtonGeometry.h"

#include <string>

namespace vrui
{

class OPENVRUIEXPORT coTextButtonGeometry : public coButtonGeometry
{
public:
    coTextButtonGeometry(float width, float height, const std::string &name);
    virtual ~coTextButtonGeometry();

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

    void setColors(float r1, float g1, float b1, float a1, float r2, float g2, float b2, float a2);

    float c1r, c1g, c1b, c1a, c2r, c2g, c2b, c2a;

private:
    float width;
    float height;
};
}
#endif
