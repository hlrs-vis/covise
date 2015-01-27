/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_NULL_H
#define SG_VRUI_NULL_H

#include <util/coTypes.h>

#include <OpenVRUI/opensg/SGVruiUIElement.h>

class SGVRUIEXPORT SGVruiNull : public SGVruiUIElement
{

public:
    SGVruiNull(coUIElement *element);
    virtual ~SGVruiNull();

    void createGeometry();

    virtual float getWidth() const
    {
        return 200.0f;
    }

    virtual float getHeight() const
    {
        return 200.0f;
    }

    virtual float getDepth() const
    {
        return 0.0f;
    }

protected:
    virtual void resizeGeometry();
};
#endif
