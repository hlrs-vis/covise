/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coTypes.h>
#include <vsg/maths/vec2.h>

namespace vive
{
class VVCORE_EXPORT vvTouchTableInterface
{
public:
    vvTouchTableInterface(){};
    virtual ~vvTouchTableInterface(){};
    virtual bool isPlanar()
    {
        return false;
    };
    virtual int getMarker(std::string /*name*/)
    {
        return -1;
    };
    virtual bool isVisible(int)
    {
        return false;
    };
    virtual vsg::vec2 getPosition(int)
    {
        return vsg::vec2(0, 0);
    };
    virtual float getOrientation(int)
    {
        return 0;
    };
};

class VVCORE_EXPORT vvTouchTable
{
private:
    static vvTouchTable *tt;

public:
    bool running;
    vvTouchTable();
    virtual ~vvTouchTable();
    static vvTouchTable *instance();

    vvTouchTableInterface *ttInterface;
    void update();
    void config();
};
}
