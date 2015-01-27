/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TouchTable_H
#define TouchTable_H

#include <osg/Matrix>
#include <osg/Drawable>
#include <osg/Vec2>

namespace opencover
{
class COVEREXPORT coVRTouchTableInterface
{
public:
    coVRTouchTableInterface(){};
    virtual ~coVRTouchTableInterface(){};
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
    virtual osg::Vec2 getPosition(int)
    {
        return osg::Vec2(0, 0);
    };
    virtual float getOrientation(int)
    {
        return 0;
    };
};

class COVEREXPORT coVRTouchTable
{
private:
    static coVRTouchTable *tt;

public:
    bool running;
    coVRTouchTable();
    virtual ~coVRTouchTable();
    static coVRTouchTable *instance();

    coVRTouchTableInterface *ttInterface;
    void update();
    void config();
};
}
#endif
