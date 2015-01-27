/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TOUCH_INTERSECTION_H
#define CO_TOUCH_INTERSECTION_H

#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <osg/Matrix>
namespace opencover
{
class COVEREXPORT coTouchIntersection : public virtual covise::vruiIntersection
{
public:
    // internal methods (do not call)
    coTouchIntersection();
    ~coTouchIntersection();

    virtual const char *getClassName() const;
    virtual const char *getActionName() const;

    static coTouchIntersection *instance();

private:
    void intersect(); // do the intersection
    void intersect(const osg::Matrix &mat, bool mouseHit); // do the intersection

    float handSize;

    static int myFrameIndex;
    static int myFrame;

    static coTouchIntersection *touchIntersector;
};
}
#endif
