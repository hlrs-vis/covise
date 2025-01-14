/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVRUI/sginterface/vruiHit.h>

#include <vsg/utils/LineSegmentIntersector.h>

namespace vrui
{

class vruiNode;

class VSGVRUIEXPORT VSGVruiHit : public vruiHit
{

public:
    VSGVruiHit(const vsg::LineSegmentIntersector::Intersection &isect, bool mouseHit);
    virtual ~VSGVruiHit();

    virtual coVector &getLocalIntersectionPoint() const;
    virtual coVector &getWorldIntersectionPoint() const;
    virtual coVector &getWorldIntersectionNormal() const;
    virtual bool isMouseHit() const;

    vruiNode *getNode();

    const vsg::LineSegmentIntersector::Intersection &getHit() const;

private:
    vsg::LineSegmentIntersector::Intersection hit;
    bool mouseHit;

    mutable coVector *isecLocalPoint;
    mutable coVector *isecWorldPoint;
    mutable coVector *isecNormal;
    mutable vruiNode *node;
};
}
