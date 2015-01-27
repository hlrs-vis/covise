/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_HIT_H
#define OSG_VRUI_HIT_H

#include <OpenVRUI/sginterface/vruiHit.h>

#include <osgUtil/IntersectVisitor>

namespace vrui
{

class vruiNode;

class OSGVRUIEXPORT OSGVruiHit : public vruiHit
{

public:
    OSGVruiHit(osgUtil::Hit &hit, bool mouseHit);
    virtual ~OSGVruiHit();

    virtual coVector &getLocalIntersectionPoint() const;
    virtual coVector &getWorldIntersectionPoint() const;
    virtual coVector &getIntersectionNormal() const;
    virtual bool isMouseHit() const;

    vruiNode *getNode();

    const osgUtil::Hit &getHit() const;

private:
    osgUtil::Hit hit;
    bool mouseHit;

    mutable coVector *isecLocalPoint;
    mutable coVector *isecWorldPoint;
    mutable coVector *isecNormal;
    mutable vruiNode *node;
};
}
#endif
