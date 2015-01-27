/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_HIT_H
#define SG_VRUI_HIT_H

#include <OpenVRUI/sginterface/vruiHit.h>

#include <OpenSG/OSGIntersectAction.h>

class vruiNode;

class SGVRUIEXPORT SGVruiHit : public vruiHit
{

public:
    SGVruiHit(osg::IntersectAction *hit);
    virtual ~SGVruiHit();

    virtual coVector &getLocalIntersectionPoint() const;
    virtual coVector &getWorldIntersectionPoint() const;
    virtual coVector &getIntersectionNormal() const;

    vruiNode *getNode();

    const osg::IntersectAction *getHit() const;

private:
    osg::IntersectAction *hit;

    mutable coVector *isecLocalPoint;
    mutable coVector *isecWorldPoint;
    mutable coVector *isecNormal;
    mutable vruiNode *node;
};
#endif
