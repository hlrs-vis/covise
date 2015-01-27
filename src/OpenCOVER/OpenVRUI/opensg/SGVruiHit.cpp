/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiHit.h>
#include <OpenVRUI/opensg/SGVruiNode.h>

#include <OpenSG/OSGVector.h>

#include <OpenVRUI/util/vruiLog.h>

OSG_USING_NAMESPACE

SGVruiHit::SGVruiHit(IntersectAction *hit)
{
    this->hit = hit;
    isecLocalPoint = 0;
    isecWorldPoint = 0;
    isecNormal = 0;
    node = 0;
}

SGVruiHit::~SGVruiHit()
{
    delete isecLocalPoint;
    delete isecWorldPoint;
    delete isecNormal;
    delete node;
}

coVector &SGVruiHit::getLocalIntersectionPoint() const
{

    if (!isecLocalPoint)
    {
        isecLocalPoint = new coVector(3);
        Pnt3f tmp = hit->getHitPoint();
        Matrix m = hit->getHitObject()->getToWorld();
        m.invert();
        m.multPntMatrix(tmp);
        for (int ctr = 0; ctr < 3; ++ctr)
            isecLocalPoint->vec[ctr] = tmp[ctr];
        VRUILOG("SGVruiHit::getLocalIntersectionPoint fixme: Check correctness of result");
    }

    return *isecLocalPoint;
}

coVector &SGVruiHit::getWorldIntersectionPoint() const
{

    if (!isecWorldPoint)
    {
        isecWorldPoint = new coVector(3);
        Pnt3f tmp = hit->getHitPoint();
        for (int ctr = 0; ctr < 3; ++ctr)
            isecWorldPoint->vec[ctr] = tmp[ctr];
    }

    return *isecWorldPoint;
}

coVector &SGVruiHit::getIntersectionNormal() const
{

    if (!isecNormal)
    {
        isecNormal = new coVector(3);
        Pnt3f tmp = hit->getHitNormal();
        for (int ctr = 0; ctr < 3; ++ctr)
            isecNormal->vec[ctr] = tmp[ctr];
    }

    return *isecNormal;
}

vruiNode *SGVruiHit::getNode()
{

    if (!node)
    {
        node = new SGVruiNode(hit->getHitObject());
    }

    return node;
}

const IntersectAction *SGVruiHit::getHit() const
{
    return hit;
}
