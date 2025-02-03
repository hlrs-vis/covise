/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiHit.h>
#include <OpenVRUI/vsg/VSGVruiNode.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <vsg/maths/vec3.h>

using namespace vsg;

namespace vrui
{

VSGVruiHit::VSGVruiHit(const LineSegmentIntersector::Intersection &hit, bool mouseHit)
{
    this->hit = hit;
    this->mouseHit = mouseHit;
    isecLocalPoint = 0;
    isecWorldPoint = 0;
    isecNormal = 0;
    node = 0;
}

VSGVruiHit::~VSGVruiHit()
{
    delete isecLocalPoint;
    delete isecWorldPoint;
    delete isecNormal;
    delete node;
}

coVector &VSGVruiHit::getLocalIntersectionPoint() const
{
    if (!isecLocalPoint)
    {
        isecLocalPoint = new coVector(3);
        dvec3 tmp = hit.localIntersection;
        for (int ctr = 0; ctr < 3; ++ctr)
            isecLocalPoint->vec[ctr] = tmp[ctr];
    }

    return *isecLocalPoint;
}

coVector &VSGVruiHit::getWorldIntersectionPoint() const
{
    if (!isecWorldPoint)
    {
        isecWorldPoint = new coVector(3);
        dvec3 tmp = hit.worldIntersection;
        for (int ctr = 0; ctr < 3; ++ctr)
            isecWorldPoint->vec[ctr] = tmp[ctr];
    }

    return *isecWorldPoint;
}

coVector &VSGVruiHit::getWorldIntersectionNormal() const
{
    if (!isecNormal)
    {
        isecNormal = new coVector(3);
        dvec3 tmp = hit.worldIntersection;
        for (int ctr = 0; ctr < 3; ++ctr)
            isecNormal->vec[ctr] = tmp[ctr];
    }

    return *isecNormal;
}

vruiNode *VSGVruiHit::getNode()
{
    if (!node)
    {
        ;
        if (hit.nodePath.size() > 1)
        {
            const vsg::Node *geode = hit.nodePath[hit.nodePath.size()-1];
            node = new VSGVruiNode(vsg::ref_ptr<vsg::Node>((vsg::Node * )geode));
        }
    }
    return node;
}

const vsg::LineSegmentIntersector::Intersection &VSGVruiHit::getHit() const
{
    return hit;
}

bool VSGVruiHit::isMouseHit() const
{
    return mouseHit;
}

}
