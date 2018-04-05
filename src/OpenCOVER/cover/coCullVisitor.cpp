/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2003 Robert Osfield 
 *
 * This library is open source and may be redistributed and/or modified under  
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or 
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * OpenSceneGraph Public License for more details.
*/

#ifdef WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#endif
#include <osg/Version>
#include <osg/Transform>
#include <osg/Projection>
#include <osg/Geode>
#include <osg/LOD>
#include <osg/Billboard>
#include <osg/LightSource>
#include <osg/ClipNode>
//#include <osg/Callback>
#include <osg/TexGenNode>
#include <osg/OccluderNode>
#include <osg/OcclusionQueryNode>
#include <osg/Notify>
#include <osg/Version>
#include <osg/TexEnv>
#include <osg/AlphaFunc>
#include <osg/LineSegment>
#include <osg/TemplatePrimitiveFunctor>
#include <osg/Geometry>
#include <osg/io_utils>
#include <config/CoviseConfig.h>

#include "coCullVisitor.h"

#include <float.h>
#include <algorithm>

#include <osg/Timer>
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
#else
#define getBoundingBox getBound
#endif

using namespace osg;
using namespace osgUtil;
using namespace opencover;

inline float MAX_F(float a, float b)
{
    return a > b ? a : b;
}
inline int EQUAL_F(float a, float b)
{
    return a == b || fabsf(a - b) <= MAX_F(fabsf(a), fabsf(b)) * 1e-3f;
}

class PrintVisitor : public NodeVisitor
{

public:
    PrintVisitor(std::ostream &out)
        : NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
        , _out(out)
    {
        _indent = 0;
        _step = 4;
    }

    inline void moveIn() { _indent += _step; }
    inline void moveOut() { _indent -= _step; }
    inline void writeIndent()
    {
        for (int i = 0; i < _indent; ++i)
            _out << " ";
    }

    virtual void apply(Node &node)
    {
        moveIn();
        writeIndent();
        _out << node.className() << std::endl;
        traverse(node);
        moveOut();
    }

    virtual void apply(Geode &node) { apply((Node &)node); }
    virtual void apply(Billboard &node) { apply((Geode &)node); }
    virtual void apply(LightSource &node) { apply((Group &)node); }
    virtual void apply(ClipNode &node) { apply((Group &)node); }

    virtual void apply(Group &node) { apply((Node &)node); }
    virtual void apply(Transform &node) { apply((Group &)node); }
    virtual void apply(Projection &node) { apply((Group &)node); }
    virtual void apply(Switch &node) { apply((Group &)node); }
    virtual void apply(LOD &node) { apply((Group &)node); }

protected:
    std::ostream &_out;
    int _indent;
    int _step;
};

coCullVisitor::coCullVisitor()
    : CullVisitor()
{
    float cullSize = covise::coCoviseConfig::getFloat("value", "COVER.SmallFeatureCulling", 2.0f);
    setSmallFeatureCullingPixelSize(cullSize);
}

coCullVisitor::~coCullVisitor()
{
}

float coCullVisitor::getDistanceToEyePoint(const Vec3 &pos, bool withLODScale) const
{
    if (withLODScale)
        return (pos - getEyeLocal()).length() * getLODScale();
    else
        return (pos - getEyeLocal()).length();
}

inline coCullVisitor::value_type distance(const osg::Vec3 &coord, const osg::Matrix &matrix)
{

    //std::cout << "distance("<<coord<<", "<<matrix<<")"<<std::endl;
    osg::Vec3 dist = matrix.preMult(coord);
    return sqrt(dist * dist);
    //return -((coCullVisitor::value_type)coord[0]*(coCullVisitor::value_type)matrix(0,2)+(coCullVisitor::value_type)coord[1]*(coCullVisitor::value_type)matrix(1,2)+(coCullVisitor::value_type)coord[2]*(coCullVisitor::value_type)matrix(2,2)+matrix(3,2));
}

float coCullVisitor::getDistanceFromEyePoint(const osg::Vec3 &pos, bool withLODScale) const
{
    const Matrix &matrix = *_modelviewStack.back();
    float dist = distance(pos, matrix);

    if (withLODScale)
        return dist * getLODScale();
    else
        return dist;
}

template <typename Comparator>
struct ComputeNearFarFunctor
{

    ComputeNearFarFunctor()
        : _planes(0)
    {
    }

    void set(CullVisitor::value_type znear, const osg::Matrix &matrix, const osg::Polytope::PlaneList *planes)
    {
        _znear = znear;
        _matrix = matrix;
        _planes = planes;
    }

    typedef std::pair<float, osg::Vec3> DistancePoint;
    typedef std::vector<DistancePoint> Polygon;

    Comparator _comparator;

    CullVisitor::value_type _znear;
    osg::Matrix _matrix;
    const osg::Polytope::PlaneList *_planes;
    Polygon _polygonOriginal;
    Polygon _polygonNew;

    Polygon _pointCache;

    // Handle Points
    inline void operator()(const osg::Vec3 &v1, bool)
    {
        CullVisitor::value_type n1 = distance(v1, _matrix);

        // check if point is behind znear, if so discard
        if (_comparator.greaterEqual(n1, _znear))
        {
            //OSG_NOTICE<<"Point beyond znear"<<std::endl;
            return;
        }

        if (n1 < 0.0)
        {
            // OSG_NOTICE<<"Point behind eye point"<<std::endl;
            return;
        }

        // If point is outside any of frustum planes, discard.
        osg::Polytope::PlaneList::const_iterator pitr;
        for (pitr = _planes->begin();
             pitr != _planes->end();
             ++pitr)
        {
            const osg::Plane &plane = *pitr;
            float d1 = plane.distance(v1);

            if (d1 < 0.0)
            {
                //OSG_NOTICE<<"Point outside frustum "<<d1<<std::endl;
                return;
            }
            //OSG_NOTICE<<"Point ok w.r.t plane "<<d1<<std::endl;
        }

        _znear = n1;
        //OSG_NOTICE<<"Near plane updated "<<_znear<<std::endl;
    }

    // Handle Lines
    inline void operator()(const osg::Vec3 &v1, const osg::Vec3 &v2, bool)
    {
        CullVisitor::value_type n1 = distance(v1, _matrix);
        CullVisitor::value_type n2 = distance(v2, _matrix);

        // check if line is totally behind znear, if so discard
        if (_comparator.greaterEqual(n1, _znear) && _comparator.greaterEqual(n2, _znear))
        {
            //OSG_NOTICE<<"Line totally beyond znear"<<std::endl;
            return;
        }

        if (n1 < 0.0 && n2 < 0.0)
        {
            // OSG_NOTICE<<"Line totally behind eye point"<<std::endl;
            return;
        }

        // Check each vertex to each frustum plane.
        osg::Polytope::ClippingMask selector_mask = 0x1;
        osg::Polytope::ClippingMask active_mask = 0x0;

        osg::Polytope::PlaneList::const_iterator pitr;
        for (pitr = _planes->begin();
             pitr != _planes->end();
             ++pitr)
        {
            const osg::Plane &plane = *pitr;
            float d1 = plane.distance(v1);
            float d2 = plane.distance(v2);

            unsigned int numOutside = ((d1 < 0.0) ? 1 : 0) + ((d2 < 0.0) ? 1 : 0);
            if (numOutside == 2)
            {
                //OSG_NOTICE<<"Line totally outside frustum "<<d1<<"\t"<<d2<<std::endl;
                return;
            }
            unsigned int numInside = ((d1 >= 0.0) ? 1 : 0) + ((d2 >= 0.0) ? 1 : 0);
            if (numInside < 2)
            {
                active_mask = active_mask | selector_mask;
            }

            //OSG_NOTICE<<"Line ok w.r.t plane "<<d1<<"\t"<<d2<<std::endl;

            selector_mask <<= 1;
        }

        if (active_mask == 0)
        {
            _znear = minimum(_znear, n1);
            _znear = minimum(_znear, n2);
            // OSG_NOTICE<<"Line all inside frustum "<<n1<<"\t"<<n2<<" number of plane="<<_planes->size()<<std::endl;
            return;
        }

        //OSG_NOTICE<<"Using brute force method of line cutting frustum walls"<<std::endl;
        DistancePoint p1(0, v1);
        DistancePoint p2(0, v2);

        selector_mask = 0x1;

        for (pitr = _planes->begin();
             pitr != _planes->end();
             ++pitr)
        {
            if (active_mask & selector_mask)
            {
                // clip line to plane
                const osg::Plane &plane = *pitr;

                // assign the distance from the current plane.
                p1.first = plane.distance(p1.second);
                p2.first = plane.distance(p2.second);

                if (p1.first >= 0.0f)
                {
                    // p1 is in.
                    if (p2.first < 0.0)
                    {
                        // p2 is out.
                        // replace p2 with intersection
                        float r = p1.first / (p1.first - p2.first);
                        p2 = DistancePoint(0.0f, p1.second * (1.0f - r) + p2.second * r);
                    }
                }
                else if (p2.first >= 0.0f)
                {
                    // p1 is out and p2 is in.
                    // replace p1 with intersection
                    float r = p1.first / (p1.first - p2.first);
                    p1 = DistancePoint(0.0f, p1.second * (1.0f - r) + p2.second * r);
                }
                // The case where both are out was handled above.
            }
            selector_mask <<= 1;
        }

        n1 = distance(p1.second, _matrix);
        n2 = distance(p2.second, _matrix);
        _znear = _comparator.minimum(n1, n2);
        //OSG_NOTICE<<"Near plane updated "<<_znear<<std::endl;
    }

    // Handle Triangles
    inline void operator()(const osg::Vec3 &v1, const osg::Vec3 &v2, const osg::Vec3 &v3, bool)
    {
        CullVisitor::value_type n1 = distance(v1, _matrix);
        CullVisitor::value_type n2 = distance(v2, _matrix);
        CullVisitor::value_type n3 = distance(v3, _matrix);

        // check if triangle is total behind znear, if so discard
        if (_comparator.greaterEqual(n1, _znear) && _comparator.greaterEqual(n2, _znear) && _comparator.greaterEqual(n3, _znear))
        {
            //OSG_NOTICE<<"Triangle totally beyond znear"<<std::endl;
            return;
        }

        if (n1 < 0.0 && n2 < 0.0 && n3 < 0.0)
        {
            // OSG_NOTICE<<"Triangle totally behind eye point"<<std::endl;
            return;
        }

        // Check each vertex to each frustum plane.
        osg::Polytope::ClippingMask selector_mask = 0x1;
        osg::Polytope::ClippingMask active_mask = 0x0;

        osg::Polytope::PlaneList::const_iterator pitr;
        for (pitr = _planes->begin();
             pitr != _planes->end();
             ++pitr)
        {
            const osg::Plane &plane = *pitr;
            float d1 = plane.distance(v1);
            float d2 = plane.distance(v2);
            float d3 = plane.distance(v3);

            unsigned int numOutside = ((d1 < 0.0) ? 1 : 0) + ((d2 < 0.0) ? 1 : 0) + ((d3 < 0.0) ? 1 : 0);
            if (numOutside == 3)
            {
                //OSG_NOTICE<<"Triangle totally outside frustum "<<d1<<"\t"<<d2<<"\t"<<d3<<std::endl;
                return;
            }
            unsigned int numInside = ((d1 >= 0.0) ? 1 : 0) + ((d2 >= 0.0) ? 1 : 0) + ((d3 >= 0.0) ? 1 : 0);
            if (numInside < 3)
            {
                active_mask = active_mask | selector_mask;
            }

            //OSG_NOTICE<<"Triangle ok w.r.t plane "<<d1<<"\t"<<d2<<"\t"<<d3<<std::endl;

            selector_mask <<= 1;
        }

        if (active_mask == 0)
        {
            _znear = _comparator.minimum(_znear, n1);
            _znear = _comparator.minimum(_znear, n2);
            _znear = _comparator.minimum(_znear, n3);
            // OSG_NOTICE<<"Triangle all inside frustum "<<n1<<"\t"<<n2<<"\t"<<n3<<" number of plane="<<_planes->size()<<std::endl;
            return;
        }

        //return;

        // numPartiallyInside>0) so we have a triangle cutting an frustum wall,
        // this means that use brute force methods for dividing up triangle.

        //OSG_NOTICE<<"Using brute force method of triangle cutting frustum walls"<<std::endl;
        _polygonOriginal.clear();
        _polygonOriginal.push_back(DistancePoint(0, v1));
        _polygonOriginal.push_back(DistancePoint(0, v2));
        _polygonOriginal.push_back(DistancePoint(0, v3));

        selector_mask = 0x1;

        for (pitr = _planes->begin();
             pitr != _planes->end() && !_polygonOriginal.empty();
             ++pitr)
        {
            if (active_mask & selector_mask)
            {
                // polygon bisects plane so need to divide it up.
                const osg::Plane &plane = *pitr;
                _polygonNew.clear();

                // assign the distance from the current plane.
                for (Polygon::iterator polyItr = _polygonOriginal.begin();
                     polyItr != _polygonOriginal.end();
                     ++polyItr)
                {
                    polyItr->first = plane.distance(polyItr->second);
                }

                // create the new polygon by clamping against the
                unsigned int psize = _polygonOriginal.size();

                for (unsigned int ci = 0; ci < psize; ++ci)
                {
                    unsigned int ni = (ci + 1) % psize;
                    bool computeIntersection = false;
                    if (_polygonOriginal[ci].first >= 0.0f)
                    {
                        _polygonNew.push_back(_polygonOriginal[ci]);

                        if (_polygonOriginal[ni].first < 0.0f)
                            computeIntersection = true;
                    }
                    else if (_polygonOriginal[ni].first > 0.0f)
                        computeIntersection = true;

                    if (computeIntersection)
                    {
                        // segment intersects with the plane, compute new position.
                        float r = _polygonOriginal[ci].first / (_polygonOriginal[ci].first - _polygonOriginal[ni].first);
                        _polygonNew.push_back(DistancePoint(0.0f, _polygonOriginal[ci].second * (1.0f - r) + _polygonOriginal[ni].second * r));
                    }
                }
                _polygonOriginal.swap(_polygonNew);
            }
            selector_mask <<= 1;
        }

        // now take the nearst points to the eye point.
        for (Polygon::iterator polyItr = _polygonOriginal.begin();
             polyItr != _polygonOriginal.end();
             ++polyItr)
        {
            CullVisitor::value_type dist = distance(polyItr->second, _matrix);
            if (_comparator.less(dist, _znear))
            {
                _znear = dist;
                //OSG_NOTICE<<"Near plane updated "<<_znear<<std::endl;
            }
        }
    }

    // Handle Quadrilaterals
    inline void operator()(const osg::Vec3 &v1, const osg::Vec3 &v2, const osg::Vec3 &v3, const osg::Vec3 &v4, bool treatVertexDataAsTemporary)
    {
        this->operator()(v1, v2, v3, treatVertexDataAsTemporary);
        this->operator()(v1, v3, v4, treatVertexDataAsTemporary);
    }
};

struct LessComparator
{
    inline bool less(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs < rhs; }
    inline bool lessEqual(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs <= rhs; }
    inline bool greaterEqual(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs >= rhs; }
    inline CullVisitor::value_type minimum(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs < rhs ? lhs : rhs; }
};
typedef ComputeNearFarFunctor<LessComparator> ComputeNearestPointFunctor;

struct GreaterComparator
{
    inline bool less(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs > rhs; }
    inline bool lessEqual(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs >= rhs; }
    inline bool greaterEqual(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs <= rhs; }
    inline CullVisitor::value_type minimum(CullVisitor::value_type lhs, CullVisitor::value_type rhs) const { return lhs > rhs ? lhs : rhs; }
};
typedef ComputeNearFarFunctor<GreaterComparator> ComputeFurthestPointFunctor;

CullVisitor::value_type coCullVisitor::computeNearestPointInFrustum(const osg::Matrix &matrix, const osg::Polytope::PlaneList &planes, const osg::Drawable &drawable)
{
    // OSG_NOTICE<<"coCullVisitor::computeNearestPointInFrustum("<<getTraversalNumber()<<"\t"<<planes.size()<<std::endl;

    osg::TemplatePrimitiveFunctor<ComputeNearestPointFunctor> cnpf;
    cnpf.set(FLT_MAX, matrix, &planes);

    drawable.accept(cnpf);

    return cnpf._znear;
}

CullVisitor::value_type coCullVisitor::computeFurthestPointInFrustum(const osg::Matrix &matrix, const osg::Polytope::PlaneList &planes, const osg::Drawable &drawable)
{
    //OSG_NOTICE<<"CullVisitor::computeFurthestPointInFrustum("<<getTraversalNumber()<<"\t"<<planes.size()<<")"<<std::endl;

    osg::TemplatePrimitiveFunctor<ComputeFurthestPointFunctor> cnpf;
    cnpf.set(-FLT_MAX, matrix, &planes);

    drawable.accept(cnpf);

    return cnpf._znear;
}

bool coCullVisitor::updateCalculatedNearFar(const osg::Matrix &matrix, const osg::BoundingBox &bb)
{
    // efficient computation of near and far, only taking into account the nearest and furthest
    // corners of the bounding box.
    value_type d_near = distance(bb.corner(_bbCornerNear), matrix);
    value_type d_far = distance(bb.corner(_bbCornerFar), matrix);

    if (d_near > d_far)
    {
        std::swap(d_near, d_far);
        if (!EQUAL_F(d_near, d_far))
        {
            osg::notify(osg::WARN) << "Warning: coCullVisitor::updateCalculatedNearFar(.) near>far in range calculation," << std::endl;
            osg::notify(osg::WARN) << "         correcting by swapping values d_near=" << d_near << " dfar=" << d_far << std::endl;
        }
    }

    if (d_far < 0.0)
    {
        // whole object behind the eye point so disguard
        return false;
    }

    if (d_near < _computed_znear)
        _computed_znear = d_near;
    if (d_far > _computed_zfar)
        _computed_zfar = d_far;

    return true;
}

bool coCullVisitor::updateCalculatedNearFar(const osg::Matrix &matrix, const osg::Drawable &drawable, bool isBillboard)
{
    const osg::BoundingBox &bb = drawable.getBoundingBox();

    value_type d_near, d_far;

    if (isBillboard)
    {

#ifdef TIME_BILLBOARD_NEAR_FAR_CALCULATION
        static unsigned int lastFrameNumber = getTraversalNumber();
        static unsigned int numBillboards = 0;
        static double elapsed_time = 0.0;
        if (lastFrameNumber != getTraversalNumber())
        {
            OSG_NOTICE << "Took " << elapsed_time << "ms to test " << numBillboards << " billboards" << std::endl;
            numBillboards = 0;
            elapsed_time = 0.0;
            lastFrameNumber = getTraversalNumber();
        }
        osg::Timer_t start_t = osg::Timer::instance()->tick();
#endif

        osg::Vec3 lookVector(-matrix(0, 2), -matrix(1, 2), -matrix(2, 2));

        unsigned int bbCornerFar = (lookVector.x() >= 0 ? 1 : 0) + (lookVector.y() >= 0 ? 2 : 0) + (lookVector.z() >= 0 ? 4 : 0);

        unsigned int bbCornerNear = (~bbCornerFar) & 7;

        d_near = distance(bb.corner(bbCornerNear), matrix);
        d_far = distance(bb.corner(bbCornerFar), matrix);

        OSG_NOTICE.precision(15);

        if (false)
        {

            OSG_NOTICE << "TESTING Billboard near/far computation" << std::endl;

            // OSG_WARN<<"Checking corners of billboard "<<std::endl;
            // deprecated brute force way, use all corners of the bounding box.
            value_type nd_near, nd_far;
            nd_near = nd_far = distance(bb.corner(0), matrix);
            for (unsigned int i = 0; i < 8; ++i)
            {
                value_type d = distance(bb.corner(i), matrix);
                if (d < nd_near)
                    nd_near = d;
                if (d > nd_far)
                    nd_far = d;
                OSG_NOTICE << "\ti=" << i << "\td=" << d << std::endl;
            }

            if (nd_near == d_near && nd_far == d_far)
            {
                OSG_NOTICE << "\tBillboard near/far computation correct " << std::endl;
            }
            else
            {
                OSG_NOTICE << "\tBillboard near/far computation ERROR\n\t\t" << d_near << "\t" << nd_near
                           << "\n\t\t" << d_far << "\t" << nd_far << std::endl;
            }
        }

#ifdef TIME_BILLBOARD_NEAR_FAR_CALCULATION
        osg::Timer_t end_t = osg::Timer::instance()->tick();

        elapsed_time += osg::Timer::instance()->delta_m(start_t, end_t);
        ++numBillboards;
#endif
    }
    else
    {
        // efficient computation of near and far, only taking into account the nearest and furthest
        // corners of the bounding box.
        d_near = distance(bb.corner(_bbCornerNear), matrix);
        d_far = distance(bb.corner(_bbCornerFar), matrix);
    }

    if (d_near > d_far)
    {
        std::swap(d_near, d_far);
        if (!EQUAL_F(d_near, d_far))
        {
            OSG_WARN << "Warning: CullVisitor::updateCalculatedNearFar(.) near>far in range calculation," << std::endl;
            OSG_WARN << "         correcting by swapping values d_near=" << d_near << " dfar=" << d_far << std::endl;
        }
    }

    if (d_far < 0.0)
    {
        // whole object behind the eye point so discard
        return false;
    }

    if (_computeNearFar == COMPUTE_NEAR_FAR_USING_PRIMITIVES || _computeNearFar == COMPUTE_NEAR_USING_PRIMITIVES)
    {
        if (d_near < _computed_znear || d_far > _computed_zfar)
        {
            osg::Polytope &frustum = getCurrentCullingSet().getFrustum();
            if (frustum.getResultMask())
            {
                MatrixPlanesDrawables mpd;
                if (isBillboard)
                {
                    // OSG_WARN<<"Adding billboard into deffered list"<<std::endl;
                    osg::Polytope transformed_frustum;
                    transformed_frustum.setAndTransformProvidingInverse(getProjectionCullingStack().back().getFrustum(), matrix);
                    mpd.set(matrix, &drawable, transformed_frustum);
                }
                else
                {
                    mpd.set(matrix, &drawable, frustum);
                }

                if (d_near < _computed_znear)
                {
                    _nearPlaneCandidateMap.insert(DistanceMatrixDrawableMap::value_type(d_near, mpd));
                }

                if (_computeNearFar == COMPUTE_NEAR_FAR_USING_PRIMITIVES)
                {
                    if (d_far > _computed_zfar)
                    {
                        _farPlaneCandidateMap.insert(DistanceMatrixDrawableMap::value_type(d_far, mpd));
                    }
                }

                // use the far point if its nearer than current znear as this is a conservative estimate of the znear
                // while the final computation for this drawable is deferred.
                if (d_far >= 0.0 && d_far < _computed_znear)
                {
                    //_computed_znear = d_far;
                }

                if (_computeNearFar == COMPUTE_NEAR_FAR_USING_PRIMITIVES)
                {
                    // use the near point if its further than current zfar as this is a conservative estimate of the zfar
                    // while the final computation for this drawable is deferred.
                    if (d_near >= 0.0 && d_near > _computed_zfar)
                    {
                        // _computed_zfar = d_near;
                    }
                }
                else // computing zfar using bounding sphere
                {
                    if (d_far > _computed_zfar)
                        _computed_zfar = d_far;
                }
            }
            else
            {
                if (d_near < _computed_znear)
                    _computed_znear = d_near;
                if (d_far > _computed_zfar)
                    _computed_zfar = d_far;
            }
        }
    }
    else
    {
        if (d_near < _computed_znear)
            _computed_znear = d_near;
        if (d_far > _computed_zfar)
            _computed_zfar = d_far;
    }

    /*
    // deprecated brute force way, use all corners of the bounding box.
    updateCalculatedNearFar(bb.corner(0));
    updateCalculatedNearFar(bb.corner(1));
    updateCalculatedNearFar(bb.corner(2));
    updateCalculatedNearFar(bb.corner(3));
    updateCalculatedNearFar(bb.corner(4));
    updateCalculatedNearFar(bb.corner(5));
    updateCalculatedNearFar(bb.corner(6));
    updateCalculatedNearFar(bb.corner(7));
*/

    return true;
}

void coCullVisitor::updateCalculatedNearFar(const osg::Vec3 &pos)
{
    float d;
    if (!_modelviewStack.empty())
    {
        const osg::Matrix &matrix = *(_modelviewStack.back());
        d = distance(pos, matrix);
    }
    else
    {
        d = -pos.z();
    }

    if (d < _computed_znear)
    {
        _computed_znear = d;
        if (d < 0.0)
            osg::notify(osg::WARN) << "Alerting billboard =" << d << std::endl;
    }
    if (d > _computed_zfar)
        _computed_zfar = d;
}

/* we do need our own version, it is a 1:1 copy of the original but uses another inline distance computation, see above, otherwise depth sorting does not work */

void coCullVisitor::apply(osg::Drawable &drawable)
{
    RefMatrix& matrix = *getModelViewMatrix();

    const BoundingBox &bb =drawable.getBoundingBox();

    if( drawable.getCullCallback() )
    {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 5, 2)
        osg::DrawableCullCallback* dcb = drawable.getCullCallback()->asDrawableCullCallback();
#else
        osg::Drawable::CullCallback* dcb = dynamic_cast<osg::Drawable::CullCallback *>(drawable.getCullCallback());
#endif
        if (dcb)
        {
            if( dcb->cull( this, &drawable, &_renderInfo ) == true ) return;
        }
#if OSG_VERSION_GREATER_OR_EQUAL(3, 5, 2)
        else
        {
            drawable.getCullCallback()->run(&drawable,this);
        }
#endif
    }

#if OSG_VERSION_GREATER_OR_EQUAL(3, 5, 2)
    if (drawable.isCullingActive() && isCulled(bb))
        return;
#else
    if (!getNodePath().empty() && getNodePath().back()->isCullingActive() && isCulled(bb))
        return;
#endif


    if (_computeNearFar && bb.valid())
    {
        if (!updateCalculatedNearFar(matrix,drawable,false)) return;
    }

    // need to track how push/pops there are, so we can unravel the stack correctly.
    unsigned int numPopStateSetRequired = 0;

    // push the geoset's state on the geostate stack.
    StateSet* stateset = drawable.getStateSet();
    if (stateset)
    {
        ++numPopStateSetRequired;
        pushStateSet(stateset);
    }

    CullingSet& cs = getCurrentCullingSet();
    if (!cs.getStateFrustumList().empty())
    {
        osg::CullingSet::StateFrustumList& sfl = cs.getStateFrustumList();
        for(osg::CullingSet::StateFrustumList::iterator itr = sfl.begin();
                itr != sfl.end();
                ++itr)
        {
            if (itr->second.contains(bb))
            {
                ++numPopStateSetRequired;
                pushStateSet(itr->first.get());
            }
        }
    }

    float depth = bb.valid() ? distance(bb.center(),matrix) : 0.0f;

    if (osg::isNaN(depth))
    {
        OSG_NOTICE<<"CullVisitor::apply(Geode&) detected NaN,"<<std::endl
            <<"    depth="<<depth<<", center=("<<bb.center()<<"),"<<std::endl
            <<"    matrix="<<matrix<<std::endl;
        OSG_DEBUG << "    NodePath:" << std::endl;
        for (NodePath::const_iterator i = getNodePath().begin(); i != getNodePath().end(); ++i)
        {
            OSG_DEBUG << "        \"" << (*i)->getName() << "\"" << std::endl;
        }
    }
    else
    {
        addDrawableAndDepth(&drawable,&matrix,depth);
    }

    for(unsigned int i=0;i< numPopStateSetRequired; ++i)
    {
        popStateSet();
    }
}

void coCullVisitor::apply(Billboard &node)
{
    if (isCulled(node)) return;

    // push the node's state.
    StateSet* node_state = node.getStateSet();
    if (node_state) pushStateSet(node_state);

    // Don't traverse billboard, since drawables are handled manually below
    //handle_cull_callbacks_and_traverse(node);

    const Vec3& eye_local = getEyeLocal();
    const RefMatrix& modelview = *getModelViewMatrix();

    for(unsigned int i=0;i<node.getNumDrawables();++i)
    {
        const Vec3& pos = node.getPosition(i);

        Drawable* drawable = node.getDrawable(i);
        // need to modify isCulled to handle the billboard offset.
        // if (isCulled(drawable->getBound())) continue;

        if( drawable->getCullCallback() )
        {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
            osg::DrawableCullCallback* dcb = drawable->getCullCallback()->asDrawableCullCallback();
#else
            osg::Drawable::CullCallback* dcb = dynamic_cast<osg::Drawable::CullCallback *>(drawable->getCullCallback());
#endif
            if (dcb && dcb->cull( this, drawable, &_renderInfo ) == true )
                continue;
        }

        RefMatrix* billboard_matrix = createOrReuseMatrix(modelview);

        node.computeMatrix(*billboard_matrix,eye_local,pos);


        if (_computeNearFar && drawable->getBoundingBox().valid()) updateCalculatedNearFar(*billboard_matrix,*drawable,true);
        float depth = distance(pos,modelview);
/*
        if (_computeNearFar)
        {
            if (d<_computed_znear)
            {
                if (d<0.0) OSG_WARN<<"Alerting billboard handling ="<<d<< std::endl;
                _computed_znear = d;
            }
            if (d>_computed_zfar) _computed_zfar = d;
        }
*/
        StateSet* stateset = drawable->getStateSet();
        if (stateset) pushStateSet(stateset);

        if (osg::isNaN(depth))
        {
            OSG_NOTICE<<"CullVisitor::apply(Billboard&) detected NaN,"<<std::endl
                                    <<"    depth="<<depth<<", pos=("<<pos<<"),"<<std::endl
                                    <<"    *billboard_matrix="<<*billboard_matrix<<std::endl;
            OSG_DEBUG << "    NodePath:" << std::endl;
            for (NodePath::const_iterator itr = getNodePath().begin(); itr != getNodePath().end(); ++itr)
            {
                OSG_DEBUG << "        \"" << (*itr)->getName() << "\"" << std::endl;
            }
        }
        else
        {
            addDrawableAndDepth(drawable,billboard_matrix,depth);
        }

        if (stateset) popStateSet();

    }

    // pop the node's state off the geostate stack.
    if (node_state) popStateSet();
}
