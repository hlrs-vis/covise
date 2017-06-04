/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coIntersectionUtil.h"

#include "coIntersection.h"
#include <osg/Version>
#include <osg/io_utils>
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
#define getBound getBoundingBox
#endif
#include <iostream>
#include <algorithm>
#include <cmath>
#include <util/unixcompat.h>

using namespace osg;
using namespace osgUtil;

namespace opencover
{
namespace Private
{

    struct TriangleHit
    {
        TriangleHit(unsigned int index, const osg::Vec3 &normal, float r1, const osg::Vec3 &v1, float r2, const osg::Vec3 &v2, float r3, const osg::Vec3 &v3)
            : _index(index)
            , _normal(normal)
            , _r1(r1)
            , _v1(v1)
            , _r2(r2)
            , _v2(v2)
            , _r3(r3)
            , _v3(v3)
        {
        }

        unsigned int _index;
        const osg::Vec3 _normal;
        float _r1;
        const osg::Vec3 _v1;
        float _r2;
        const osg::Vec3 _v2;
        float _r3;
        const osg::Vec3 _v3;

    protected:
        TriangleHit &operator=(const TriangleHit &) { return *this; }
    };

    struct TriangleIntersect
    {
        Vec3 _s;
        Vec3 _d;
        float _length;

        unsigned int _index;
        float _ratio;
        bool _hit;

        typedef std::multimap<float, TriangleHit> TriangleHitList;

        TriangleHitList _thl;

        TriangleIntersect()
        : _length(-1)
        {
        }

        TriangleIntersect(const LineSegment &seg, float ratio = FLT_MAX)
        : _length(-1)
        {
            set(seg, ratio);
        }

        void set(const LineSegment &seg, float ratio = FLT_MAX)
        {

            _hit = false;
            _index = 0;
            _ratio = ratio;

            _s = seg.start();
            _d = seg.end() - seg.start();
            if (std::isnan(_s[0]) || std::isnan(_s[1]) || std::isnan(_s[2]))
            {
                _length = -1;
                std::cerr << "TriangleIntersect: invalid line segment - start" << std::endl;
                return;
            }
            if (std::isnan(_d[0]) || std::isnan(_d[1]) || std::isnan(_d[2]))
            {
                _length = -1;
                std::cerr << "TriangleIntersect: invalid line segment - direction" << std::endl;
                return;
            }
            _length = _d.length();
            _d /= _length;
        }

        //   bool intersect(const Vec3& v1,const Vec3& v2,const Vec3& v3,float& r)
        inline void operator()(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3, bool=false)
        {
            ++_index; //TODO Not really useful in parallel....

            if (_length < 0)
            {
                // invalid line segment
                return;
            }

            const float eps = 1e-9;
#ifdef NDEBUG
#define CHECK(x)
#else
#define CHECK(x) \
            if (std::isnan(x) || std::isinf(x)) { \
                std::cerr << "not finite: " << #x << "=" << x << std::endl; \
                std::cerr << "\tv1=" << v1 << ", v2=" << v2 << ", v3=" << v3 << std::endl; \
                std::cerr << "\t_d=" << _d << ", _s=" << _s << std::endl; \
                return; \
            }
#endif

            Vec3 v12 = v2 - v1;
            if (v12.length() < eps)
                return;
            Vec3 v23 = v3 - v2;
            if (v23.length() < eps)
                return;
            Vec3 v31 = v1 - v3;
            if (v31.length() < eps)
                return;

            Vec3 n12 = v12 ^ _d;
            float ds12 = (_s - v1) * n12;
            CHECK(ds12);
            float d312 = (v3 - v1) * n12;
            CHECK(d312);
            if (d312 >= 0.0f)
            {
                if (ds12 < 0.0f)
                    return;
                if (ds12 > d312)
                    return;
            }
            else // d312 < 0
            {
                if (ds12 > 0.0f)
                    return;
                if (ds12 < d312)
                    return;
            }

            Vec3 n23 = v23 ^ _d;
            float ds23 = (_s - v2) * n23;
            CHECK(ds23);
            float d123 = (v1 - v2) * n23;
            CHECK(d123);
            if (d123 >= 0.0f)
            {
                if (ds23 < 0.0f)
                    return;
                if (ds23 > d123)
                    return;
            }
            else // d123 < 0
            {
                if (ds23 > 0.0f)
                    return;
                if (ds23 < d123)
                    return;
            }

            Vec3 n31 = v31 ^ _d;
            float ds31 = (_s - v3) * n31;
            CHECK(ds31);
            float d231 = (v2 - v3) * n31;
            CHECK(d231);
            if (d231 >= 0.0f)
            {
                if (ds31 < 0.0f)
                    return;
                if (ds31 > d231)
                    return;
            }
            else // d231 < 0
            {
                if (ds31 > 0.0f)
                    return;
                if (ds31 < d231)
                    return;
            }

            float r3;
            if (fabs(ds12) < eps)
                r3 = 0.0f;
            else if (d312 != 0.0f)
                r3 = ds12 / d312;
            else
                return; // the triangle and the line must be parallel intersection.
            CHECK(r3);

            float r1;
            if (fabs(ds23) < eps)
                r1 = 0.0f;
            else if (d123 != 0.0f)
                r1 = ds23 / d123;
            else
                return; // the triangle and the line must be parallel intersection.
            CHECK(r1);

            float r2;
            if (fabs(ds31) < eps)
                r2 = 0.0f;
            else if (d231 != 0.0f)
                r2 = ds31 / d231;
            else
                return; // the triangle and the line must be parallel intersection.
            CHECK(r2);

            float total_r = (r1 + r2 + r3);
            if (fabs(total_r) > eps)
            {
                if (total_r == 0.0f)
                    return; // the triangle and the line must be parallel intersection.
                float inv_total_r = 1.0f / total_r;
                r1 *= inv_total_r;
                r2 *= inv_total_r;
                r3 *= inv_total_r;
            }
            CHECK(total_r);

            Vec3 in = v1 * r1 + v2 * r2 + v3 * r3;
            if (!in.valid())
            {
                std::cerr << "Warning:: Picked up error in TriangleIntersect" << std::endl;
                OSG_WARN<<"   v=("<<v1<<",\t"<<v2<<",\t"<<v3<<")"<<std::endl;
                OSG_WARN<<"   1/r=("<<r1<<",\t"<<r2<<",\t"<<r3<<")"<<std::endl;
                OSG_WARN<<"   total_r=" << total_r << std::endl;
                return;
            }

            float d = (in - _s) * _d;

            if (d < 0.0f)
                return;
            if (d > _length)
                return;

            osg::Vec3 normal = v12 ^ v23;
            normal.normalize();

            float r = d / _length;

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                _thl.insert(std::pair<const float, TriangleHit>(r, TriangleHit(_index - 1, normal, r1, v1, r2, v2, r3, v3)));
                _hit = true;
            }
        }
    };
}
}

void opencover::Private::coIntersectionVisitor::intersect(osg::Geometry *geo, osg::PrimitiveFunctor &functor)
{
    const osg::Array *vertices = geo->getVertexArray();

    if (!vertices && geo->getVertexAttribArrayList().size() > 0)
    {
        //   std::cerr<<"Using vertex attribute instead"<<std::endl; //TODO get Positions
        //  vertices = geo->getVertexAttribArrayList()[0].get();
    }

    if (!vertices || vertices->getNumElements() == 0)
        return;

    if (opencover::coIntersection::isVerboseIntersection())
        std::cerr << "p(" << geo->getPrimitiveSetList().size() << ")";

    switch (vertices->getType())
    {
    case (osg::Array::Vec2ArrayType):
        functor.setVertexArray(vertices->getNumElements(), static_cast<const Vec2 *>(vertices->getDataPointer()));
        break;
    case (osg::Array::Vec3ArrayType):
        functor.setVertexArray(vertices->getNumElements(), static_cast<const Vec3 *>(vertices->getDataPointer()));
        break;
    case (osg::Array::Vec4ArrayType):
        functor.setVertexArray(vertices->getNumElements(), static_cast<const Vec4 *>(vertices->getDataPointer()));
        break;
    case (osg::Array::Vec2dArrayType):
        functor.setVertexArray(vertices->getNumElements(), static_cast<const Vec2d *>(vertices->getDataPointer()));
        break;
    case (osg::Array::Vec3dArrayType):
        functor.setVertexArray(vertices->getNumElements(), static_cast<const Vec3d *>(vertices->getDataPointer()));
        break;
    case (osg::Array::Vec4dArrayType):
        functor.setVertexArray(vertices->getNumElements(), static_cast<const Vec4d *>(vertices->getDataPointer()));
        break;
    default:
        std::cerr << "Warning: Geometry::accept(PrimitiveFunctor&) cannot handle Vertex Array type" << vertices->getType() << std::endl;
        return;
    }

    osg::Geometry::PrimitiveSetList &pList = geo->getPrimitiveSetList();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ssize_t ctr = 0; ctr < (ssize_t)pList.size(); ++ctr)
    {

        if (pList[ctr]->getType() == osg::PrimitiveSet::DrawArrayLengthsPrimitiveType)
        {
            osg::DrawArrayLengths *dal = dynamic_cast<osg::DrawArrayLengths *>(pList[ctr].get());
            GLint first = dal->getFirst();

            if (opencover::coIntersection::isVerboseIntersection())
                std::cerr << "d(" << dal->size() << ")";

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (ssize_t dalIndex = 0; dalIndex < (ssize_t)dal->size(); ++dalIndex)
            {
                //std::cerr << omp_get_thread_num();
                functor.drawArrays(dal->getMode(), first, (*dal)[dalIndex]);
                first += (*dal)[dalIndex];
            }
        }
        else
        {
            pList[ctr]->accept(functor);
        }
    }
}

bool opencover::Private::coIntersectionVisitor::intersect(Drawable &drawable)
{

    //if (std::cerr.bad()) std::cerr.clear();
    //std::cerr << "i";
    bool hitFlag = false;

    IntersectState *cis = _intersectStateStack.back().get();

    const BoundingBox &bb = drawable.getBound();

    for (IntersectState::LineSegmentList::iterator sitr = cis->_segList.begin();
         sitr != cis->_segList.end();
         ++sitr)
    {
        if (sitr->second->intersect(bb))
        {

            ParallelTriangleFunctor<TriangleIntersect> ti;
            ti.set(*sitr->second);
            osg::Geometry *geo = dynamic_cast<osg::Geometry *>(&drawable);
            if (geo)
                intersect(geo, ti);
            else
            {
                //std::cerr << "d";
                drawable.accept(ti);
            }

            if (ti._hit)
            {

                //osg::Geometry* geometry = drawable.asGeometry();

                for (TriangleIntersect::TriangleHitList::iterator thitr = ti._thl.begin();
                     thitr != ti._thl.end();
                     ++thitr)
                {

                    Hit hit;
                    hit._nodePath = _nodePath;
                    hit._matrix = cis->_model_matrix;
                    hit._inverse = cis->_model_inverse;
                    hit._drawable = &drawable;
                    if (_nodePath.empty())
                        hit._geode = NULL;
                    else
                        hit._geode = dynamic_cast<Geode *>(_nodePath.back());

                    TriangleHit &triHit = thitr->second;

                    hit._ratio = thitr->first;
                    hit._primitiveIndex = triHit._index; // TODO Not really.... index is undefined
                    hit._originalLineSegment = sitr->first;
                    hit._localLineSegment = sitr->second;

                    hit._intersectPoint = sitr->second->start() * (1.0f - hit._ratio) + sitr->second->end() * hit._ratio;

                    hit._intersectNormal = triHit._normal;

                    //               if (geometry)
                    //               {
                    //                  osg::Vec3Array* vertices = dynamic_cast<osg::Vec3Array*>(geometry->getVertexArray());
                    //                  if (vertices)
                    //                  {
                    //                     osg::Vec3 first = vertices->front();
                    //                     hit._vecIndexList.push_back(triHit._v1-first);
                    //                     hit._vecIndexList.push_back(triHit._v2-first);
                    //                     hit._vecIndexList.push_back(triHit._v3-first);
                    //                  }
                    //               }

                    _segHitList[sitr->first.get()].push_back(hit);

                    std::sort(_segHitList[sitr->first.get()].begin(), _segHitList[sitr->first.get()].end());

                    hitFlag = true;
                }
            }
        }
    }

    return hitFlag;
}

#if 0
void opencover::Private::coIntersectionVisitor::apply(osg::Geode& geode)
{
   {
   if (enterNode(geode))
   {

      int location = 0;

      {
      for(unsigned int i = 0; i < geode.getNumDrawables(); i++ )
      {
         //std::cerr << "create: " << omp_get_thread_num() << std::endl;
         assert(omp_get_thread_num() == 0);
         this->subVisitors.push_back(coIntersectionSubVisitor(*this));
         location = this->subVisitors.size() - 1;

#ifdef _OPENMP
#pragma omp task shared(geode) firstprivate(location)
#endif
         {

            coIntersectionSubVisitor & visitor = this->subVisitors[location];

            if (std::cerr.bad()) std::cerr.clear();
            std::cerr << omp_get_thread_num() << std::endl;
            visitor.intersect(geode.getDrawable(i));
            if (visitor.hits())
            {
               LineSegmentHitListMap hitListMap = visitor.getSegHitList();

#ifdef _OPENMP
#pragma omp critical(x)
#endif
               {
                  for(LineSegmentHitListMap::const_iterator i = hitListMap.begin(); i != hitListMap.end(); ++i)
                  {
                     _segHitList[i->first].insert(_segHitList[i->first].end(), i->second.begin(), i->second.end());
                     std::sort(_segHitList[i->first].begin(), _segHitList[i->first].end());
                  }

               }
               // END #pragma omp critical

            }
         }
         // END #pragma omp task
      }
      }
   }

   leaveNode();
   }
}

opencover::Private::coIntersectionSubVisitor::coIntersectionSubVisitor(const coIntersectionVisitor & visitor)
   : IntersectVisitor(visitor)
{
   // Everything is copied but _segHitList
   _segHitList.clear();
}


void opencover::Private::coIntersectionSubVisitor::intersect(Drawable * drawable)
{
   IntersectVisitor::intersect(*drawable);
}
#else

void opencover::Private::coIntersectionVisitor::apply(osg::Geode &geode)
{
    if (enterNode(geode))
    {
        for (unsigned int i = 0; i < geode.getNumDrawables(); i++)
        {
            osg::Drawable *d = geode.getDrawable(i);
            if (d)
            {
                intersect(*d);
            }
        }

        leaveNode();
    }
}

#endif
