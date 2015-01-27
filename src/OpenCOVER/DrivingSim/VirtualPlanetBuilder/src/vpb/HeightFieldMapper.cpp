/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
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

#include <vpb/HeightFieldMapper>

#include <osg/Array>
#include <osg/Geometry>
#include <osg/Shape>
#include <osg/Geode>

#include <osgUtil/OperationArrayFunctor>
#include <osgUtil/EdgeCollector>
#include <osgUtil/ConvertVec>
#include <osgUtil/Tessellator>
#include <stdio.h>

namespace vpb
{

HeightFieldMapper::HeightFieldMapper(osg::HeightField &hf)
    : _mappingMode(PER_VERTEX)
    , _hf(hf)
{
    _xMin = _hf.getOrigin().x();
    _yMin = _hf.getOrigin().y();
    _xMax = _xMin + _hf.getXInterval() * double(_hf.getNumColumns() - 1);
    _yMax = _yMin + _hf.getYInterval() * double(_hf.getNumRows() - 1);
}
HeightFieldMapper::HeightFieldMapper(osg::HeightField &hf, double xMin, double xMax, double yMin, double yMax)
    : _mappingMode(PER_VERTEX)
    , _hf(hf)
    , _xMin(xMin)
    , _yMin(yMin)
    , _xMax(xMax)
    , _yMax(yMax)
{
}

HeightFieldMapper::~HeightFieldMapper()
{
}

class SetZeroToZVisitor : public osg::ArrayVisitor
{
public:
    void apply(osg::Vec3Array &array) { computeCentroid<osg::Vec3Array>(array); }
    void apply(osg::Vec4Array &array) { computeCentroid<osg::Vec4Array>(array); }

    void apply(osg::Vec3dArray &array) { computeCentroid<osg::Vec3dArray>(array); }
    void apply(osg::Vec4dArray &array) { computeCentroid<osg::Vec4dArray>(array); }

    template <typename ArrayType>
    void computeCentroid(ArrayType &array)
    {
        unsigned int size = array.size();

        for (unsigned int i = 0; i < size; ++i)
            array[i].z() = 0.0;
    }
};

class ComputeCentroidVisitor : public osg::ArrayVisitor
{
public:
    void apply(osg::Vec3Array &array) { computeCentroid<osg::Vec3Array>(array); }
    void apply(osg::Vec4Array &array) { computeCentroid<osg::Vec4Array>(array); }

    void apply(osg::Vec3dArray &array) { computeCentroid<osg::Vec3dArray>(array); }
    void apply(osg::Vec4dArray &array) { computeCentroid<osg::Vec4dArray>(array); }

    template <typename ArrayType>
    void computeCentroid(ArrayType &array)
    {
        unsigned int size = array.size();
        osg::Vec3d total(0.0, 0.0, 0.0);

        for (unsigned int i = 0; i < size; ++i)
        {
            typename ArrayType::ElementDataType &vec = array[i];
            total.x() += vec.x();
            total.y() += vec.y();
            total.z() += vec.z();
        }

        _centroid = total / double(size);
    }

    osg::Vec3d _centroid;
};

struct CopyIndexedOperator
{
    template <typename ArrayType>
    void process(ArrayType &array)
    {
        ArrayType *va = new ArrayType();

        osg::UIntArray::iterator it, end = _indexArray->end();
        for (it = _indexArray->begin(); it < end; ++it)
            va->push_back(array[*it]);

        _copyArray = va;
    }

    osg::ref_ptr<osg::UIntArray> _indexArray;
    osg::ref_ptr<osg::Array> _copyArray;
};
typedef osgUtil::OperationArrayFunctor<CopyIndexedOperator> CopyIndexedFunctor;

struct IsConvexPolygonVisitor : public osg::ArrayVisitor
{
    virtual void apply(osg::Vec3Array &array) { process<osg::Vec3Array>(array); }
    virtual void apply(osg::Vec4Array &array) { process<osg::Vec4Array>(array); }

    virtual void apply(osg::Vec3dArray &array) { process<osg::Vec3dArray>(array); }
    virtual void apply(osg::Vec4dArray &array) { process<osg::Vec4dArray>(array); }

    template <typename ArrayType>
    void process(ArrayType &array)
    {
        typedef typename ArrayType::ElementDataType VecType;

        bool positif;

        {
            const VecType &v1 = array[0];
            const VecType &v2 = array[1];
            const VecType &v3 = array[2];

            positif = (((v1.x() - v2.x()) * (v3.y() - v2.y()) - (v1.y() - v2.y()) * (v3.x() - v2.x())) >= 0.0);
        }

        unsigned int size = array.size() - 2;
        unsigned int i;
        for (i = 1; i < size; ++i)
        {
            const VecType &v1 = array[i];
            const VecType &v2 = array[i + 1];
            const VecType &v3 = array[i + 2];

            if (positif != (((v1.x() - v2.x()) * (v3.y() - v2.y()) - (v1.y() - v2.y()) * (v3.x() - v2.x())) >= 0.0))
            {
                _isConvex = false;
                return;
            }
        }

        {
            const VecType &v1 = array[i];
            const VecType &v2 = array[i + 1];
            const VecType &v3 = array[0];

            if (positif != (((v1.x() - v2.x()) * (v3.y() - v2.y()) - (v1.y() - v2.y()) * (v3.x() - v2.x())) >= 0.0))
            {
                _isConvex = false;
                return;
            }
        }

        {
            const VecType &v1 = array[i + 1];
            const VecType &v2 = array[0];
            const VecType &v3 = array[1];

            if (positif != (((v1.x() - v2.x()) * (v3.y() - v2.y()) - (v1.y() - v2.y()) * (v3.x() - v2.x())) >= 0.0))
            {
                _isConvex = false;
                return;
            }
        }

        _isConvex = true;
    }

    bool _isConvex;
};

// ** This operator map each vertex and add them only if they overlap the HeightField
class HeightFieldMapperArrayVisitor : public osg::ArrayVisitor
{
public:
    HeightFieldMapperArrayVisitor(const HeightFieldMapper &hfm)
        : _hfm(hfm)
    {
    }

    virtual void apply(osg::Vec3Array &array) { process<osg::Vec3Array>(array); }
    virtual void apply(osg::Vec4Array &array) { process<osg::Vec4Array>(array); }

    virtual void apply(osg::Vec3dArray &array) { process<osg::Vec3dArray>(array); }
    virtual void apply(osg::Vec4dArray &array) { process<osg::Vec4dArray>(array); }

    template <typename ArrayType>
    void process(ArrayType &array)
    {
        unsigned int size = array.size();
        osg::ref_ptr<ArrayType> newArray(new ArrayType);
        newArray->reserve(size);

        ArrayType &refNewArray = *newArray.get();

        for (unsigned int i = 0; i < size; ++i)
        {
            typename ArrayType::ElementDataType &vec = array[i];
            vec.z() = _hfm.getZfromXY(vec.x(), vec.y());

            if (vec.z() != DBL_MAX)
                refNewArray.push_back(vec);
        }

        array = refNewArray;
    }

    const HeightFieldMapper &_hfm;
};

// ** This operator reverse the array if its normal is not (0,0,1)
class AssertUpNormalVisitor : public osg::ArrayVisitor
{
public:
    virtual void apply(osg::Vec3Array &array) { process<osg::Vec3Array>(array); }
    virtual void apply(osg::Vec4Array &array) { process<osg::Vec4Array>(array); }

    virtual void apply(osg::Vec3dArray &array) { process<osg::Vec3dArray>(array); }
    virtual void apply(osg::Vec4dArray &array) { process<osg::Vec4dArray>(array); }

    template <typename ArrayType>
    void process(ArrayType &array) const
    {
        typedef typename ArrayType::ElementDataType VecType;

        const VecType &v1 = array[0];
        const VecType &v2 = array[1];
        const VecType &v3 = array[2];

        if (((v1.x() - v2.x()) * (v3.y() - v2.y()) - (v1.y() - v2.y()) * (v3.x() - v2.x())) > 0.0)
        {
            std::reverse(array.begin(), array.end());
        }
    }
};

// ** This visitor cut the geometry to overlap exactly the HeightField
template <typename ArrayType, typename CreatePolicy>
class CutGeometryToOverlapHeightField
{
public:
    typedef typename ArrayType::ElementDataType VecType;

    CutGeometryToOverlapHeightField(double xMin, double yMin, double xMax, double yMax)
        : _xMin(xMin)
        , _yMin(yMin)
        , _xMax(xMax)
        , _yMax(yMax)
    {
    }

    bool overlapHeightField(double x, double y) const
    {
        return !((x > _xMax) || (x < _xMin) || (y > _yMax) || (y < _yMin));
    }

    void process(ArrayType &array)
    {

        unsigned int size = array.size();
        _cuttenVertexArray = new ArrayType;

        ArrayType &refNewArray = *static_cast<ArrayType *>(_cuttenVertexArray.get());
        refNewArray.reserve(size);

        // ** find the first vertex overlaping the HeightField
        unsigned int i = 0;
        bool notFound = true;
        while (notFound && (i < size))
        {
            VecType &vec = array[i];
            if (overlapHeightField(vec.x(), vec.y()))
                notFound = false;
            else
                ++i;
        }

        if (i == size)
            return;

        // ** move vertex at the end of the array. Now, the first vertex in the array overlap the HeightField
        if (i)
        {
            std::rotate(array.begin(), array.begin() + i, array.end());
        }

        VecType inToOutVec, outToInVec;

        refNewArray.push_back(array[0]);
        i = 1;
        while (i < size)
        {
            VecType &vec = array[i];

            // ** if the vertex not overlap the HeightField
            if (overlapHeightField(vec.x(), vec.y()) == false)
            {
                computeIntersection(array[i - 1], array[i], inToOutVec);
                ++i;

                bool notOverlap = true;
                while (notOverlap && (i < size))
                {
                    VecType &v = array[i];
                    if (overlapHeightField(v.x(), v.y()))
                        notOverlap = false;
                    else
                        ++i;
                }

                if (notOverlap == false)
                    computeIntersection(array[i], array[i - 1], outToInVec);
                else
                    computeIntersection(array[0], array[i - 1], outToInVec);

                insertIntersectedVertex(refNewArray, inToOutVec, outToInVec);

                if (notOverlap == false)
                    refNewArray.push_back(array[i]);
            }
            else
            {
                refNewArray.push_back(vec);
            }

            ++i;
        }
    }

    enum Corner
    {
        NotACorner = 0,
        C00, // corner(_xMin, _yMin)
        C10, // corner(_xMax, _yMin)
        C11, // corner(_xMax, _yMax)
        C01 // corner(_xMin, _yMax)
    };

    Corner isCorner(const VecType &v)
    {
        if (v.x() == _xMin)
        {
            if (v.y() == _yMin)
                return C00;
            if (v.y() == _yMax)
                return C01;
        }
        if (v.x() == _xMax)
        {
            if (v.y() == _yMin)
                return C10;
            if (v.y() == _yMax)
                return C11;
        }
        return (NotACorner);
    }

    void insertIntersectedVertex(ArrayType &array, const VecType &v1, const VecType &v2)
    {
        array.push_back(v1);

        if (v1.x() == _xMin)
        {
            if (v2.y() == _yMin)
            {
                if ((isCorner(v1) != C00) && (isCorner(v2) != C00))
                    array.push_back(CreatePolicy::create(_xMin, _yMin));
            }
            else if (v2.x() == _xMax)
            {
                if (isCorner(v1) != C00)
                    array.push_back(CreatePolicy::create(_xMin, _yMin));
                array.push_back(CreatePolicy::create(_xMax, _yMin));
            }
            else if (v2.y() == _yMax)
            {
                if (isCorner(v1) != C00)
                    array.push_back(CreatePolicy::create(_xMin, _yMin));
                array.push_back(CreatePolicy::create(_xMax, _yMin));
                array.push_back(CreatePolicy::create(_xMax, _yMax));
            }
            else if (v2.x() == _xMin)
            {
                if (v1.y() < v2.y())
                {
                    if (isCorner(v1) != C00)
                        array.push_back(CreatePolicy::create(_xMin, _yMin));
                    array.push_back(CreatePolicy::create(_xMax, _yMin));
                    array.push_back(CreatePolicy::create(_xMax, _yMax));
                    array.push_back(CreatePolicy::create(_xMin, _yMax));
                }
            }
        }

        else if (v1.y() == _yMin)
        {
            if (v2.x() == _xMax)
            {
                if ((isCorner(v1) != C10) && (isCorner(v2) != C10))
                    array.push_back(CreatePolicy::create(_xMax, _yMin));
            }
            else if (v2.y() == _yMax)
            {
                if (isCorner(v1) != C10)
                    array.push_back(CreatePolicy::create(_xMax, _yMin));
                array.push_back(CreatePolicy::create(_xMax, _yMax));
            }
            else if (v2.x() == _xMin)
            {
                if (isCorner(v1) != C10)
                    array.push_back(CreatePolicy::create(_xMax, _yMin));
                array.push_back(CreatePolicy::create(_xMax, _yMax));
                array.push_back(CreatePolicy::create(_xMin, _yMax));
            }
            else if (v2.y() == _yMin)
            {
                if (v1.x() > v2.x())
                {
                    if (isCorner(v1) != C10)
                        array.push_back(CreatePolicy::create(_xMax, _yMin));
                    array.push_back(CreatePolicy::create(_xMax, _yMax));
                    array.push_back(CreatePolicy::create(_xMin, _yMax));
                    array.push_back(CreatePolicy::create(_xMin, _yMin));
                }
            }
        }

        else if (v1.x() == _xMax)
        {
            if (v2.y() == _yMax)
            {
                if ((isCorner(v1) != C11) && (isCorner(v2) != C11))
                    array.push_back(CreatePolicy::create(_xMax, _yMax));
            }
            else if (v2.x() == _xMin)
            {
                if (isCorner(v1) != C11)
                    array.push_back(CreatePolicy::create(_xMax, _yMax));
                array.push_back(CreatePolicy::create(_xMax, _yMin));
            }
            else if (v2.y() == _yMin)
            {
                if (isCorner(v1) != C11)
                    array.push_back(CreatePolicy::create(_xMax, _yMax));
                array.push_back(CreatePolicy::create(_xMin, _yMax));
                array.push_back(CreatePolicy::create(_xMin, _yMin));
            }
            else if (v2.x() == _xMax)
            {
                if (v1.y() > v2.y())
                {
                    if (isCorner(v1) != C11)
                        array.push_back(CreatePolicy::create(_xMax, _yMax));
                    array.push_back(CreatePolicy::create(_xMin, _yMax));
                    array.push_back(CreatePolicy::create(_xMin, _yMin));
                    array.push_back(CreatePolicy::create(_xMax, _yMin));
                }
            }
        }

        else if (v1.y() == _yMax)
        {
            if (v2.x() == _xMin)
            {
                if ((isCorner(v1) != C01) && (isCorner(v2) != C01))
                    array.push_back(CreatePolicy::create(_xMin, _yMax));
            }
            else if (v2.y() == _yMin)
            {
                if (isCorner(v1) != C01)
                    array.push_back(CreatePolicy::create(_xMin, _yMax));
                array.push_back(CreatePolicy::create(_xMin, _yMin));
            }
            else if (v2.x() == _xMax)
            {
                if (isCorner(v1) != C01)
                    array.push_back(CreatePolicy::create(_xMin, _yMax));
                array.push_back(CreatePolicy::create(_xMin, _yMin));
                array.push_back(CreatePolicy::create(_xMax, _yMin));
            }
            else if (v2.y() == _yMax)
            {
                if (v1.x() < v2.x())
                {
                    if (isCorner(v1) != C01)
                        array.push_back(CreatePolicy::create(_xMin, _yMax));
                    array.push_back(CreatePolicy::create(_xMin, _yMin));
                    array.push_back(CreatePolicy::create(_xMax, _yMin));
                    array.push_back(CreatePolicy::create(_xMax, _yMax));
                }
            }
        }

        array.push_back(v2);
    }

    void computeIntersection(const VecType &v1, const VecType &v2, VecType &intersec)
    {
        if (v2.x() < _xMin)
        {
            double xCoef = (v2.y() - v1.y()) / (v2.x() - v1.x());
            double dist = _xMin - v1.x();

            intersec.x() = _xMin;
            intersec.y() = v1.y() + xCoef * dist;
        }
        else if (v2.x() > _xMax)
        {
            double xCoef = (v2.y() - v1.y()) / (v2.x() - v1.x());
            double dist = _xMax - v1.x();

            intersec.x() = _xMax;
            intersec.y() = v1.y() + xCoef * dist;
        }
        else if (v2.y() < _yMin)
        {
            double yCoef = (v2.x() - v1.x()) / (v2.y() - v1.y());
            double dist = _yMin - v1.y();

            intersec.x() = v1.x() + yCoef * dist;
            intersec.y() = _yMin;
        }
        else if (v2.y() > _yMax)
        {
            double yCoef = (v2.x() - v1.x()) / (v2.y() - v1.y());
            double dist = _yMax - v1.y();

            intersec.x() = v1.x() + yCoef * dist;
            intersec.y() = _yMax;
        }
    }

    osg::ref_ptr<osg::Array> _cuttenVertexArray;

    double _xMin;
    double _yMin;
    double _xMax;
    double _yMax;
};

template <typename ArrayType>
struct CreatePolicy3
{
    typedef typename ArrayType::ElementDataType VecType;
    static VecType create(double x, double y) { return VecType(x, y, 0); }
};

template <typename ArrayType>
struct CreatePolicy4
{
    typedef typename ArrayType::ElementDataType VecType;
    static VecType create(double x, double y) { return VecType(x, y, 0, 1); }
};

class CutGeometryToOverlapHeightFieldVisitor : public osg::ArrayVisitor
{
public:
    CutGeometryToOverlapHeightFieldVisitor(double xMin, double yMin, double xMax, double yMax)
        : _xMin(xMin)
        , _yMin(yMin)
        , _xMax(xMax)
        , _yMax(yMax)
    {
    }

    void apply(osg::Vec3Array &array)
    {
        CutGeometryToOverlapHeightField<osg::Vec3Array, CreatePolicy3<osg::Vec3Array> > cutter(_xMin, _yMin, _xMax, _yMax);
        cutter.process(array);
        _cuttenVertexArray = cutter._cuttenVertexArray;
    }
    void apply(osg::Vec4Array &array)
    {
        CutGeometryToOverlapHeightField<osg::Vec4Array, CreatePolicy4<osg::Vec4Array> > cutter(_xMin, _yMin, _xMax, _yMax);
        cutter.process(array);
        _cuttenVertexArray = cutter._cuttenVertexArray;
    }
    void apply(osg::Vec3dArray &array)
    {
        CutGeometryToOverlapHeightField<osg::Vec3dArray, CreatePolicy3<osg::Vec3dArray> > cutter(_xMin, _yMin, _xMax, _yMax);
        cutter.process(array);
        _cuttenVertexArray = cutter._cuttenVertexArray;
    }
    void apply(osg::Vec4dArray &array)
    {
        CutGeometryToOverlapHeightField<osg::Vec4dArray, CreatePolicy4<osg::Vec4dArray> > cutter(_xMin, _yMin, _xMax, _yMax);
        cutter.process(array);
        _cuttenVertexArray = cutter._cuttenVertexArray;
    }

    osg::ref_ptr<osg::Array> _cuttenVertexArray;

    double _xMin;
    double _yMin;
    double _xMax;
    double _yMax;
};

bool HeightFieldMapper::getCentroid(osg::Geometry &geometry, osg::Vec3d &centroid) const
{
    if (_mappingMode != PER_GEOMETRY)
        return false;

    // ** compute the OutEdge line
    osgUtil::EdgeCollector ec;
    ec.setGeometry(&geometry);
    if (ec._triangleSet.empty())
        return false;

    // ** get IndexArray of each Edgeloop
    osgUtil::EdgeCollector::IndexArrayList indexArrayList;
    ec.getEdgeloopIndexList(indexArrayList);
    if (indexArrayList.empty())
        return false;

    // ** create a new vertexArray only with vertex composing the out edge line
    CopyIndexedFunctor cif;
    cif._indexArray = indexArrayList.front();
    geometry.getVertexArray()->accept(cif);
    geometry.setVertexArray(cif._copyArray.get());
    if (cif._copyArray.valid() == false)
        return false;

    SetZeroToZVisitor zeroVis;
    geometry.getVertexArray()->accept(zeroVis);

    AssertUpNormalVisitor aunv;
    geometry.getVertexArray()->accept(aunv);

    // ** check if outedge line is concave
    IsConvexPolygonVisitor convexTest;
    geometry.getVertexArray()->accept(convexTest);
    if (convexTest._isConvex == false)
        return false;

    // ** Remove vertex which not overlap the HeightField and insert
    CutGeometryToOverlapHeightFieldVisitor cutter(_xMin, _yMin, _xMax, _yMax);
    geometry.getVertexArray()->accept(cutter);
    geometry.setVertexArray(cutter._cuttenVertexArray.get());
    if (geometry.getVertexArray()->getNumElements() == 0)
        return false;

    // ** compute centroid
    ComputeCentroidVisitor ccv;
    geometry.getVertexArray()->accept(ccv);
    centroid = ccv._centroid;

    // ** recreate the geometry
    geometry.getPrimitiveSetList().clear();
    geometry.addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, geometry.getVertexArray()->getNumElements()));

    return true;
}

bool HeightFieldMapper::map(osg::Geometry &geometry) const
{
    if (_mappingMode == PER_VERTEX)
    {
        HeightFieldMapperArrayVisitor hfmv(*this);
        geometry.getVertexArray()->accept(hfmv);

        return geometry.getVertexArray()->getNumElements() != 0;
    }

    if (_mappingMode == PER_GEOMETRY)
    {
        osg::Vec3d centroid(0.0, 0.0, 0.0);
        if (getCentroid(geometry, centroid) == false)
            return false;

        // ** get z value to add to Z coordinates
        double zHeightField = getZfromXY(centroid.x(), centroid.y());
        if (zHeightField == DBL_MAX)
            return false;

        double z = zHeightField - centroid.z();

        // add z value to z coordinates to all vertex
        osgUtil::AddRangeFunctor arf;
        arf._vector = osg::Vec3d(0, 0, z);
        arf._begin = 0;
        arf._count = geometry.getVertexArray()->getNumElements();
        geometry.getVertexArray()->accept(arf);

        return true;
    }

    return false;
}

double HeightFieldMapper::getZfromXY(double x, double y) const
{
    if ((x > _xMax) || (x < _xMin) || (y > _yMax) || (y < _yMin))
        return DBL_MAX;

    /*fprintf(stderr,"X %lf\n",x);
    if(x > 3491510)
        return 10.0;
    if(x < 3491310)
        return 100.0;*/

    double dx_origin = x - _hf.getOrigin().x();
    double dy_origin = y - _hf.getOrigin().y();

    // compute the cell coordinates
    double cx = dx_origin / double(_hf.getXInterval());
    double cy = dy_origin / double(_hf.getYInterval());

    // compute the cell by taking the floor
    double fx = floor(cx);
    double fy = floor(cy);
    int c = static_cast<int>(fx);
    int r = static_cast<int>(fy);

    // compute the local cell ratio.
    double rx = cx - fx;
    double ry = cy - fy;

    double total_ratio = 0.0;
    double total_height = 0.0;

    int numColumns = static_cast<int>(_hf.getNumColumns());
    int numRows = static_cast<int>(_hf.getNumRows());

    if ((c >= 0 && c < numColumns) && (r >= 0 && r < numRows))
    {
        double local_ratio = (1.0 - rx) * (1.0 - ry);
        total_ratio += local_ratio;
        total_height += _hf.getHeight(c, r) * local_ratio;
    }

    if (((c + 1) >= 0 && (c + 1) < numColumns) && (r >= 0 && r < numRows))
    {
        double local_ratio = rx * (1.0 - ry);
        total_ratio += local_ratio;
        total_height += _hf.getHeight(c + 1, r) * local_ratio;
    }

    if ((c >= 0 && c < numColumns) && ((r + 1) >= 0 && (r + 1) < numRows))
    {
        double local_ratio = (1.0 - rx) * ry;
        total_ratio += local_ratio;
        total_height += _hf.getHeight(c, r + 1) * local_ratio;
    }

    if (((c + 1) >= 0 && (c + 1) < numColumns) && ((r + 1) >= 0 && (r + 1) < numRows))
    {
        double local_ratio = rx * ry;
        total_ratio += local_ratio;
        total_height += _hf.getHeight(c + 1, r + 1) * local_ratio;
    }

    if (total_ratio > 0.0)
        return _hf.getOrigin().z() + total_height / total_ratio;
    else
        return _hf.getOrigin().z();
}

void HeightFieldMapperVisitor::apply(osg::Geode &node)
{
    unsigned int numDrawable = node.getNumDrawables();

    for (unsigned int i = 0; i < numDrawable; ++i)
    {
        osg::Geometry *geo = dynamic_cast<osg::Geometry *>(node.getDrawable(i));

        if (geo)
        {
            _hfm.map(*geo);
        }
    }
}

} // end vpb namespace
