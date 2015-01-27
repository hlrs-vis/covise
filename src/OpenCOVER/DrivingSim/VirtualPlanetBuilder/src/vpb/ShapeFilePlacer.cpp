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

#include <vpb/ShapeFilePlacer>
#include <vpb/Destination>
#include <vpb/DataSet>
#include <vpb/HeightFieldMapper>
#include <vpb/ExtrudeVisitor>
#include <vpb/System>

#include <osg/NodeVisitor>
#include <osg/Array>

#include <osgUtil/ConvertVec>
#include <osgUtil/DrawElementTypeSimplifier>
#include <osgUtil/SmoothingVisitor>

#include <osgSim/ShapeAttribute>

#include <osg/Material>

using namespace vpb;

struct MatrixMultiplyArrayFunctor
{
    MatrixMultiplyArrayFunctor(const osg::Matrixd &matrix, const osg::EllipsoidModel *em = 0)
        : _matrix(matrix)
        , _em(em)
    {
    }

    void operator()(osg::Vec3d &vec)
    {
        if (_em)
        {
            double latitude = osg::DegreesToRadians(vec.y());
            double longitude = osg::DegreesToRadians(vec.x());
            double height = vec.z();
            _em->convertLatLongHeightToXYZ(latitude, longitude, height,
                                           vec.x(), vec.y(), vec.z());
        }

        vec = osg::Vec3d(vec.x() * _matrix(0, 0) + vec.y() * _matrix(1, 0) + vec.z() * _matrix(2, 0) + _matrix(3, 0),
                         vec.x() * _matrix(0, 1) + vec.y() * _matrix(1, 1) + vec.z() * _matrix(2, 1) + _matrix(3, 1),
                         vec.x() * _matrix(0, 2) + vec.y() * _matrix(1, 2) + vec.z() * _matrix(2, 2) + _matrix(3, 2));
    }

    const osg::Matrixd &_matrix;
    const osg::EllipsoidModel *_em;
};

class DoubleToFloatVisitor : public osg::ArrayVisitor
{
public:
    virtual void apply(osg::Vec3dArray &array)
    {
        unsigned int size = array.size();
        _vertexArray = new osg::Vec3Array(size);
        osg::Vec3Array &va = *(static_cast<osg::Vec3Array *>(_vertexArray.get()));

        for (unsigned int i = 0; i < size; ++i)
            osgUtil::ConvertVec<osg::Vec3d, osg::Vec3>::convert(array[i], va[i]);
    }
    virtual void apply(osg::Vec4dArray &array)
    {
        unsigned int size = array.size();
        _vertexArray = new osg::Vec4Array(size);
        osg::Vec4Array &va = *(static_cast<osg::Vec4Array *>(_vertexArray.get()));

        for (unsigned int i = 0; i < size; ++i)
            osgUtil::ConvertVec<osg::Vec4d, osg::Vec4>::convert(array[i], va[i]);
    }

    osg::ref_ptr<osg::Array> _vertexArray;
};

class BoundingBoxd
{
public:
    /** Minimum extent. (Smallest X, Y, and Z values of all coordinates.) */
    osg::Vec3d _min;
    /** Maximum extent. (Greatest X, Y, and Z values of all coordinates.) */
    osg::Vec3d _max;

    /** Creates an uninitialized bounding box. */
    inline BoundingBoxd()
        : _min(DBL_MAX, DBL_MAX, DBL_MAX)
        , _max(-DBL_MAX, -DBL_MAX, -DBL_MAX)
    {
    }

    /** Creates a bounding box initialized to the given extents. */
    inline BoundingBoxd(double xmin, double ymin, double zmin,
                        double xmax, double ymax, double zmax)
        : _min(xmin, ymin, zmin)
        , _max(xmax, ymax, zmax)
    {
    }

    /** Creates a bounding box initialized to the given extents. */
    inline BoundingBoxd(const osg::Vec3d &min, const osg::Vec3d &max)
        : _min(min)
        , _max(max)
    {
    }

    /** Clear the bounding box. Erases existing minimum and maximum extents. */
    inline void init()
    {
        _min.set(DBL_MAX, DBL_MAX, DBL_MAX);
        _max.set(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    }

    /** Returns true if the bounding box extents are valid, false otherwise. */
    inline bool valid() const
    {
        return _max.x() >= _min.x() && _max.y() >= _min.y() && _max.z() >= _min.z();
    }

    /** Sets the bounding box extents. */
    inline void set(double xmin, double ymin, double zmin,
                    double xmax, double ymax, double zmax)
    {
        _min.set(xmin, ymin, zmin);
        _max.set(xmax, ymax, zmax);
    }

    /** Sets the bounding box extents. */
    inline void set(const osg::Vec3d &min, const osg::Vec3d &max)
    {
        _min = min;
        _max = max;
    }

    inline double &xMin() { return _min.x(); }
    inline double xMin() const { return _min.x(); }

    inline double &yMin() { return _min.y(); }
    inline double yMin() const { return _min.y(); }

    inline double &zMin() { return _min.z(); }
    inline double zMin() const { return _min.z(); }

    inline double &xMax() { return _max.x(); }
    inline double xMax() const { return _max.x(); }

    inline double &yMax() { return _max.y(); }
    inline double yMax() const { return _max.y(); }

    inline double &zMax() { return _max.z(); }
    inline double zMax() const { return _max.z(); }

    /** Calculates and returns the bounding box center. */
    inline const osg::Vec3d center() const
    {
        return (_min + _max) * 0.5f;
    }

    /** Calculates and returns the bounding box radius. */
    inline double radius() const
    {
        return sqrtf(radius2());
    }

    /** Calculates and returns the squared length of the bounding box radius.
          * Note, radius2() is faster to calculate than radius(). */
    inline double radius2() const
    {
        return 0.25f * ((_max - _min).length2());
    }

    /** Returns a specific corner of the bounding box.
          * pos specifies the corner as a number between 0 and 7.
          * Each bit selects an axis, X, Y, or Z from least- to
          * most-significant. Unset bits select the minimum value
          * for that axis, and set bits select the maximum. */
    inline const osg::Vec3d corner(unsigned int pos) const
    {
        return osg::Vec3d(pos & 1 ? _max.x() : _min.x(), pos & 2 ? _max.y() : _min.y(), pos & 4 ? _max.z() : _min.z());
    }

    /** Expands the bounding box to include the given coordinate.
          * If the box is uninitialized, set its min and max extents to v. */
    inline void expandBy(const osg::Vec3d &v)
    {
        if (v.x() < _min.x())
            _min.x() = v.x();
        if (v.x() > _max.x())
            _max.x() = v.x();

        if (v.y() < _min.y())
            _min.y() = v.y();
        if (v.y() > _max.y())
            _max.y() = v.y();

        if (v.z() < _min.z())
            _min.z() = v.z();
        if (v.z() > _max.z())
            _max.z() = v.z();
    }

    /** Expands the bounding box to include the given coordinate.
          * If the box is uninitialized, set its min and max extents to
          * osg::Vec3d(x,y,z). */
    inline void expandBy(double x, double y, double z)
    {
        if (x < _min.x())
            _min.x() = x;
        if (x > _max.x())
            _max.x() = x;

        if (y < _min.y())
            _min.y() = y;
        if (y > _max.y())
            _max.y() = y;

        if (z < _min.z())
            _min.z() = z;
        if (z > _max.z())
            _max.z() = z;
    }

    /** Returns the intersection of this bounding box and the specified bounding box. */
    BoundingBoxd intersect(const BoundingBoxd &bb) const
    {
        return BoundingBoxd(osg::maximum(xMin(), bb.xMin()), osg::maximum(yMin(), bb.yMin()), osg::maximum(zMin(), bb.zMin()),
                            osg::minimum(xMax(), bb.xMax()), osg::minimum(yMax(), bb.yMax()), osg::minimum(zMax(), bb.zMax()));
    }

    /** Return true if this bounding box intersects the specified bounding box. */
    bool intersects(const BoundingBoxd &bb) const
    {
        return osg::maximum(xMin(), bb.xMin()) <= osg::minimum(xMax(), bb.xMax()) && osg::maximum(yMin(), bb.yMin()) <= osg::minimum(yMax(), bb.yMax()) && osg::maximum(zMin(), bb.zMin()) <= osg::minimum(zMax(), bb.zMax());
    }

    /** Returns true if this bounding box contains the specified coordinate. */
    inline bool contains(const osg::Vec3d &v) const
    {
        return valid() && (v.x() >= _min.x() && v.x() <= _max.x()) && (v.y() >= _min.y() && v.y() <= _max.y()) && (v.z() >= _min.z() && v.z() <= _max.z());
    }
};

struct ComputeBoundd : public osg::PrimitiveFunctor
{
    virtual void setVertexArray(unsigned int, const osg::Vec2 *) {}
    virtual void setVertexArray(unsigned int, const osg::Vec4 *) {}
    virtual void setVertexArray(unsigned int, const osg::Vec2d *) {}
    virtual void setVertexArray(unsigned int, const osg::Vec4d *) {}

    virtual void vertex(const osg::Vec2 &) {}
    virtual void vertex(const osg::Vec4 &) {}
    virtual void vertex(float, float) {}
    virtual void vertex(float, float, float) {}
    virtual void vertex(float, float, float, float) {}

    virtual void begin(GLenum) {}
    virtual void end() {}

    ComputeBoundd()
    {
        _vertices3f = 0;
        _vertices3d = 0;
    }

    virtual void setVertexArray(unsigned int, const osg::Vec3 *vertices) { _vertices3f = vertices; }
    virtual void setVertexArray(unsigned int, const osg::Vec3d *vertices) { _vertices3d = vertices; }

    template <typename T>
    void _drawArrays(T *vert, T *end)
    {
        for (; vert < end; ++vert)
        {
            vertex(*vert);
        }
    }

    template <typename T, typename I>
    void _drawElements(T *vert, I *indices, I *end)
    {
        for (; indices < end; ++indices)
        {
            vertex(vert[*indices]);
        }
    }

    virtual void drawArrays(GLenum, GLint first, GLsizei count)
    {
        if (_vertices3f)
            _drawArrays(_vertices3f + first, _vertices3f + (first + count));
        else if (_vertices3d)
            _drawArrays(_vertices3d + first, _vertices3d + (first + count));
    }

    virtual void drawElements(GLenum, GLsizei count, const GLubyte *indices)
    {
        if (_vertices3f)
            _drawElements(_vertices3f, indices, indices + count);
        else if (_vertices3d)
            _drawElements(_vertices3d, indices, indices + count);
    }

    virtual void drawElements(GLenum, GLsizei count, const GLushort *indices)
    {
        if (_vertices3f)
            _drawElements(_vertices3f, indices, indices + count);
        else if (_vertices3d)
            _drawElements(_vertices3d, indices, indices + count);
    }

    virtual void drawElements(GLenum, GLsizei count, const GLuint *indices)
    {
        if (_vertices3f)
            _drawElements(_vertices3f, indices, indices + count);
        else if (_vertices3d)
            _drawElements(_vertices3d, indices, indices + count);
    }

    virtual void vertex(const osg::Vec3 &vert) { _bb.expandBy(vert); }
    virtual void vertex(const osg::Vec3d &vert) { _bb.expandBy(vert); }

    const osg::Vec3 *_vertices3f;
    const osg::Vec3d *_vertices3d;
    BoundingBoxd _bb;
};

class ShapeFileOverlapingHeightFieldPlacer : public osg::NodeVisitor
{
public:
    enum ShapeFileType
    {
        Forest,
        Building
    };

    ShapeFileOverlapingHeightFieldPlacer(DestinationTile &dt, osg::HeightField &hf)
        : _hf(hf)
        , _dt(dt)
        , _TypeAttributeName("TypeAttributeName")
        , _HeightAttributeName("HeightAttributeName")
    {
        _createdModel = new osg::Group;
        _nodeStack.push_back(_createdModel.get());
        _typeAttributeNameStack.push_back("NAME");
        _heightAttributeNameStack.push_back("HGT");
    }

    virtual void apply(osg::Node &node)
    {
        // ** if sub-graph overlap the HeightField
        if (overlap(node.getBound()))
        {
            // ** clone the node
            osg::ref_ptr<osg::Node> clonedNode = static_cast<osg::Node *>(node.clone(osg::CopyOp::SHALLOW_COPY));

            // ** if it's osg::Group type, clean children
            osg::Group *clonedGroup = clonedNode->asGroup();
            if (clonedGroup)
                clonedGroup->removeChild(0, clonedGroup->getNumChildren());

            addAndTraverse(node, clonedNode.get());
        }
    }

    void addAndTraverse(osg::Node &node, osg::Node *clonedNode)
    {
        // ** link the child with new scene graph
        osg::Group *group = _nodeStack.back()->asGroup();
        if (group)
            group->addChild(clonedNode);

        // ** push node, traverse, pop node
        _nodeStack.push_back(clonedNode);
        traverse(node);
        _nodeStack.pop_back();
    }

    bool pushAttributeNames(osg::Node &node)
    {
        if (node.getDescriptions().empty())
            return false;

        std::string heightAttributeName;
        std::string typeAttributeName;

        const osg::Node::DescriptionList &descriptions = node.getDescriptions();
        for (osg::Node::DescriptionList::const_iterator itr = descriptions.begin();
             itr != descriptions.end();
             ++itr)
        {
            const std::string &desc = *itr;
            getAttributeValue(desc, _TypeAttributeName, typeAttributeName);
            getAttributeValue(desc, _HeightAttributeName, heightAttributeName);
        }

        if (!heightAttributeName.empty() || !typeAttributeName.empty())
        {
            if (typeAttributeName.empty() && !_typeAttributeNameStack.empty())
                typeAttributeName = _typeAttributeNameStack.back();
            if (heightAttributeName.empty() && !_heightAttributeNameStack.empty())
                heightAttributeName = _heightAttributeNameStack.back();

            // osg::notify(osg::NOTICE)<<"PushingAttributeName("<<typeAttributeName<<","<<heightAttributeName<<")"<<std::endl;

            _typeAttributeNameStack.push_back(typeAttributeName);
            _heightAttributeNameStack.push_back(heightAttributeName);

            return true;
        }

        return false;
    }

    void popAttributeNames()
    {
        if (_typeAttributeNameStack.size() > 1)
            _typeAttributeNameStack.pop_back();
        if (_heightAttributeNameStack.size() > 1)
            _heightAttributeNameStack.pop_back();
    }

    const std::string &getTypeAttributeName() const
    {
        return _typeAttributeNameStack.back();
    }

    const std::string &getHeightAttributeName() const
    {
        return _heightAttributeNameStack.back();
    }

    virtual void apply(osg::Geode &node)
    {
        const osg::EllipsoidModel *em = _dt._dataSet->getEllipsoidModel();
        bool mapLatLongsToXYZ = _dt._dataSet->mapLatLongsToXYZ();
        bool useLocalToTileTransform = _dt._dataSet->getUseLocalTileTransform();
        double verticalScale = _dt._dataSet->getVerticalScale();

        // const osg::Matrixd& localToWorld = _dt._localToWorld;
        const osg::Matrixd &worldToLocal = _dt._worldToLocal;

        const osg::BoundingBox &bb = node.getBoundingBox();

        // ** if sub-graph overlap the HeightField
        if (overlap(bb.xMin(), bb.yMin(), bb.xMax(), bb.yMax()))
        {
            // ** clone the geode
            osg::ref_ptr<osg::Geode> clonedGeode = static_cast<osg::Geode *>(node.clone(osg::CopyOp::SHALLOW_COPY));

            // ** clean drawable
            unsigned int numDrawables = clonedGeode->getNumDrawables();
            clonedGeode->removeDrawables(0, numDrawables);

            bool pushedAttributeNames = pushAttributeNames(node);

            for (unsigned int i = 0; i < numDrawables; ++i)
            {
                // ** get the geometry
                osg::Drawable *drawable = node.getDrawable(i);
                if (drawable == NULL)
                    continue;

                osg::Geometry *geom = drawable->asGeometry();
                if (geom == NULL)
                    continue;

                ShapeFileType shapeType = Building;
                double height = 10.0;

                osgSim::ShapeAttributeList *sal = dynamic_cast<osgSim::ShapeAttributeList *>(geom->getUserData());
                for (osgSim::ShapeAttributeList::iterator sitr = sal->begin(); sitr != sal->end(); ++sitr)
                {
                    if ((sitr->getName() == getTypeAttributeName()) && (sitr->getType() == osgSim::ShapeAttribute::STRING))
                    {
                        if (sitr->getString())
                        {
                            if (strncmp(sitr->getString(), "Building", 8) == 0)
                                shapeType = Building;
                            else if (strncmp(sitr->getString(), "Forest", 6) == 0)
                                shapeType = Forest;
                        }
                    }

                    else if (sitr->getName() == getHeightAttributeName())
                    {
                        if (sitr->getType() == osgSim::ShapeAttribute::DOUBLE)
                            height = sitr->getDouble();
                        else if (sitr->getType() == osgSim::ShapeAttribute::INTEGER)
                            height = double(sitr->getInt());
                    }
                }

                height *= verticalScale;

                // ** if geometry overlap the HeightField
                ComputeBoundd cb;
                geom->accept(cb);

                const BoundingBoxd &geoBB = cb._bb;
                if (overlap(geoBB.xMin(), geoBB.yMin(), geoBB.xMax(), geoBB.yMax()))
                {
                    osg::ref_ptr<osg::Geometry> clonedGeom = static_cast<osg::Geometry *>(geom->clone(osg::CopyOp::DEEP_COPY_ARRAYS | osg::CopyOp::DEEP_COPY_PRIMITIVES));

                    HeightFieldMapper hfm(_hf, _dt._extents.xMin(), _dt._extents.xMax(), _dt._extents.yMin(), _dt._extents.yMax());
                    hfm.setMode(shapeType == Building ? HeightFieldMapper::PER_GEOMETRY : HeightFieldMapper::PER_VERTEX);

                    // ** if the geometry have centroid out of the HeightField,
                    // **  don't extrude and insert the geometry in scene graph
                    if (hfm.map(*clonedGeom.get()))
                    {
                        osg::Vec3d vec(0.0, 0.0, height);

                        // ** Extrude the geometry
                        ExtrudeVisitor ev(shapeType == Building ? ExtrudeVisitor::PER_GEOMETRY : ExtrudeVisitor::PER_VERTEX,
                                          ExtrudeVisitor::Replace,
                                          vec);
                        ev.extrude(*clonedGeom.get());

                        if (useLocalToTileTransform || mapLatLongsToXYZ)
                        {
                            osg::Vec3dArray *vertexArray = dynamic_cast<osg::Vec3dArray *>(clonedGeom->getVertexArray());
                            MatrixMultiplyArrayFunctor mmaf(worldToLocal, mapLatLongsToXYZ ? em : 0);
                            std::for_each(vertexArray->begin(), vertexArray->end(), mmaf);
                        }

                        // ** replace VertexArray type osg::Vec3dArray by osg::Vec3Array
                        DoubleToFloatVisitor dtfVisitor;
                        clonedGeom->getVertexArray()->accept(dtfVisitor);
                        clonedGeom->setVertexArray(dtfVisitor._vertexArray.get());

                        // ** replace IndexArray type osg::ArrayUInt by osg::ArrayUShort or osg::ArrayUBytes if possible
                        osgUtil::DrawElementTypeSimplifier dets;
                        dets.simplify(*(clonedGeom.get()));

                        osg::Vec4Array *colours = dynamic_cast<osg::Vec4Array *>(clonedGeom->getColorArray());
                        if (!colours)
                        {
                            colours = new osg::Vec4Array(1);
                            (*colours)[0].set(1.0f, 1.0f, 1.0f, 1.0f);
                            clonedGeom->setColorArray(colours);
                            clonedGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
                        }

                        // ** insert the geometry in scnene graph
                        clonedGeode->addDrawable(clonedGeom.get());

                        osgUtil::SmoothingVisitor sv;
                        sv.smooth(*clonedGeom); // this will replace the normal vector with a new one
                    }
                }
            }

            osg::notify(osg::WARN) << clonedGeode->getNumDrawables() << " added on " << numDrawables << "." << std::endl;
            addAndTraverse(node, clonedGeode.get());

            if (pushedAttributeNames)
                popAttributeNames();
        }
    }

    bool overlap(double xMin, double yMin, double xMax, double yMax)
    {
        if ((_dt._extents.xMin() > xMax) || (xMin > _dt._extents.xMax()) || (_dt._extents.yMin() > yMax) || (yMin > _dt._extents.yMax()))
            return false;
        else
            return true;
    }

    bool overlap(const osg::BoundingSphere &bs)
    {
        double xMin = bs.center().x() - bs.radius();
        double yMin = bs.center().y() - bs.radius();
        double xMax = bs.center().x() + bs.radius();
        double yMax = bs.center().y() + bs.radius();
        return overlap(xMin, yMin, xMax, yMax);
    }

    osg::Node *getCreatedModel() { return _createdModel.get(); }

private:
    osg::HeightField &_hf;
    DestinationTile &_dt;

    std::string _TypeAttributeName;
    std::string _HeightAttributeName;

    osg::fast_back_stack<osg::Node *> _nodeStack;
    osg::ref_ptr<osg::Node> _createdModel;

    typedef std::list<std::string> StringStack;
    StringStack _typeAttributeNameStack;
    StringStack _heightAttributeNameStack;
};

bool ShapeFilePlacer::place(DestinationTile &destinationTile, osg::Node *model)
{
    log(osg::NOTICE, "ShapeFilePlacer::place(%s)", model->getName().c_str());

    osg::HeightField *hf = destinationTile.getSourceHeightField();
    if (hf == NULL)
        return true;

    ShapeFileOverlapingHeightFieldPlacer shapePlacer(destinationTile, *hf);
    model->accept(shapePlacer);

#if 0
    osg::Material * mat = new osg::Material;
    mat->setDiffuse(osg::Material::FRONT, osg::Vec4f(1.0f,1.0f,1.0f,1.0f));
    model->getOrCreateStateSet()->setAttributeAndModes(mat, osg::StateAttribute::ON);
#endif

    if (shapePlacer.getCreatedModel())
        destinationTile.addNodeToScene(shapePlacer.getCreatedModel(), true);

    return true;
}
