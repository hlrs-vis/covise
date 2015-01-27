/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Shape.h"
#include "Room.h"
#include "Events/SetSizeEvent.h"
#include "Events/TransformChangedEvent.h"
#include "Events/SetAppearanceColorEvent.h"
#include "SceneUtils.h"

#include <cover/coVRPluginSupport.h>

#include <osg/CullFace>

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

Shape::Shape()
{
    _name = "";
    _type = SceneObjectTypes::SHAPE;
    _geometryType = GEOMETRY_UNSPECIFIED;
    _width = 1.0f;
    _height = 1.0f;
    _length = 1.0f;
    _automaticHeight = false;
    _group = NULL;
    _numSegments = 0;
    _wireframe = false;
}

Shape::~Shape()
{
    if (_numSegments > 0)
    {
        for (int i = 0; i < _numSegments; ++i)
        {
            delete _segments[i];
        }
        delete[] _segments;
    }
}

EventErrors::Type Shape::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::SET_SIZE_EVENT)
    {
        SetSizeEvent *sse = dynamic_cast<SetSizeEvent *>(e);
        setSize(sse->getWidth(), sse->getHeight(), sse->getLength());
    }
    else if (e->getType() == EventTypes::APPLY_MOUNT_RESTRICTIONS_EVENT) // the size of the room might have been changed
    {
        if (_automaticHeight)
        {
            Room *room = dynamic_cast<Room *>(getParent());
            if (room)
            {
                _height = room->getHeight();
                _updateGeometry();
                // Important: Don't send the TransformChangedEvent since it will cause a loop.
                //            We are already in the process of ApplyMountRestrictions so no event nescessary.
            }
        }
    }
    else if (e->getType() == EventTypes::TRANSFORM_CHANGED_EVENT)
    {
        _updateVisibility();
    }
    else if (e->getType() == EventTypes::PRE_FRAME_EVENT)
    {
        _updatePicking();
    }
    else if (e->getType() == EventTypes::SET_APPEARANCE_COLOR_EVENT)
    {
        SetAppearanceColorEvent *sace = dynamic_cast<SetAppearanceColorEvent *>(e);
        osg::Vec4 color = sace->getColor();
        _wireframe = (color[3] < 0.001f);
        _updateVisibility();
    }
    return SceneObject::receiveEvent(e);
}

void Shape::setSize(float width, float height, float length)
{
    _width = width;
    _height = height;
    _length = length;
    _updateGeometry();

    TransformChangedEvent tce;
    tce.setSender(this);
    receiveEvent(&tce);
}

void Shape::setWidth(float width)
{
    _width = width;
    _updateGeometry();

    TransformChangedEvent tce;
    tce.setSender(this);
    receiveEvent(&tce);
}

void Shape::setHeight(float height)
{
    _height = height;
    _updateGeometry();

    TransformChangedEvent tce;
    tce.setSender(this);
    receiveEvent(&tce);
}

void Shape::setLength(float length)
{
    _length = length;
    _updateGeometry();

    TransformChangedEvent tce;
    tce.setSender(this);
    receiveEvent(&tce);
}

float Shape::getWidth()
{
    return _width;
}

float Shape::getHeight()
{
    return _height;
}

float Shape::getLength()
{
    return _length;
}

void Shape::setGeometryType(Shape::GeometryType geometryType)
{
    _geometryType = geometryType;
}

Shape::GeometryType Shape::getGeometryType()
{
    return _geometryType;
}

void Shape::setAutomaticHeight(bool ah)
{
    _automaticHeight = ah;
}

// [similar to Room::getClosestWall]
// We ignore the shapes transformation but transform the queryPosition respectively.
// The result is a vector pointing outside
void Shape::getClosestSide(osg::Vec3 queryPosition, osg::Vec3 &result_normal, osg::Vec3 &result_position)
{
    std::vector<osg::Vec3> normals;
    std::vector<osg::Vec3> positions;
    std::vector<float> widths;
    if (_geometryType == GEOMETRY_CUBOID)
    {
        normals.push_back(osg::Vec3(1.0f, 0.0f, 0.0f));
        positions.push_back(osg::Vec3(_width / 2.0f, 0.0f, 0.0f));
        widths.push_back(_length);
        normals.push_back(osg::Vec3(-1.0f, 0.0f, 0.0f));
        positions.push_back(osg::Vec3(-_width / 2.0f, 0.0f, 0.0f));
        widths.push_back(_length);
        normals.push_back(osg::Vec3(0.0f, 1.0f, 0.0f));
        positions.push_back(osg::Vec3(0.0f, _length / 2.0f, 0.0f));
        widths.push_back(_width);
        normals.push_back(osg::Vec3(0.0f, -1.0f, 0.0f));
        positions.push_back(osg::Vec3(0.0f, -_length / 2.0f, 0.0f));
        widths.push_back(_width);
    }
    else if (_geometryType == GEOMETRY_PRISM)
    {
        normals.push_back(osg::Vec3(-1.0f, 0.0f, 0.0f));
        positions.push_back(osg::Vec3(-_width / 2.0f, 0.0f, 0.0f));
        widths.push_back(_length);
        normals.push_back(osg::Vec3(0.0f, -1.0f, 0.0f));
        positions.push_back(osg::Vec3(0.0f, -_length / 2.0f, 0.0f));
        widths.push_back(_width);
        osg::Vec3 n = osg::Vec3(_width, _length, 0.0f);
        n.normalize();
        normals.push_back(n);
        positions.push_back(osg::Vec3(0.0f, 0.0f, 0.0f));
        widths.push_back(sqrt(_length * _length + _width * _width));
    }
    else
    {
        return;
    }

    // transform the position
    queryPosition = queryPosition * osg::Matrix::inverse(getTranslate()) * osg::Matrix::inverse(getRotate());
    queryPosition[2] = 0.0f;

    int bestHit = -1;
    float bestQuery = 99999999.9f;
    for (int i = 0; i < normals.size(); ++i)
    {
        // length of projection of queryPosition onto normal (may be negative)
        float proj_query = (queryPosition - positions[i]) * normals[i];
        // length of projection of surface position onto normal
        float proj_position = positions[i] * normals[i];

        // factor = how much has the queryPosition to be scaled to be on the surface
        float f = 1.0f;
        if ((proj_position + proj_query) != 0.0f)
        {
            f = proj_position / (proj_position + proj_query);
        }

        // scale position
        osg::Vec3 scaledPosition = queryPosition * f;

        // check if we are inside the extent of the surface
        if ((positions[i] - scaledPosition).length() <= widths[i] / 2.0f)
        {
            // we want the surface we are closest to in order to avoid jumping on the opposite side
            if (fabs(proj_query) < fabs(bestQuery))
            {
                bestHit = i;
                bestQuery = proj_query;
            }
        }
    }

    if (bestHit > -1)
    {
        result_normal = normals[bestHit];
        result_position = positions[bestHit];
    }
}

void Shape::_updateGeometry()
{
    if (_geometryType == GEOMETRY_UNSPECIFIED)
    {
        return;
    }

    if (!_group)
    {
        _group = new osg::Group();
        setGeometryNode(_group);
        osg::StateSet *sset = _group->getOrCreateStateSet();
        osg::CullFace *cullFace = new osg::CullFace();
        cullFace->setMode(osg::CullFace::BACK);
        sset->setAttributeAndModes(cullFace, osg::StateAttribute::ON);

        if (_geometryType == GEOMETRY_CUBOID)
        {
            _numSegments = 6;
            _segments = new ShapeSegment *[_numSegments];
            for (int i = 0; i < _numSegments; ++i)
            {
                _segments[i] = new ShapeSegment(_group, 4);
                _segments[i]->geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
                _segments[i]->wireframeGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, 4));
            }
        }
        else if (_geometryType == GEOMETRY_PRISM)
        {
            _numSegments = 5;
            _segments = new ShapeSegment *[_numSegments];
            _segments[PRISM_TOP] = new ShapeSegment(_group, 3);
            _segments[PRISM_TOP]->geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 3));
            _segments[PRISM_TOP]->wireframeGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, 3));
            _segments[PRISM_BOTTOM] = new ShapeSegment(_group, 3);
            _segments[PRISM_BOTTOM]->geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 3));
            _segments[PRISM_BOTTOM]->wireframeGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, 3));
            _segments[PRISM_FRONT] = new ShapeSegment(_group, 4);
            _segments[PRISM_FRONT]->geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
            _segments[PRISM_FRONT]->wireframeGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, 4));
            _segments[PRISM_LEFT] = new ShapeSegment(_group, 4);
            _segments[PRISM_LEFT]->geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
            _segments[PRISM_LEFT]->wireframeGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, 4));
            _segments[PRISM_DIAGONAL] = new ShapeSegment(_group, 4);
            _segments[PRISM_DIAGONAL]->geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
            _segments[PRISM_DIAGONAL]->wireframeGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, 4));
        }
        else if (_geometryType == GEOMETRY_CYLINDER)
        {
            osg::Geode *geode = new osg::Geode();
            _group->addChild(geode);
            _cylinderDrawable = new osg::ShapeDrawable();
            geode->addDrawable(_cylinderDrawable);
            osg::Cylinder *cylinder = new osg::Cylinder(osg::Vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f);
            _cylinderDrawable->setShape(cylinder);
        }
    }

    if (_geometryType == GEOMETRY_CUBOID)
    {
        _segments[CUBOID_BOTTOM]->setCoord(0, osg::Vec3(_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_BOTTOM]->setCoord(1, osg::Vec3(_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_BOTTOM]->setCoord(2, osg::Vec3(-_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_BOTTOM]->setCoord(3, osg::Vec3(-_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_BOTTOM]->setNormals(osg::Vec3(0.0f, 0.0f, -1.0f));
        _segments[CUBOID_TOP]->setCoord(0, osg::Vec3(_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[CUBOID_TOP]->setCoord(1, osg::Vec3(-_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[CUBOID_TOP]->setCoord(2, osg::Vec3(-_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[CUBOID_TOP]->setCoord(3, osg::Vec3(_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[CUBOID_TOP]->setNormals(osg::Vec3(0.0f, 0.0f, 1.0f));
        _segments[CUBOID_FRONT]->setCoord(0, osg::Vec3(-_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_FRONT]->setCoord(1, osg::Vec3(_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_FRONT]->setCoord(2, osg::Vec3(_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[CUBOID_FRONT]->setCoord(3, osg::Vec3(-_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[CUBOID_FRONT]->setNormals(osg::Vec3(0.0f, -1.0f, 0.0f));
        _segments[CUBOID_BACK]->setCoord(0, osg::Vec3(-_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_BACK]->setCoord(1, osg::Vec3(-_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[CUBOID_BACK]->setCoord(2, osg::Vec3(_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[CUBOID_BACK]->setCoord(3, osg::Vec3(_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_BACK]->setNormals(osg::Vec3(0.0f, 1.0f, 0.0f));
        _segments[CUBOID_LEFT]->setCoord(0, osg::Vec3(-_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_LEFT]->setCoord(1, osg::Vec3(-_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[CUBOID_LEFT]->setCoord(2, osg::Vec3(-_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[CUBOID_LEFT]->setCoord(3, osg::Vec3(-_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_LEFT]->setNormals(osg::Vec3(-1.0f, 0.0f, 0.0f));
        _segments[CUBOID_RIGHT]->setCoord(0, osg::Vec3(_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_RIGHT]->setCoord(1, osg::Vec3(_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[CUBOID_RIGHT]->setCoord(2, osg::Vec3(_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[CUBOID_RIGHT]->setCoord(3, osg::Vec3(_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[CUBOID_RIGHT]->setNormals(osg::Vec3(1.0f, 0.0f, 0.0f));
    }
    else if (_geometryType == GEOMETRY_PRISM)
    {
        _segments[PRISM_BOTTOM]->setCoord(0, osg::Vec3(_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[PRISM_BOTTOM]->setCoord(1, osg::Vec3(-_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[PRISM_BOTTOM]->setCoord(2, osg::Vec3(-_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[PRISM_BOTTOM]->setNormals(osg::Vec3(0.0f, 0.0f, -1.0f));
        _segments[PRISM_TOP]->setCoord(0, osg::Vec3(-_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[PRISM_TOP]->setCoord(1, osg::Vec3(-_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[PRISM_TOP]->setCoord(2, osg::Vec3(_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[PRISM_TOP]->setNormals(osg::Vec3(0.0f, 0.0f, 1.0f));
        _segments[PRISM_FRONT]->setCoord(0, osg::Vec3(-_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[PRISM_FRONT]->setCoord(1, osg::Vec3(_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[PRISM_FRONT]->setCoord(2, osg::Vec3(_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[PRISM_FRONT]->setCoord(3, osg::Vec3(-_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[PRISM_FRONT]->setNormals(osg::Vec3(0.0f, -1.0f, 0.0f));
        _segments[PRISM_LEFT]->setCoord(0, osg::Vec3(-_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[PRISM_LEFT]->setCoord(1, osg::Vec3(-_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[PRISM_LEFT]->setCoord(2, osg::Vec3(-_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[PRISM_LEFT]->setCoord(3, osg::Vec3(-_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[PRISM_LEFT]->setNormals(osg::Vec3(-1.0f, 0.0f, 0.0f));
        _segments[PRISM_DIAGONAL]->setCoord(0, osg::Vec3(-_width / 2.0f, _length / 2.0f, -_height / 2.0f));
        _segments[PRISM_DIAGONAL]->setCoord(1, osg::Vec3(-_width / 2.0f, _length / 2.0f, _height / 2.0f));
        _segments[PRISM_DIAGONAL]->setCoord(2, osg::Vec3(_width / 2.0f, -_length / 2.0f, _height / 2.0f));
        _segments[PRISM_DIAGONAL]->setCoord(3, osg::Vec3(_width / 2.0f, -_length / 2.0f, -_height / 2.0f));
        _segments[PRISM_DIAGONAL]->setNormals(osg::Vec3(0.7071f, 0.7071f, 0.0f));
    }
    else if (_geometryType == GEOMETRY_CYLINDER)
    {
        osg::Cylinder *cylinder = dynamic_cast<osg::Cylinder *>(_cylinderDrawable->getShape());
        cylinder->setRadius(_width / 2.0f);
        cylinder->setHeight(_height);
    }
    else
    {
        return;
    }

    for (int i = 0; i < _numSegments; ++i)
    {
        _segments[i]->geometry->dirtyBound();
        _segments[i]->wireframeGeometry->dirtyBound();
    }
    _group->dirtyBound();

    _updateVisibility();
}

void Shape::_updateVisibility()
{
    Room *room = dynamic_cast<Room *>(getParent());
    if (!room)
    {
        return;
    }
    if ((_geometryType != GEOMETRY_CUBOID) && (_geometryType != GEOMETRY_PRISM))
    {
        return;
    }
    for (int i = 0; i < _numSegments; ++i)
    {
        ShapeSegment *s = _segments[i];
        osg::Vec3 pmin(9999999.9, 9999999.9, 9999999.9), pmax(-9999999.9, -9999999.9, -9999999.9);
        for (int j = 0; j < s->numVertices; ++j)
        {
            osg::Vec3 pos = s->getCoord(j);
            pos = pos * getRotate() * getTranslate();
            pmin[0] = MIN(pmin[0], pos[0]);
            pmin[1] = MIN(pmin[1], pos[1]);
            pmin[2] = MIN(pmin[2], pos[2]);
            pmax[0] = MAX(pmax[0], pos[0]);
            pmax[1] = MAX(pmax[1], pos[1]);
            pmax[2] = MAX(pmax[2], pos[2]);
        }
        if ((pmin[0] > room->getPosition()[0] - room->getWidth() / 2.0f - 0.1f)
            && (pmax[0] < room->getPosition()[0] - room->getWidth() / 2.0f + 0.1f))
        {
            s->geode->setNodeMask(s->geode->getNodeMask() & ~opencover::Isect::Visible);
        }
        else if ((pmin[0] > room->getPosition()[0] + room->getWidth() / 2.0f - 0.1f)
                 && (pmax[0] < room->getPosition()[0] + room->getWidth() / 2.0f + 0.1f))
        {
            s->geode->setNodeMask(s->geode->getNodeMask() & ~opencover::Isect::Visible);
        }
        else if ((pmin[1] > room->getPosition()[1] - room->getLength() / 2.0f - 0.1f)
                 && (pmax[1] < room->getPosition()[1] - room->getLength() / 2.0f + 0.1f))
        {
            s->geode->setNodeMask(s->geode->getNodeMask() & ~opencover::Isect::Visible);
        }
        else if ((pmin[1] > room->getPosition()[1] + room->getLength() / 2.0f - 0.1f)
                 && (pmax[1] < room->getPosition()[1] + room->getLength() / 2.0f + 0.1f))
        {
            s->geode->setNodeMask(s->geode->getNodeMask() & ~opencover::Isect::Visible);
        }
        else
        {
            s->geode->setNodeMask(s->geode->getNodeMask() | opencover::Isect::Visible);
        }
        if (_wireframe)
        {
            s->wireframeGeode->setNodeMask(s->geode->getNodeMask() | opencover::Isect::Visible);
        }
        else
        {
            s->wireframeGeode->setNodeMask(s->geode->getNodeMask() & ~opencover::Isect::Visible);
        }
    }
}

void Shape::_updatePicking()
{
    if ((_geometryType != GEOMETRY_CUBOID) && (_geometryType != GEOMETRY_PRISM))
    {
        return;
    }
    for (int i = 0; i < _numSegments; ++i)
    {
        ShapeSegment *s = _segments[i];
        osg::Vec3 p = s->getCoord(0);
        p = p * getRotate() * getTranslate();
        osg::Vec3 n = s->getNormal(0);
        n = n * getRotate();
        float pv = SceneUtils::getPlaneVisibility(p, n);
        if ((pv < 0.0f) && ((_geometryNode->getNodeMask() & opencover::Isect::Visible) != 0))
        {
            s->geode->setNodeMask(s->geode->getNodeMask() | opencover::Isect::Intersection | opencover::Isect::Pick);
        }
        else
        {
            s->geode->setNodeMask(s->geode->getNodeMask() & (~opencover::Isect::Intersection) & (~opencover::Isect::Pick));
        }
    }
}
