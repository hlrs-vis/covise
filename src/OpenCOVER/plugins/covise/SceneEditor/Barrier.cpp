/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneUtils.h"
#include "Barrier.h"
#include "Room.h"
#include "Window.h"
#include "Behaviors/TransformBehavior.h"
#include "Events/PostInteractionEvent.h"

#include <osg/Material>
#include <osg/LightModel>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/CullFace>
#include <osg/StateAttribute>
#include <osgDB/ReadFile>
#include <osgUtil/Tessellator>
#include <osg/LineWidth>

#include <cover/VRSceneGraph.h>

// constructor and destructor
// ---------------------------------------------
Barrier::Barrier(Room *room, std::string name, float width, float height, Barrier::Alignment al, osg::Vec3 p)
    : _width(width)
    , _height(height)
    , _alignment(al)
    , _position(p)
    , _room(room)
{
    _startPos = p;

    // create geometry
    // node to transform
    _transformNode = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeTranslate(p);
    _transformNode->setMatrix(m);

    // create array for color, coords and normal
    _coordArray = new osg::Vec3Array(4);
    _normalArray = new osg::Vec3Array(4);
    _texcoordRegular = new osg::Vec2Array(4);
    _texcoordWallpos = new osg::Vec2Array(4);

    // create coords and normals
    updateGeometry();
    updateNormal();

    // create geometry
    _geometry = new osg::Geometry();
    _geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _geometry->setVertexArray(_coordArray.get());
    _geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
    _geometry->setNormalArray(_normalArray.get());
    _geometry->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    _geometry->setUseDisplayList(false);
    _geometry->setTexCoordArray(0, _texcoordRegular);
    _geometry->setTexCoordArray(1, _texcoordWallpos);

    // stateset for culling back
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.0);
    osg::CullFace *cullFace = new osg::CullFace();
    cullFace->setMode(osg::CullFace::BACK);

    osg::LightModel *defaultLm = new osg::LightModel();
    defaultLm->setTwoSided(true);
    defaultLm->setLocalViewer(true);
    defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);

    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::OFF);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(.5f, .5f, .5f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);

    osg::StateSet *stateSet = _geometry->getOrCreateStateSet();
    stateSet->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    stateSet->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    //stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);
    stateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);

    _transformNode->setStateSet(stateSet);

    // shadows / borders
    _geometryNode = new osg::Geode();
    _geometryNode->setName(name);
    _geometryNode->addDrawable(_geometry.get());

    initGrid();

    _transformNode->addChild(_geometryNode.get());
    _transformNode->addChild(_gridGroup.get());

    // add to Room
    ((osg::MatrixTransform *)(_room->getGeometryNode()))->addChild(_transformNode.get());
}

Barrier::~Barrier()
{
}

// setter and getter
// ---------------------------------------------
float Barrier::getWidth()
{
    return _width;
}

float Barrier::getHeight()
{
    return _height;
}

osg::Vec3 Barrier::getNormal()
{
    return _normal;
}

osg::Vec3 Barrier::getPosition()
{
    return _position;
}

Barrier::Alignment Barrier::getAlignment()
{
    return _alignment;
}

void Barrier::setSize(float w, float h, float l)
{
    if (_alignment == TOP)
    {
        _height = l;
        _width = w;
        setPosition(osg::Vec3(0.0f, 0.0f, h));
    }
    else if (_alignment == BOTTOM)
    {
        _height = l;
        _width = w;
        setPosition(osg::Vec3(0.0f, 0.0f, 0.0f));
    }
    else if (_alignment == LEFT)
    {
        _width = l;
        _height = h;
        setPosition(osg::Vec3(-w / 2.0f, 0.0f, h / 2.0f));
    }
    else if (_alignment == RIGHT)
    {
        _width = l;
        _height = h;
        setPosition(osg::Vec3(w / 2.0f, 0.0f, h / 2.0f));
    }
    else if (_alignment == BACK)
    {
        _width = w;
        _height = h;
        setPosition(osg::Vec3(0.0f, l / 2.0f, h / 2.0f));
    }
    else if (_alignment == FRONT)
    {
        _height = h;
        _width = w;
        setPosition(osg::Vec3(0.0, -l / 2.0f, h / 2.0f));
    }

    // change geometry
    updateGeometry();
    updateGrid();
}

void Barrier::setPosition(osg::Vec3 v)
{
    _position = v;
    // change geometry
    osg::Matrix m;
    m.makeTranslate(v);
    _transformNode->setMatrix(m);
}

void Barrier::setAlignment(Alignment al)
{
    _alignment = al;
    // change geometry
    updateNormal();
    updateGeometry();
    updateGrid();
}

osg::MatrixTransform *Barrier::getNode()
{
    return _transformNode.get();
}

Room *Barrier::getRoom()
{
    return _room;
}

// ----------------------------------------
void Barrier::updateGeometry()
{
    float w05 = _width * 0.5f;
    float h05 = _height * 0.5f;

    float r = _width / 2000.0f;
    float s = _height / 2000.0f;

    if (_alignment == FRONT)
    {
        (*_coordArray)[0].set(-w05, 0.0f, -h05);
        (*_coordArray)[1].set(-w05, 0.0f, h05);
        (*_coordArray)[2].set(w05, 0.0f, h05);
        (*_coordArray)[3].set(w05, 0.0f, -h05);
        (*_texcoordRegular)[0].set(-r, -s);
        (*_texcoordRegular)[1].set(-r, s);
        (*_texcoordRegular)[2].set(r, s);
        (*_texcoordRegular)[3].set(r, -s);
        (*_texcoordWallpos)[0].set(-0.5f, -0.5f);
        (*_texcoordWallpos)[1].set(-0.5f, 0.5f);
        (*_texcoordWallpos)[2].set(0.5f, 0.5f);
        (*_texcoordWallpos)[3].set(0.5f, -0.5f);
    }
    else if (_alignment == BACK)
    {
        (*_coordArray)[0].set(-w05, 0.0f, -h05);
        (*_coordArray)[1].set(w05, 0.0f, -h05);
        (*_coordArray)[2].set(w05, 0.0f, h05);
        (*_coordArray)[3].set(-w05, 0.0f, h05);
        (*_texcoordRegular)[0].set(-r, -s);
        (*_texcoordRegular)[1].set(r, -s);
        (*_texcoordRegular)[2].set(r, s);
        (*_texcoordRegular)[3].set(-r, s);
        (*_texcoordWallpos)[0].set(-0.5f, -0.5f);
        (*_texcoordWallpos)[1].set(0.5f, -0.5f);
        (*_texcoordWallpos)[2].set(0.5f, 0.5f);
        (*_texcoordWallpos)[3].set(-0.5f, 0.5f);
    }
    else if (_alignment == LEFT)
    {
        (*_coordArray)[0].set(0.0f, -w05, -h05);
        (*_coordArray)[1].set(0.0f, w05, -h05);
        (*_coordArray)[2].set(0.0f, w05, h05);
        (*_coordArray)[3].set(0.0f, -w05, h05);
        (*_texcoordRegular)[0].set(-r, -s);
        (*_texcoordRegular)[1].set(r, -s);
        (*_texcoordRegular)[2].set(r, s);
        (*_texcoordRegular)[3].set(-r, s);
        (*_texcoordWallpos)[0].set(-0.5f, -0.5f);
        (*_texcoordWallpos)[1].set(0.5f, -0.5f);
        (*_texcoordWallpos)[2].set(0.5f, 0.5f);
        (*_texcoordWallpos)[3].set(-0.5f, 0.5f);
    }
    else if (_alignment == RIGHT)
    {
        (*_coordArray)[0].set(0.0f, -w05, -h05);
        (*_coordArray)[1].set(0.0f, -w05, h05);
        (*_coordArray)[2].set(0.0f, w05, h05);
        (*_coordArray)[3].set(0.0f, w05, -h05);
        (*_texcoordRegular)[0].set(-r, -s);
        (*_texcoordRegular)[1].set(-r, s);
        (*_texcoordRegular)[2].set(r, s);
        (*_texcoordRegular)[3].set(r, -s);
        (*_texcoordWallpos)[0].set(-0.5f, -0.5f);
        (*_texcoordWallpos)[1].set(-0.5f, 0.5f);
        (*_texcoordWallpos)[2].set(0.5f, 0.5f);
        (*_texcoordWallpos)[3].set(0.5f, -0.5f);
    }
    else if (_alignment == TOP)
    {
        (*_coordArray)[0].set(-w05, -h05, 0.0f);
        (*_coordArray)[1].set(-w05, h05, 0.0f);
        (*_coordArray)[2].set(w05, h05, 0.0f);
        (*_coordArray)[3].set(w05, -h05, 0.0f);
        (*_texcoordRegular)[0].set(-r, s);
        (*_texcoordRegular)[1].set(-r, -s);
        (*_texcoordRegular)[2].set(r, -s);
        (*_texcoordRegular)[3].set(r, s);
        (*_texcoordWallpos)[0].set(-0.5f, -0.5f);
        (*_texcoordWallpos)[1].set(-0.5f, 0.5f);
        (*_texcoordWallpos)[2].set(0.5f, 0.5f);
        (*_texcoordWallpos)[3].set(0.5f, -0.5f);
    }
    else if (_alignment == BOTTOM)
    {
        (*_coordArray)[0].set(-w05, -h05, 0.0f);
        (*_coordArray)[1].set(w05, -h05, 0.0f);
        (*_coordArray)[2].set(w05, h05, 0.0f);
        (*_coordArray)[3].set(-w05, h05, 0.0f);
        (*_texcoordRegular)[0].set(-r, -s);
        (*_texcoordRegular)[1].set(r, -s);
        (*_texcoordRegular)[2].set(r, s);
        (*_texcoordRegular)[3].set(-r, s);
        (*_texcoordWallpos)[0].set(-0.5f, -0.5f);
        (*_texcoordWallpos)[1].set(0.5f, -0.5f);
        (*_texcoordWallpos)[2].set(0.5f, 0.5f);
        (*_texcoordWallpos)[3].set(-0.5f, 0.5f);
    }

    if (_geometryNode)
    {
        _geometryNode->dirtyBound();
    }

    if (_geometry)
    {
        _geometry->setVertexArray(_coordArray.get());
    }
}

void Barrier::updateNormal()
{
    if (_alignment == FRONT)
        _normal = osg::Vec3(0, 1, 0);
    else if (_alignment == BACK)
        _normal = osg::Vec3(0, -1, 0);
    else if (_alignment == LEFT)
        _normal = osg::Vec3(1, 0, 0);
    else if (_alignment == RIGHT)
        _normal = osg::Vec3(-1, 0, 0);
    else if (_alignment == TOP)
        _normal = osg::Vec3(0, 0, -1);
    else if (_alignment == BOTTOM)
        _normal = osg::Vec3(0, 0, 1);

    // normal
    (*_normalArray)[0].set(_normal);
    (*_normalArray)[1].set(_normal);
    (*_normalArray)[2].set(_normal);
    (*_normalArray)[3].set(_normal);

    if (_geometryNode)
    {
        _geometry->setNormalArray(_normalArray.get());
        _geometryNode->dirtyBound();
    }
}

// ----------------------------------------
void Barrier::preFrame()
{
    float pv = SceneUtils::getPlaneVisibility(_position, _normal);
    if (pv < 0.0f)
    {
        _geometryNode->setNodeMask(opencover::Isect::Visible | opencover::Isect::Intersection | opencover::Isect::Pick);
    }
    else
    {
        _geometryNode->setNodeMask(opencover::Isect::Visible & (~opencover::Isect::Intersection) & (~opencover::Isect::Pick));
    }
}

/**
  * is called if the barrier needs to be repainted (e.g. window is added or removed)
**/
void Barrier::repaint()
{
    int numCoords = 4; // 4 coords for wall

    // find windows which belong to that wall
    std::list<Window *> wins;
    std::list<Window *>::iterator it;
    std::list<Window *> roomWindows = _room->getWindows();
    for (it = roomWindows.begin(); it != roomWindows.end(); it++)
    {
        // look for position of window
        osg::Vec3 pos = (*it)->getTranslate().getTrans();

        // check position of window against position of wall
        float diff = -1;
        osg::Vec3 posRoom = _room->getPosition();
        if (_alignment == FRONT || _alignment == BACK)
            diff = pos[1] - _position[1] - posRoom[1];
        else if (_alignment == LEFT || _alignment == RIGHT)
            diff = pos[0] - _position[0] - posRoom[0];
        else if (_alignment == TOP || _alignment == BOTTOM)
            diff = pos[2] - _position[2] - posRoom[2];

        // window is part of this wall
        if (diff < 0.1 && diff > -0.1)
        {
            wins.push_back(*it);
            numCoords += 4;
        }
    } // end of loop through all windows

    //create new coord array
    _coordArray = new osg::Vec3Array(numCoords);
    _texcoordRegular = new osg::Vec2Array(numCoords);
    _texcoordWallpos = new osg::Vec2Array(numCoords);
    // add coords of wall first
    updateGeometry();

    // add coords af windows
    int startPos = 4; // because of wall coords
    for (it = wins.begin(); it != wins.end(); it++)
    {
        float widthHalf = (*it)->getWidth() / 2.f;
        float heightHalf = (*it)->getHeight() / 2.f;
        // get position of window
        osg::Vec3 pos = osg::Vec3(0, 0, 0);

        pos = (*it)->getTranslate().getTrans();
        pos = pos - _position - _room->getPosition();

        //TODO rotate windows
        float wMin, wMax, hMin, hMax;
        float w05 = _width * 0.5f;
        float h05 = _height * 0.5f;

        // this is not correct, but I don't care because we have no images on barriers with windows
        (*_texcoordRegular)[startPos + 0].set(0.0f, 0.0f);
        (*_texcoordRegular)[startPos + 1].set(0.0f, 0.0f);
        (*_texcoordRegular)[startPos + 2].set(0.0f, 0.0f);
        (*_texcoordRegular)[startPos + 3].set(0.0f, 0.0f);

        if (_alignment == FRONT)
        {
            wMin = pos[0] - widthHalf;
            wMax = pos[0] + widthHalf;
            hMin = pos[2] - heightHalf;
            hMax = pos[2] + heightHalf;
            if (wMin < (-w05))
                wMin = (-w05);
            if (wMax > w05)
                wMax = w05;
            if (hMin < (-h05))
                hMin = (-h05);
            if (hMax > h05)
                hMax = h05;
            (*_coordArray)[startPos + 0].set(wMin, 0.0f, hMin);
            (*_coordArray)[startPos + 1].set(wMax, 0.0f, hMin);
            (*_coordArray)[startPos + 2].set(wMax, 0.0f, hMax);
            (*_coordArray)[startPos + 3].set(wMin, 0.0f, hMax);
            float sMin = (pos[0] - widthHalf) / _width;
            float sMax = (pos[0] + widthHalf) / _width;
            float tMin = (pos[2] - heightHalf) / _height;
            float tMax = (pos[2] + heightHalf) / _height;
            (*_texcoordWallpos)[startPos + 0].set(sMin, tMin);
            (*_texcoordWallpos)[startPos + 1].set(sMax, tMin);
            (*_texcoordWallpos)[startPos + 2].set(sMax, tMax);
            (*_texcoordWallpos)[startPos + 3].set(sMin, tMax);
        }
        else if (_alignment == BACK)
        {
            wMin = pos[0] - widthHalf;
            wMax = pos[0] + widthHalf;
            hMin = pos[2] - heightHalf;
            hMax = pos[2] + heightHalf;
            if (wMin < (-w05))
                wMin = (-w05);
            if (wMax > w05)
                wMax = w05;
            if (hMin < (-h05))
                hMin = (-h05);
            if (hMax > h05)
                hMax = h05;
            (*_coordArray)[startPos + 0].set(wMin, 0.0f, hMin);
            (*_coordArray)[startPos + 1].set(wMin, 0.0f, hMax);
            (*_coordArray)[startPos + 2].set(wMax, 0.0f, hMax);
            (*_coordArray)[startPos + 3].set(wMax, 0.0f, hMin);
            float sMin = (pos[0] - widthHalf) / _width;
            float sMax = (pos[0] + widthHalf) / _width;
            float tMin = (pos[2] - heightHalf) / _height;
            float tMax = (pos[2] + heightHalf) / _height;
            (*_texcoordWallpos)[startPos + 0].set(sMin, tMin);
            (*_texcoordWallpos)[startPos + 1].set(sMin, tMax);
            (*_texcoordWallpos)[startPos + 2].set(sMax, tMax);
            (*_texcoordWallpos)[startPos + 3].set(sMax, tMin);
        }
        else if (_alignment == LEFT)
        {
            wMin = pos[1] - widthHalf;
            wMax = pos[1] + widthHalf;
            hMin = pos[2] - heightHalf;
            hMax = pos[2] + heightHalf;
            if (wMin < (-w05))
                wMin = (-w05);
            if (wMax > w05)
                wMax = w05;
            if (hMin < (-h05))
                hMin = (-h05);
            if (hMax > h05)
                hMax = h05;
            (*_coordArray)[startPos + 0].set(0.0f, wMin, hMin);
            (*_coordArray)[startPos + 1].set(0.0f, wMin, hMax);
            (*_coordArray)[startPos + 2].set(0.0f, wMax, hMax);
            (*_coordArray)[startPos + 3].set(0.0f, wMax, hMin);
            float sMin = (pos[1] - widthHalf) / _width;
            float sMax = (pos[1] + widthHalf) / _width;
            float tMin = (pos[2] - heightHalf) / _height;
            float tMax = (pos[2] + heightHalf) / _height;
            (*_texcoordWallpos)[startPos + 0].set(sMin, tMin);
            (*_texcoordWallpos)[startPos + 1].set(sMin, tMax);
            (*_texcoordWallpos)[startPos + 2].set(sMax, tMax);
            (*_texcoordWallpos)[startPos + 3].set(sMax, tMin);
        }
        else if (_alignment == RIGHT)
        {
            wMin = pos[1] - widthHalf;
            wMax = pos[1] + widthHalf;
            hMin = pos[2] - heightHalf;
            hMax = pos[2] + heightHalf;
            if (wMin < (-w05))
                wMin = (-w05);
            if (wMax > w05)
                wMax = w05;
            if (hMin < (-h05))
                hMin = (-h05);
            if (hMax > h05)
                hMax = h05;
            (*_coordArray)[startPos + 0].set(0.0f, wMin, hMin);
            (*_coordArray)[startPos + 1].set(0.0f, wMax, hMin);
            (*_coordArray)[startPos + 2].set(0.0f, wMax, hMax);
            (*_coordArray)[startPos + 3].set(0.0f, wMin, hMax);
            float sMin = (pos[1] - widthHalf) / _width;
            float sMax = (pos[1] + widthHalf) / _width;
            float tMin = (pos[2] - heightHalf) / _height;
            float tMax = (pos[2] + heightHalf) / _height;
            (*_texcoordWallpos)[startPos + 0].set(sMin, tMin);
            (*_texcoordWallpos)[startPos + 1].set(sMax, tMin);
            (*_texcoordWallpos)[startPos + 2].set(sMax, tMax);
            (*_texcoordWallpos)[startPos + 3].set(sMin, tMax);
        }
        else if (_alignment == TOP)
        {
            wMin = pos[0] - widthHalf;
            wMax = pos[0] + widthHalf;
            hMin = pos[1] - heightHalf;
            hMax = pos[1] + heightHalf;
            if (wMin < (-w05))
                wMin = (-w05);
            if (wMax > w05)
                wMax = w05;
            if (hMin < (-h05))
                hMin = (-h05);
            if (hMax > h05)
                hMax = h05;
            (*_coordArray)[startPos + 0].set(wMin, hMin, 0.0f);
            (*_coordArray)[startPos + 1].set(wMax, hMin, 0.0f);
            (*_coordArray)[startPos + 2].set(wMax, hMax, 0.0f);
            (*_coordArray)[startPos + 3].set(wMin, hMax, 0.0f);
            float sMin = (pos[0] - widthHalf) / _width;
            float sMax = (pos[0] + widthHalf) / _width;
            float tMin = (pos[1] - heightHalf) / _height;
            float tMax = (pos[1] + heightHalf) / _height;
            (*_texcoordWallpos)[startPos + 0].set(sMin, tMin);
            (*_texcoordWallpos)[startPos + 1].set(wMax, hMin);
            (*_texcoordWallpos)[startPos + 2].set(sMax, tMax);
            (*_texcoordWallpos)[startPos + 3].set(wMin, tMax);
        }
        else if (_alignment == BOTTOM)
        {
            wMin = pos[0] - widthHalf;
            wMax = pos[0] + widthHalf;
            hMin = pos[1] - heightHalf;
            hMax = pos[1] + heightHalf;
            if (wMin < (-w05))
                wMin = (-w05);
            if (wMax > w05)
                wMax = w05;
            if (hMin < (-h05))
                hMin = (-h05);
            if (hMax > h05)
                hMax = h05;
            (*_coordArray)[startPos + 0].set(wMin, hMin, 0.0f);
            (*_coordArray)[startPos + 1].set(wMin, hMax, 0.0f);
            (*_coordArray)[startPos + 2].set(wMax, hMax, 0.0f);
            (*_coordArray)[startPos + 3].set(wMax, hMin, 0.0f);
            float sMin = (pos[0] - widthHalf) / _width;
            float sMax = (pos[0] + widthHalf) / _width;
            float tMin = (pos[1] - heightHalf) / _height;
            float tMax = (pos[1] + heightHalf) / _height;
            (*_texcoordWallpos)[startPos + 0].set(sMin, tMin);
            (*_texcoordWallpos)[startPos + 1].set(wMin, tMax);
            (*_texcoordWallpos)[startPos + 2].set(sMax, tMax);
            (*_texcoordWallpos)[startPos + 3].set(sMax, tMin);
        }
        startPos += 4;
    }

    // create primitive sets for wall and window
    int numWindows = wins.size();

    // clear all primitive sets
    int numPrimitiveSets = _geometry->getNumPrimitiveSets();
    _geometry->removePrimitiveSet(0, numPrimitiveSets);

    _geometry->setVertexArray(_coordArray.get());
    _geometry->setTexCoordArray(0, _texcoordRegular.get());
    _geometry->setTexCoordArray(1, _texcoordWallpos.get());
    _geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));

    for (int i = 1; i <= numWindows; i++)
    {
        _geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 4 * i, 4));
    }

    // tesselate for windows
    if (numWindows > 0)
        tessellate();
}

void Barrier::tessellate()
{
    if (_geometry)
    {
        // tesselate for windows
        osg::ref_ptr<osgUtil::Tessellator> tscx = new osgUtil::Tessellator;
        tscx->setTessellationType(osgUtil::Tessellator::TESS_TYPE_GEOMETRY);
        tscx->setBoundaryOnly(false);
        tscx->setWindingType(osgUtil::Tessellator::TESS_WINDING_POSITIVE);
        tscx->retessellatePolygons(*_geometry);
    }
}

void Barrier::initGrid()
{
    if (_alignment != TOP)
    {
        _gridGroup = NULL;
        return;
    }

    _gridGroup = new osg::Group();
    _gridGroup->setNodeMask(_gridGroup->getNodeMask() & ~opencover::Isect::Visible & ~opencover::Isect::Intersection & ~opencover::Isect::Pick);

    _gridGeode = new osg::Geode();
    _gridGroup->addChild(_gridGeode.get());

    _gridGeometry = new osg::Geometry();
    _gridGeode->addDrawable(_gridGeometry.get());

    _gridGeometry->setStateSet(opencover::VRSceneGraph::instance()->loadDefaultGeostate());
    _gridGeometry->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(1.0f), osg::StateAttribute::ON);
    osg::ref_ptr<osg::Vec4Array> gridColor = new osg::Vec4Array();
    gridColor->push_back(osg::Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    _gridGeometry->setColorArray(gridColor.get());
    _gridGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _gridGeometry->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _gridGeometry->setUseDisplayList(false);

    _gridVertices = new osg::Vec3Array();
    _gridGeometry->setVertexArray(_gridVertices.get());

    _gridCenterGeode = new osg::Geode();
    _gridGroup->addChild(_gridCenterGeode.get());

    _gridCenterGeometry = new osg::Geometry();
    _gridCenterGeode->addDrawable(_gridCenterGeometry.get());

    _gridCenterGeometry->setStateSet(opencover::VRSceneGraph::instance()->loadDefaultGeostate());
    _gridCenterGeometry->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(1.5f), osg::StateAttribute::ON);
    osg::ref_ptr<osg::Vec4Array> gridCenterColor = new osg::Vec4Array();
    gridCenterColor->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    _gridCenterGeometry->setColorArray(gridCenterColor.get());
    _gridCenterGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _gridCenterGeometry->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _gridCenterGeometry->setUseDisplayList(false);

    _gridCenterVertices = new osg::Vec3Array(4);
    _gridCenterGeometry->setVertexArray(_gridCenterVertices.get());
    _gridCenterGeometry->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 4));

    updateGrid();
}

void Barrier::updateGrid()
{
    if (!_gridGroup)
    {
        return;
    }

    _gridVertices->clear();
    if (_gridGeometry->getNumPrimitiveSets() > 0)
    {
        _gridGeometry->removePrimitiveSet(0, 1);
    }

    float offset = -20.0f;

    float current = 500.0f;
    while (current <= _width / 2.0f + 1.0f)
    {
        _gridVertices->push_back(osg::Vec3(current, -_height / 2.0f, offset));
        _gridVertices->push_back(osg::Vec3(current, _height / 2.0f, offset));
        _gridVertices->push_back(osg::Vec3(-current, -_height / 2.0f, offset));
        _gridVertices->push_back(osg::Vec3(-current, _height / 2.0f, offset));
        current += 500.0f;
    }

    current = 500.0f;
    while (current <= _height / 2.0f + 1.0f)
    {
        _gridVertices->push_back(osg::Vec3(-_width / 2.0f, current, offset));
        _gridVertices->push_back(osg::Vec3(_width / 2.0f, current, offset));
        _gridVertices->push_back(osg::Vec3(-_width / 2.0f, -current, offset));
        _gridVertices->push_back(osg::Vec3(_width / 2.0f, -current, offset));
        current += 500.0f;
    }

    _gridGeometry->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, _gridVertices->size()));

    _gridCenterVertices->clear();
    _gridCenterVertices->push_back(osg::Vec3(0.0f, -_height / 2.0f, offset));
    _gridCenterVertices->push_back(osg::Vec3(0.0f, _height / 2.0f, offset));
    _gridCenterVertices->push_back(osg::Vec3(-_width / 2.0f, 0.0f, offset));
    _gridCenterVertices->push_back(osg::Vec3(_width / 2.0f, 0.0f, offset));
}

void Barrier::setGridVisible(bool visible)
{
    if (!_gridGroup)
    {
        return;
    }

    if (visible)
    {
        _gridGroup->setNodeMask(_gridGroup->getNodeMask() | opencover::Isect::Visible);
    }
    else
    {
        _gridGroup->setNodeMask(_gridGroup->getNodeMask() & ~opencover::Isect::Visible);
    }
}
