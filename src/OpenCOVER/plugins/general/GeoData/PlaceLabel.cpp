/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PlaceLabel.h"

#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Geometry>
#include <osg/Geode>
#include <osgText/Font>
#include <osgText/Text>
#include <osg/Array>
#include <osg/AlphaFunc>

using namespace opencover;

static osg::Vec4 WHITE(1, 1, 1, 1);
static osg::Vec4 GRAY(1, 1, 1, 0.5);

PlaceLabel::PlaceLabel(const std::string &value, const osg::Vec3 &position, osg::ref_ptr<osg::Group> parent, int size)
    : value(value)
    , position(position)
    , size(size)
{

    auto font = coVRFileManager::instance()->loadFont(NULL);

    auto color = size == 1 ? GRAY : WHITE;

    // unlighted geostate
    osg::Material *mtl = new osg::Material;
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, color);
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, color);
    mtl->setEmission(osg::Material::FRONT_AND_BACK, color);
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    // position dsc
    transform = new osg::MatrixTransform;
    parent->addChild(transform);

    // billboarding label
    billboard = new coBillboard();
    billboard->setNodeMask(billboard->getNodeMask() & ~Isect::Intersection & ~Isect::Pick);
    billboard->setMode(coBillboard::POINT_ROT_WORLD);
    billboard->setAxis(osg::Vec3(0, 1, 0));
    billboard->setNormal(osg::Vec3(0, 0, 1));
    transform->addChild(billboard);

    geode = new osg::Geode();
    geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    billboard->addChild(geode);

    text = new osgText::Text();
    text->setAlignment(osgText::Text::CENTER_BASE_LINE);
    text->setColor(color);
    text->setFont(font);
    text->setCharacterSize(fontSize);
    text->setText(value, osgText::String::ENCODING_UTF8);
    text->setPosition(osg::Vec3(0, lineLength, 0));
    geode->addDrawable(text);

    osg::StateSet *ss = geode->getOrCreateStateSet();
    ss->setAttributeAndModes(mtl, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::OFF);
    ss->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    ss->setAttributeAndModes(new osg::AlphaFunc(osg::AlphaFunc::GEQUAL, 0.1f), osg::StateAttribute::ON);

    auto vertices = new osg::Vec3Array();
    vertices->push_back(osg::Vec3(0.0, 0.0, 0.0));
    vertices->push_back(osg::Vec3(0.0, lineLength, 0.0));

    auto colors = new osg::Vec4Array();
    colors->push_back(color);

    lineGeometry = new osg::Geometry();
    lineGeometry->setColorArray(colors);
    lineGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    osg::DrawArrays *primitives = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2);
    lineGeometry->setVertexArray(vertices);
    lineGeometry->addPrimitiveSet(primitives);
    lineGeometry->setStateSet(ss);
    geode->addDrawable(lineGeometry);

    reposition();
}

void PlaceLabel::reposition()
{
    // Make cities bigger
    float s = size >= 3 ? 4.f : size >= 2 ? 2.f
                                          : 1.f;
    transform->setMatrix(osg::Matrix::scale(s, s, s) * osg::Matrix::translate(position));
}
