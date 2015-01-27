/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coPresetButtonGeometry.h"
#include <cover/coVRPluginSupport.h>

#include <osg/Geode>
#include <osg/Texture>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/ShapeDrawable>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#define STYLE_IN 1
#define STYLE_OUT 2
#define DETAIL_LEVEL 40 // number of triangles

using namespace osg;
using namespace vrui;
using namespace opencover;

coPresetButtonGeometry::coPresetButtonGeometry()
    : coButtonGeometry("")
    , normalNode(0)
    , pressedNode(0)
    , highlightNode(0)
    , pressedHighlightNode(0)
    , myDCS(0)
{
    createGeometry();
}

coPresetButtonGeometry::~coPresetButtonGeometry()
{
}

void coPresetButtonGeometry::createGeometry()
{
    if (normalNode == 0)
    {
        normalNode = createNode(false, false);
        pressedNode = createNode(true, false);
        highlightNode = createNode(false, true);
        pressedHighlightNode = createNode(true, true);

        ref_ptr<MatrixTransform> transformNode = new MatrixTransform();
        switchNode = new Switch();

        switchNode->addChild(normalNode.get());
        switchNode->addChild(pressedNode.get());
        switchNode->addChild(highlightNode.get());
        switchNode->addChild(pressedHighlightNode.get());

        transformNode->addChild(switchNode.get());

        myDCS = new OSGVruiTransformNode(transformNode.get());
    }
}

void coPresetButtonGeometry::resizeGeometry()
{
}

ref_ptr<StateSet> coPresetButtonGeometry::createGeoState(bool highlighted)
{
    ref_ptr<Material> material;
    ref_ptr<StateSet> geostate;

    // Create material:
    material = new Material();
    material->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    if (highlighted)
    {
        material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.0f, 1.0f));
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.0f, 1.0f));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.0f, 1.0f));
    }
    else
    {
        material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    }
    material->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(Material::FRONT_AND_BACK, 80.0f);

    // Create GeoState:
    geostate = new StateSet();
    geostate->setGlobalDefaults();
    geostate->setAttributeAndModes(material.get(), StateAttribute::ON);
    geostate->setMode(GL_LIGHTING, StateAttribute::ON);
    geostate->setMode(GL_BLEND, StateAttribute::OFF);
    return geostate.get();
}

ref_ptr<Node> coPresetButtonGeometry::createNode(bool pressed, bool highlighted)
{
    ref_ptr<ShapeDrawable> geoset;
    ref_ptr<Geode> geode;
    ref_ptr<MatrixTransform> dcs;
    ref_ptr<StateSet> geostate;

    geostate = createGeoState(highlighted);

    // Create GeoSet:
    ref_ptr<Cylinder> cylinder = new Cylinder(Vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f);
    ref_ptr<TessellationHints> th = new TessellationHints();

    th->setTargetNumFaces(DETAIL_LEVEL);
    th->setCreateFrontFace(true);
    th->setCreateBackFace(true);
    th->setCreateNormals(true);
    th->setCreateTop(true);
    th->setCreateBottom(true);
    th->setCreateBody(true);
    th->setTessellationMode(TessellationHints::USE_TARGET_NUM_FACES);

    geoset = new ShapeDrawable(cylinder.get(), th.get());

    // Create Geode:
    geode = new Geode();
    geode->setStateSet(geostate.get());
    geode->setName("Preset Button");
    geode->addDrawable(geoset.get());

    // Create node:
    dcs = new MatrixTransform();
    dcs->addChild(geode.get());

    Matrix m;
    Matrix sm;
    m.makeIdentity();

    if (pressed)
    {
        m.setTrans(2.5, 2.5, 2);
        sm.makeScale(5, 5, 2);
    }
    else
    {
        m.setTrans(2.5, 2.5, 5);
        sm.makeScale(5, 5, 5);
    }

    m *= sm;
    dcs->setMatrix(m);

    return dcs.get();
}

vruiTransformNode *coPresetButtonGeometry::getDCS()
{
    return myDCS;
}

void coPresetButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{
    switchNode->setSingleChildOn(active);
}
