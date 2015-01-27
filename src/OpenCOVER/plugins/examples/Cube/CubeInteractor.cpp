/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include "CubeInteractor.h"

#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>

using namespace osg;

CubeInteractor::CubeInteractor(coInteraction::InteractionType type, const char *name, coInteraction::InteractionPriority priority = Medium)
    : coTrackerButtonInteraction(type, name, priority)
{
    // create nodes
    root = new Group();
    // transformation and scale matrix
    DCS = new PositionAttitudeTransform();
    // geometry node
    cubeGeode = new Geode();
    // geometry containing the wireframe cube
    Geometry *cubeGeometry = createWireframeUnitCube();
    // stateset containing the material and line width
    StateSet *cubeState = createWireframeCubeMaterial();

    // create scenegraph
    cover->getObjectsRoot()->addChild(root.get());
    root->addChild(DCS);
    DCS->addChild(cubeGeode);
    // adding cube
    cubeGeode->addDrawable(cubeGeometry);
    // setting the material and line width state of geode
    cubeGeode->setStateSet(cubeState);
}

CubeInteractor::~CubeInteractor()
{
    // delete nodes and scenegraph
    if (root->getNumParents())
    {
        root->getParent(0)->removeChild(root.get());
    }
}

void
CubeInteractor::setSize(float s)
{
    // set scale component of matrix
    DCS->setScale(Vec3(s, s, s));
}

void
CubeInteractor::setCenter(Vec3 pos)
{
    // set center position of transforming matrix
    DCS->setPosition(pos);
}

Vec3
CubeInteractor::getCenter()
{
    // get center position of transforming matrix
    return DCS->getPosition();
}

void
CubeInteractor::hide()
{
    // removing scenegraph
    cover->getObjectsRoot()->removeChild(root.get());
}

void
CubeInteractor::show()
{
    // adding scenegraph
    cover->getObjectsRoot()->addChild(root.get());
}

Geometry *
CubeInteractor::createWireframeUnitCube()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCubeInteractor::createWireframeUnitCube\n");

    Geometry *cube = new Geometry();

    // create the coordinates of the cube

    //      5.......6
    //    .       . .
    //  4.......7   .
    //  .       .   .
    //  .   1   .   2
    //  .       . .
    //  0.......3

    Vec3Array *cubeVertices = new Vec3Array;
    cubeVertices->push_back(Vec3(-0.5, -0.5, -0.5));
    cubeVertices->push_back(Vec3(-0.5, 0.5, -0.5));
    cubeVertices->push_back(Vec3(0.5, 0.5, -0.5));
    cubeVertices->push_back(Vec3(0.5, -0.5, -0.5));

    cubeVertices->push_back(Vec3(-0.5, -0.5, 0.5));
    cubeVertices->push_back(Vec3(-0.5, 0.5, 0.5));
    cubeVertices->push_back(Vec3(0.5, 0.5, 0.5));
    cubeVertices->push_back(Vec3(0.5, -0.5, 0.5));

    // add the coordinates to the geometry object
    cube->setVertexArray(cubeVertices);

    // create the lines
    DrawElementsUInt *wireframe = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    // line 0-3
    wireframe->push_back(0);
    wireframe->push_back(3);
    // line 3-7
    wireframe->push_back(3);
    wireframe->push_back(7);
    // line 7-4
    wireframe->push_back(7);
    wireframe->push_back(4);
    // line 4-0
    wireframe->push_back(4);
    wireframe->push_back(0);
    // line 1-2
    wireframe->push_back(1);
    wireframe->push_back(2);
    // line 2-6
    wireframe->push_back(2);
    wireframe->push_back(6);
    // line 6-5
    wireframe->push_back(6);
    wireframe->push_back(5);
    // line 5-1
    wireframe->push_back(5);
    wireframe->push_back(1);
    // line 4-5
    wireframe->push_back(4);
    wireframe->push_back(5);
    // line 7-6
    wireframe->push_back(7);
    wireframe->push_back(6);
    // line 0-1
    wireframe->push_back(0);
    wireframe->push_back(1);
    // line 3-2
    wireframe->push_back(3);
    wireframe->push_back(2);

    // add the lines to the geometry
    cube->addPrimitiveSet(wireframe);

    // create the green wireframe color
    Vec4Array *color = new Vec4Array;
    color->push_back(Vec4(0.0, 0.75, 0.0, 1.0));

    // add color to geometry
    cube->setColorArray(color);
    // specifying that color should be added to the whole cube
    cube->setColorBinding(Geometry::BIND_PER_PRIMITIVE_SET);

    return cube;
}

StateSet *
CubeInteractor::createWireframeCubeMaterial()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCubeInteractor::createWireframeCubeMaterial\n");

    // creating the material properties
    Material *mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    mtl->setShininess(Material::FRONT_AND_BACK, 16.0f);

    //create a new StateSet
    StateSet *set = new StateSet();
    //modify the attributes of the Geode-container
    //add color
    set->setTextureAttributeAndModes(0, mtl, StateAttribute::ON);
    //define line width
    set->setAttribute(new LineWidth(3.5), StateAttribute::ON);

    return set;
}

void
CubeInteractor::startInteraction()
{
    Matrix initHandMat;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nCubeInteractor::startMove\n");

    // store hand mat
    initHandMat = cover->getPointerMat();
    initHandPos = (&initHandMat)->getTrans();
}

void
CubeInteractor::stopInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCubeInteractor::stopMove\n");
}

void
CubeInteractor::doInteraction()
{
    Vec3 relPos_o;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nCubeInteractor::move\n");

    Matrix currHandMat, w_to_o;
    Vec3 currHandPos, currHandPos_o, initHandPos_o;

    // get the current hand position
    currHandMat = cover->getPointerMat();
    currHandPos = (&currHandMat)->getTrans();

    // transform the hand matrix into object coordinates
    w_to_o = cover->getInvBaseMat();

    Vec4 pos, v;

    pos = Vec4(currHandPos, 1.0f);
    v = pos * w_to_o;
    v = v / v.w();
    currHandPos_o = Vec3(v.x(), v.y(), v.z());

    pos = Vec4(initHandPos, 1.0f);
    v = pos * w_to_o;
    v = v / v.w();
    initHandPos_o = Vec3(v.x(), v.y(), v.z());

    // compute the relative movement
    relPos_o = currHandPos_o - initHandPos_o;

    // apply it to the cube
    setCenter(DCS->getPosition() + relPos_o);

    // store the current hand position
    initHandPos = currHandPos;
}
