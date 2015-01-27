/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coPreviewCube.h"
#include <cover/coVRPluginSupport.h>
#include <virvo/vvtoolshed.h>

#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#include <osg/CullFace>
#include <osg/Geode>
#include <osg/Material>

using namespace osg;
using namespace vrui;
using namespace opencover;

coPreviewCube::coPreviewCube()
{
    color = NULL;
    coord = NULL;
    normal = NULL;
    vertices = NULL;
    angle = 0.0f;
    myDCS = new OSGVruiTransformNode(new MatrixTransform());
    myDCS->getNodePtr()->asGroup()->addChild(createNode().get());
    setPos(55, -10);
    setSize(7);
    setOrientation(0.0);
    start = cover->frameTime();
    scale = 1.0; // XXX: wieso wird das sonst nirgends gesetzt?
}

coPreviewCube::~coPreviewCube()
{
    myDCS->removeAllChildren();
    myDCS->removeAllParents();
    delete myDCS;
}

void coPreviewCube::setPos(float x, float y, float)
{
    myX = x;
    myY = y;
}

void coPreviewCube::setSize(float s)
{
    xScaleFactor = yScaleFactor = zScaleFactor = s;
}

void coPreviewCube::setSize(float xs, float ys, float zs)
{
    xScaleFactor = xs;
    yScaleFactor = ys;
    zScaleFactor = zs;
}

// Sets the cube orientation.
// angle = rotational angle in degrees [0..360]
void coPreviewCube::setOrientation(float angle)
{
    Matrix animMat, rotZ, rotY, result, scaleMat, transMat;
    float angle2;

    angle2 = 180.0f * atanf(sqrt(2.0)) / TS_PI;
    rotY.makeRotate(angle2, 0, 1, 0);
    rotZ.makeRotate(45, 0, 0, 1);
    animMat.makeRotate(osg::inDegrees(angle), 0, 0, 1);
    scaleMat.makeScale(xScaleFactor, yScaleFactor, zScaleFactor);
    transMat.makeTranslate(myX + (xScaleFactor / 2), myY + (yScaleFactor / 2), 0);
    result = rotZ * rotY * animMat * scaleMat * transMat;
    dynamic_cast<MatrixTransform *>(myDCS->getNodePtr())->setMatrix(result);
}

vruiTransformNode *coPreviewCube::getDCS()
{
    return myDCS;
}

void coPreviewCube::createLists()
{

    color = new Vec4Array(1);
    coord = new Vec3Array(8);
    normal = new Vec3Array(6 * 4);

    ushort *verticesArray = new ushort[6 * 4];

    (*coord)[0].set(-1.0, 1.0, -1.0); // top left back
    (*coord)[1].set(-1.0, 1.0, 1.0); // top left front
    (*coord)[2].set(1.0, 1.0, 1.0); // top right front
    (*coord)[3].set(1.0, 1.0, -1.0); // top right back
    (*coord)[4].set(-1.0, -1.0, -1.0);
    (*coord)[5].set(-1.0, -1.0, 1.0);
    (*coord)[6].set(1.0, -1.0, 1.0);
    (*coord)[7].set(1.0, -1.0, -1.0);

    (*color)[0].set(1.0f, 1.0f, 1.0f, 1.0f); // default color is white

    for (int i = 0; i < 4; ++i)
    {
        (*normal)[0 * 4 + i].set(0.0, 1.0, 0.0);
        (*normal)[1 * 4 + i].set(0.0, 0.0, -1.0);
        (*normal)[2 * 4 + i].set(-1.0, 0.0, 0.0);
        (*normal)[3 * 4 + i].set(0.0, 0.0, 1.0);
        (*normal)[4 * 4 + i].set(1.0, 0.0, 0.0);
        (*normal)[5 * 4 + i].set(0.0, -1.0, 0.0);
    }

    verticesArray[0] = 0;
    verticesArray[1] = 1;
    verticesArray[2] = 2;
    verticesArray[3] = 3;
    verticesArray[4] = 0;
    verticesArray[5] = 3;
    verticesArray[6] = 7;
    verticesArray[7] = 4;
    verticesArray[8] = 0;
    verticesArray[9] = 4;
    verticesArray[10] = 5;
    verticesArray[11] = 1;
    verticesArray[12] = 1;
    verticesArray[13] = 5;
    verticesArray[14] = 6;
    verticesArray[15] = 2;
    verticesArray[16] = 2;
    verticesArray[17] = 6;
    verticesArray[18] = 7;
    verticesArray[19] = 3;
    verticesArray[20] = 4;
    verticesArray[21] = 7;
    verticesArray[22] = 6;
    verticesArray[23] = 5;

    vertices = new DrawElementsUShort(PrimitiveSet::QUADS, 24, verticesArray);

    delete[] verticesArray;

    ref_ptr<osg::Material> mtl = new osg::Material();
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.1, 0.1, 0.1, 1.0));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(0.6, 0.6, 0.6, 1.0));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 80.0f);

    normalGeostate = new StateSet();
    normalGeostate->setGlobalDefaults();

    normalGeostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    normalGeostate->setAttributeAndModes(mtl.get(), StateAttribute::ON);
    normalGeostate->setMode(GL_BLEND, StateAttribute::ON);
    normalGeostate->setMode(GL_LIGHTING, StateAttribute::ON);
}

ref_ptr<Node> coPreviewCube::createNode()
{

    cubeGeo = new Geometry();

    createLists();

    cubeGeo->setColorArray(color.get());
    cubeGeo->setColorBinding(Geometry::BIND_OVERALL);
    cubeGeo->setVertexArray(coord.get());
    cubeGeo->addPrimitiveSet(vertices.get());
    cubeGeo->setNormalArray(normal.get());
    cubeGeo->setNormalBinding(Geometry::BIND_PER_VERTEX);

    ref_ptr<Geode> geode = new Geode();
    geode->setStateSet(normalGeostate.get());
    geode->setName("PreviewCube");
    geode->addDrawable(cubeGeo.get());
    return geode.get();
}

// Set color values of preview cube faces. The HSV color model is used.
// h,s,v,a: all values need to be in range [0..1]
void coPreviewCube::setHSVA(float hue, float sat, float val, float alpha)
{
    float r, g, b;

    h = hue;
    s = sat;
    v = val;
    a = alpha;
    vvToolshed::HSBtoRGB(h, s, v, &r, &g, &b);
    (*color)[0].set(r, g, b, a);

    cubeGeo->dirtyDisplayList();
}

void coPreviewCube::setHS(float hue, float sat)
{
    setHSVA(hue, sat, v, a);
}

void coPreviewCube::setAlpha(float alpha)
{
    setHSVA(h, s, v, alpha);
}

void coPreviewCube::setBrightness(float bri)
{
    setHSVA(h, s, bri, a);
}

void coPreviewCube::update()
{
    const float DEG_PER_SEC = 45.0f; // cube rotation in degrees per second
    double stop;
    float diff; // time difference in seconds
    stop = cover->frameTime();
    diff = stop - start;
    start = stop;
    if (diff < 360.0f) // first pass is undefined
    {
        angle += (diff * DEG_PER_SEC);
        if (angle > 360.0)
            angle -= 360.0;
        setOrientation(angle);
    }
}
