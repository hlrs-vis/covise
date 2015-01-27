/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "primitive.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>

#include "parmblock.h"
#include "DEMOAPlugin.h"
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Shape>
#include <cover/coVRFileManager.h>
#include <osgDB/ReadFile>

#ifdef WIN32
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#endif

static const double DEG = 5.7295779513082320875e+1;
template <typename T>
inline T SQR(const T &a)
{
    return a * a;
}

D_Primitive::D_Primitive(parmblock &block)
    : bodyid_val()
    , forceid_val()
{
    GLfloat zero[4] = { 0.0, 0.0, 0.0, 0.0 };

    myname = block.name();
    myparentname = block.getstring("Parent");
    type_val = block.getstring("Type");
    triadfrom[0] = 0.0;
    triadfrom[1] = 0.0;
    triadfrom[2] = 0.0;
    triadto[0] = 0.0;
    triadto[1] = 0.0;
    triadto[2] = 0.0;

    float tmpVec[4];
    block.getvector(tmpVec, 3, "Position", zero);
    shift.set(tmpVec[0], tmpVec[1], tmpVec[2]);
    block.getvector(tmpVec, 3, "Cardan", zero);
    rotat.set(tmpVec[0], tmpVec[1], tmpVec[2]);
    block.getvector(tmpVec, 4, "Color", zero);
    color.set(tmpVec[0], tmpVec[1], tmpVec[2], tmpVec[3]);
    switch (check_option(block.getstring("Degrees")))
    {
    case _NONE:
        std::cerr << "Assuming 'rad'" << std::endl;
        rotat[0] *= DEG;
        rotat[1] *= DEG;
        rotat[2] *= DEG;
        break;
    case _DEG:
        break;
    case _RAD:
        rotat[0] *= DEG;
        rotat[1] *= DEG;
        rotat[2] *= DEG;
        break;
    default:
        std::cerr << "Unknown Degrees '" << block.getstring("Degrees")
                  << "'. Assuming 'rad'" << std::endl;
        rotat[0] *= DEG;
        rotat[1] *= DEG;
        rotat[2] *= DEG;
        break;
    }
    trans = new osg::MatrixTransform();
    trans->setName(myname);
    geode = new osg::Geode();
    vert = new osg::Vec3Array;
    norm = new osg::Vec3Array;
    geom = new osg::Geometry;
}

// setting mytriad coordinates
void D_Primitive::set_mytriads(GLfloat *AA)
{
    triadfrom[0] = AA[0];
    triadfrom[1] = AA[1];
    triadfrom[2] = AA[2];
    triadto[0] = AA[3];
    triadto[1] = AA[4];
    triadto[2] = AA[5];
    // debug
    // std::cerr << "set_mytriads()" << std::endl;
    // std::cerr << "Info(): triadfrom[0] = " << triadfrom[0] << std::endl;
    // std::cerr << "Info(): triadfrom[1] = " << triadfrom[1] << std::endl;
    // std::cerr << "Info(): triadfrom[2] = " << triadfrom[2] << std::endl;
    // std::cerr << "Info(): triadto[0] = " << triadto[0] << std::endl;
    // std::cerr << "Info(): triadto[1] = " << triadto[1] << std::endl;
    // std::cerr << "Info(): triadto[2] = " << triadto[2] << std::endl;
}

osg::ref_ptr<osg::Material> D_Primitive::globalmtl;

void D_Primitive::setMaterial()
{
    if (globalmtl.get() == NULL)
    {
        globalmtl = new osg::Material;
        globalmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));
        globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0));
        globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }
    osg::StateSet *geoState = geode->getOrCreateStateSet();
    geoState->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);

    osg::Vec4Array *colArr = new osg::Vec4Array();
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    geom->setColorArray(colArr);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
}

void D_Primitive::applyTrans()
{

    osg::Matrix translation;
    translation.makeTranslate(shift[0], shift[1], shift[2]);
    osg::Matrix rotation;
    //rotation.makeRotate(rotat[2],osg::Vec3(0,0,1),rotat[1],osg::Vec3(0,1,0),rotat[0],osg::Vec3(1,0,0));
    rotation.makeRotate(rotat[2] / 180.0 * M_PI, osg::Vec3(0, 0, 1), rotat[1] / 180.0 * M_PI, osg::Vec3(0, 1, 0), rotat[0] / 180.0 * M_PI, osg::Vec3(1, 0, 0));
    osg::Matrix transform = rotation * translation;
    osg::Vec3Array::iterator itr;
    for (itr = vert->begin();
         itr != vert->end();
         ++itr)
    {
        (*itr) = transform.preMult(*itr);
    }
    for (itr = norm->begin();
         itr != norm->end();
         ++itr)
    {
        (*itr) = osg::Matrix::transform3x3(transform, (*itr));
    }
}

////////////////////////////////////////////////////////////////////////

World::World(parmblock &block)
    : D_Primitive(block)
{
}

////////////////////////////////////////////////////////////////////////
D_Box::D_Box(parmblock &block)
    : D_Primitive(block)
{
    size[0] = block.getvalue("LengthX", 1.0);
    size[1] = block.getvalue("LengthY", 1.0);
    size[2] = block.getvalue("LengthZ", 1.0);

    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::QUADS, 0, 1);

    GLfloat x05 = 0.5 * size[0], y05 = 0.5 * size[1], z05 = 0.5 * size[2];

    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    vert->push_back(osg::Vec3(x05, y05, z05));
    vert->push_back(osg::Vec3(-x05, y05, z05));
    vert->push_back(osg::Vec3(-x05, -y05, z05));
    vert->push_back(osg::Vec3(x05, -y05, z05));

    norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
    norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
    norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
    norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
    vert->push_back(osg::Vec3(x05, y05, -z05));
    vert->push_back(osg::Vec3(x05, -y05, -z05));
    vert->push_back(osg::Vec3(-x05, -y05, -z05));
    vert->push_back(osg::Vec3(-x05, y05, -z05));

    norm->push_back(osg::Vec3(1.0, 0.0, 0.0));
    norm->push_back(osg::Vec3(1.0, 0.0, 0.0));
    norm->push_back(osg::Vec3(1.0, 0.0, 0.0));
    norm->push_back(osg::Vec3(1.0, 0.0, 0.0));
    vert->push_back(osg::Vec3(x05, -y05, z05));
    vert->push_back(osg::Vec3(x05, -y05, -z05));
    vert->push_back(osg::Vec3(x05, y05, -z05));
    vert->push_back(osg::Vec3(x05, y05, z05));

    norm->push_back(osg::Vec3(0.0, 1.0, 0.0));
    norm->push_back(osg::Vec3(0.0, 1.0, 0.0));
    norm->push_back(osg::Vec3(0.0, 1.0, 0.0));
    norm->push_back(osg::Vec3(0.0, 1.0, 0.0));
    vert->push_back(osg::Vec3(x05, y05, z05));
    vert->push_back(osg::Vec3(x05, y05, -z05));
    vert->push_back(osg::Vec3(-x05, y05, -z05));
    vert->push_back(osg::Vec3(-x05, y05, z05));

    norm->push_back(osg::Vec3(-1.0, 0.0, 0.0));
    norm->push_back(osg::Vec3(-1.0, 0.0, 0.0));
    norm->push_back(osg::Vec3(-1.0, 0.0, 0.0));
    norm->push_back(osg::Vec3(-1.0, 0.0, 0.0));
    vert->push_back(osg::Vec3(-x05, y05, z05));
    vert->push_back(osg::Vec3(-x05, y05, -z05));
    vert->push_back(osg::Vec3(-x05, -y05, -z05));
    vert->push_back(osg::Vec3(-x05, -y05, z05));

    norm->push_back(osg::Vec3(0.0, -1.0, 0.0));
    norm->push_back(osg::Vec3(0.0, -1.0, 0.0));
    norm->push_back(osg::Vec3(0.0, -1.0, 0.0));
    norm->push_back(osg::Vec3(0.0, -1.0, 0.0));
    vert->push_back(osg::Vec3(-x05, -y05, z05));
    vert->push_back(osg::Vec3(-x05, -y05, -z05));
    vert->push_back(osg::Vec3(x05, -y05, -z05));
    vert->push_back(osg::Vec3(x05, -y05, z05));
    primitives->push_back(6 * 4);

    geom->setNormalArray(norm);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);
    geode->addDrawable(geom);
    setMaterial();
    applyTrans();
    trans->addChild(geode);
}

////////////////////////////////////////////////////////////////////////

D_Sphere::D_Sphere(parmblock &block)
    : D_Primitive(block)
{
    nphi = block.getvalue("GridLong", 8);
    ntheta = block.getvalue("GridLati", 8);
    rx = block.getvalue("RadiusX", 1.0);
    ry = block.getvalue("RadiusY", 1.0);
    rz = block.getvalue("RadiusZ", 1.0);

    GLfloat phi, theta;
    GLfloat d_theta, d_phi;
    GLfloat v[3], n[3];

    // definitions
    d_theta = M_PI / ntheta;
    d_phi = 2.0 * M_PI / nphi;

    osg::DrawArrayLengths *primitivestf = new osg::DrawArrayLengths(osg::PrimitiveSet::TRIANGLE_FAN, 0, 2);
    // top cap

    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    vert->push_back(osg::Vec3(0.0, 0.0, rz));
    for (GLint i = 0; i <= nphi; ++i)
    {
        phi = i * d_phi;
        Surface(v, n, rx, ry, rz, phi, d_theta);
        norm->push_back(osg::Vec3(n[0], n[1], n[2]));
        vert->push_back(osg::Vec3(v[0], v[1], v[2]));
    }
    primitivestf->push_back(nphi + 2);

    // bottom face (definition clockwise for correct display)

    theta = M_PI - d_theta;
    norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
    vert->push_back(osg::Vec3(0.0, 0.0, -rz));
    for (GLint i = nphi; i >= 0; --i)
    {
        phi = i * d_phi;
        Surface(v, n, rx, ry, rz, phi, theta);
        norm->push_back(osg::Vec3(n[0], n[1], n[2]));
        vert->push_back(osg::Vec3(v[0], v[1], v[2]));
    }
    primitivestf->push_back(nphi + 2);

    geom->addPrimitiveSet(primitivestf);

    int offset = 0;
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::TRIANGLE_STRIP, (nphi + 2) * 2, (ntheta - 2));
    for (GLint i = 1; i < ntheta - 1; ++i)
    {

        theta = i * d_theta;
        for (GLint j = 0; j <= nphi; ++j)
        {
            phi = j * d_phi + offset * d_phi / 2.0;
            Surface(v, n, rx, ry, rz, phi, theta);
            norm->push_back(osg::Vec3(n[0], n[1], n[2]));
            vert->push_back(osg::Vec3(v[0], v[1], v[2]));
            Surface(v, n, rx, ry, rz, phi + d_phi / 2.0, theta + d_theta);
            norm->push_back(osg::Vec3(n[0], n[1], n[2]));
            vert->push_back(osg::Vec3(v[0], v[1], v[2]));
        }
        offset ^= 1;
        primitives->push_back(((nphi + 1)) * 2);
    }
    geom->setNormalArray(norm);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);
    //geom->setColor(color);
    geode->addDrawable(geom);
    setMaterial();
    applyTrans();
    trans->addChild(geode);
}

// surface of an ellipsoid
void D_Sphere::Surface(GLfloat *v, GLfloat *n,
                       const GLfloat rx, const GLfloat ry, const GLfloat rz,
                       const GLfloat phi, const GLfloat theta)
{
    v[0] = rx * std::sin(theta) * std::cos(phi);
    v[1] = ry * std::sin(theta) * std::sin(phi);
    v[2] = rz * std::cos(theta);

    n[0] = ry * rz * std::sin(theta) * std::sin(theta) * std::cos(phi);
    n[1] = rx * rz * std::sin(theta) * std::sin(theta) * std::sin(phi);
    n[2] = rx * ry * std::sin(theta) * std::cos(theta);
}

////////////////////////////////////////////////////////////////////////

D_Cylinder::D_Cylinder(parmblock &block)
    : D_Primitive(block)
{
    nphi = block.getvalue("GridLong", 8);
    rx_bot = block.getvalue("RadiusBottomX", 1.0);
    ry_bot = block.getvalue("RadiusBottomY", 1.0);
    rx_top = block.getvalue("RadiusTopX", 1.0);
    ry_top = block.getvalue("RadiusTopY", 1.0);
    h = block.getvalue("HeightZ", 1.0);

    GLfloat phi;
    GLfloat d_phi;
    GLfloat h05 = 0.5 * h;
    d_phi = 2.0 * M_PI / nphi;

    osg::DrawArrayLengths *primitivestf = new osg::DrawArrayLengths(osg::PrimitiveSet::TRIANGLE_FAN, 0, 2);
    // top face
    norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
    vert->push_back(osg::Vec3(0.0, 0.0, h05));
    for (GLint i = 0; i < nphi + 1; ++i)
    {
        phi = i * d_phi;
        norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
        vert->push_back(osg::Vec3(rx_top * std::cos(phi), ry_top * std::sin(phi), h05));
    }
    primitivestf->push_back(nphi + 2);

    // bottom face (definition clockwise for correct display)

    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    vert->push_back(osg::Vec3(0.0, 0.0, -h05));
    for (GLint i = 0; i < nphi + 1; ++i)
    {
        phi = -i * d_phi;
        norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
        vert->push_back(osg::Vec3(rx_bot * std::cos(phi), ry_bot * std::sin(phi), -h05));
    }
    primitivestf->push_back(nphi + 2);

    geom->addPrimitiveSet(primitivestf);

    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::QUAD_STRIP, (nphi + 2) * 2, 1);
    for (GLint i = 0; i <= nphi; ++i)
    {
        phi = i * d_phi;

        GLfloat co = std::cos(phi);
        GLfloat si = std::sin(phi);
        GLfloat n_top[3], n_bot[3];

        n_top[0] = ry_top * co;
        n_top[1] = rx_top * si;
        n_top[2] = -rx_top * (ry_top - ry_bot) / h * si * si
                   - ry_top * (rx_top - rx_bot) / h * co * co;

        n_bot[0] = ry_bot * co;
        n_bot[1] = rx_bot * si;
        n_bot[2] = -rx_bot * (ry_top - ry_bot) / h * si * si
                   - ry_bot * (rx_top - rx_bot) / h * co * co;
        if (rx_top == 0.0 && ry_top == 0.0)
        {
            n_top[0] = n_bot[0];
            n_top[1] = n_bot[1];
            n_top[2] = n_bot[2];
        }
        if (rx_bot == 0.0 && ry_bot == 0.0)
        {
            n_bot[0] = n_top[0];
            n_bot[1] = n_top[1];
            n_bot[2] = n_top[2];
        }
        norm->push_back(osg::Vec3(n_top[0], n_top[1], n_top[2]));
        norm->push_back(osg::Vec3(n_bot[0], n_bot[1], n_bot[2]));
        vert->push_back(osg::Vec3(rx_top * co, ry_top * si, h05));
        vert->push_back(osg::Vec3(rx_bot * co, ry_bot * si, -h05));
    }
    primitives->push_back((nphi + 1) * 2);
    geom->setNormalArray(norm);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);
    //geom->setColor(color);
    geode->addDrawable(geom);
    setMaterial();
    applyTrans();
    trans->addChild(geode);
}
////////////////////////////////////////////////////////////////////////

D_Cone::D_Cone(parmblock &block)
    : D_Primitive(block)
{
    nphi = block.getvalue("GridLong", 8);
    rx = block.getvalue("RadiusX", 1.0);
    ry = block.getvalue("RadiusY", 1.0);
    h = block.getvalue("HeightZ", 1.0);

    GLfloat phi;
    GLfloat dphi = 2.0 * M_PI / nphi;

    GLfloat v[3];

    osg::DrawArrayLengths *primitivestf = new osg::DrawArrayLengths(osg::PrimitiveSet::TRIANGLE_FAN, 0, 2);
    // top face
    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    vert->push_back(osg::Vec3(0.0, 0.0, h));
    for (GLint i = 0; i < nphi + 1; ++i)
    {
        v[0] = rx * std::cos(i * dphi);
        v[1] = ry * std::sin(i * dphi);
        v[2] = 0.0;
        vert->push_back(osg::Vec3(v[0], v[1], v[2]));
        v[0] = h * std::cos(i * dphi);
        v[1] = h * std::sin(i * dphi);
        v[2] = rx * std::cos(i * dphi) + ry * std::sin(i * dphi);
        osg::Vec3 n(v[0], v[1], v[2]);
        n.normalize();
        norm->push_back(n);
    }
    primitivestf->push_back(nphi + 2);

    // bottom face (definition clockwise for correct display)

    norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
    vert->push_back(osg::Vec3(0.0, 0.0, 0.0));
    for (GLint i = nphi; i >= 0; --i)
    {
        phi = i * dphi;
        glVertex3f(rx * std::cos(phi), ry * std::sin(phi), 0.0);
        norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
        vert->push_back(osg::Vec3(rx * std::cos(phi), ry * std::sin(phi), 0.0));
    }
    primitivestf->push_back(nphi + 2);

    geom->addPrimitiveSet(primitivestf);

    geom->setNormalArray(norm);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(vert);
    geode->addDrawable(geom);
    setMaterial();
    applyTrans();
    trans->addChild(geode);
}

////////////////////////////////////////////////////////////////////////

D_Axes::D_Axes(parmblock &block)
    : D_Primitive(block)
{
    length = block.getvalue("Length", 1.0);
    float tmpVec[4];
    float defaultColor[4] = { color[0], color[1], color[2], color[3] };
    block.getvector(tmpVec, 4, "ColorX", defaultColor);
    colorX.set(tmpVec[0], tmpVec[1], tmpVec[2], tmpVec[3]);
    block.getvector(tmpVec, 4, "ColorY", defaultColor);
    colorY.set(tmpVec[0], tmpVec[1], tmpVec[2], tmpVec[3]);

    osg::DrawArrayLengths *primitiveslines = new osg::DrawArrayLengths(osg::PrimitiveSet::LINES, 0, 1);

    if (globalmtl.get() == NULL)
    {
        globalmtl = new osg::Material;
        globalmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));
        globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0));
        globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }
    osg::StateSet *geoState = geode->getOrCreateStateSet();
    geoState->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::Vec4Array *colArr = new osg::Vec4Array();
    geom->setColorArray(colArr);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    GLfloat l95 = 0.95 * length;
    GLfloat l02 = 0.02 * length;

    vert->push_back(osg::Vec3(0.0, 0.0, 0.0));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(length, 0.0, 0.0));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));

    vert->push_back(osg::Vec3(length, 0.0, 0.0));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(l95, l02, l02));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(length, 0.0, 0.0));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(l95, -l02, l02));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(length, 0.0, 0.0));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(l95, -l02, -l02));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(length, 0.0, 0.0));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    vert->push_back(osg::Vec3(l95, l02, -l02));
    colArr->push_back(osg::Vec4(colorX[0], colorX[1], colorX[2], colorX[3]));
    // y-axis
    vert->push_back(osg::Vec3(0.0, 0.0, 0.0));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(0.0, length, 0.0));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));

    vert->push_back(osg::Vec3(0.0, length, 0.0));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(l02, l95, l02));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(0.0, length, 0.0));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(-l02, l95, l02));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(0.0, length, 0.0));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(-l02, l95, -l02));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(0.0, length, 0.0));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    vert->push_back(osg::Vec3(l02, l95, -l02));
    colArr->push_back(osg::Vec4(colorY[0], colorY[1], colorY[2], colorY[3]));
    // z-axis
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(0.0, 0.0, 0.0));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(0.0, 0.0, length));

    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(0.0, 0.0, length));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(l02, l02, l95));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(0.0, 0.0, length));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(-l02, l02, l95));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(0.0, 0.0, length));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(-l02, -l02, l95));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(0.0, 0.0, length));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    vert->push_back(osg::Vec3(l02, -l02, l95));
    colArr->push_back(osg::Vec4(color[0], color[1], color[2], color[3]));
    primitiveslines->push_back(15);

    geom->addPrimitiveSet(primitiveslines);

    geom->setNormalArray(norm);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(vert);
    geode->addDrawable(geom);
    setMaterial();
    applyTrans();
    trans->addChild(geode);
}
////////////////////////////////////////////////////////////////////////

D_Extrude::D_Extrude(parmblock &block)
    : D_Primitive(block)
{
    h = block.getvalue("HeightZ", 1.0);

    std::string all = block.getstring("VertexList");
    if (*all.begin() == '{' && *(all.end() - 1) == '}')
    {
        all.erase(all.begin());
        all.erase(all.end() - 1);
    }
    else
    {
        std::cerr << "Error! D_Primitive '" << myname
                  << "': Missing braces in 'VertexList'. Exiting.\n";
        std::exit(-1);
    }

    int dim = 3;
    int ntot = 1;
    for (unsigned int j = 0; j < all.size(); ++j)
    {
        if (all[j] == ',')
        {
            ++ntot;
        }
    }

    nvtx = ntot / dim;
    if (nvtx * dim != ntot)
    {
        std::cerr << "Error! D_Primitive '" << myname
                  << "': Missing data in 'VertexList'. Exiting.\n";
        std::exit(-1);
    }
    vtxbuf = new GLfloat *[dim];
    vtxbuf[0] = new GLfloat[dim * nvtx];
    for (int i = 1; i < dim; ++i)
    {
        vtxbuf[i] = vtxbuf[0] + i * nvtx;
    }

    int lo = 0;
    for (int i = 0; i < nvtx; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            const int hi = all.find(',', lo);
            vtxbuf[j][i] = atof(all.substr(lo, hi - lo).c_str());
            lo = hi + 1;
        }
    }

    vtx_x = vtxbuf[0];
    vtx_y = vtxbuf[1];
    vtx_z = vtxbuf[2];

    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POLYGON, 0, 2);

    for (GLint i = 0; i < nvtx; ++i)
    {
        norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
        vert->push_back(osg::Vec3(vtx_x[i], vtx_y[i], h / 2.0));
    }
    primitives->push_back(nvtx);

    for (GLint i = nvtx - 1; i >= 0; --i)
    {
        norm->push_back(osg::Vec3(0.0, 0.0, -1.0));
        vert->push_back(osg::Vec3(vtx_x[i], vtx_y[i], -h / 2.0));
    }
    primitives->push_back(nvtx);

    // bottom face (definition clockwise for correct display)

    osg::DrawArrayLengths *primitivesqs = new osg::DrawArrayLengths(osg::PrimitiveSet::QUAD_STRIP, 2 * nvtx, 1);

    for (GLint i = 0; i <= nvtx; ++i)
    {
        GLfloat nz = std::sqrt(SQR(vtx_x[i % nvtx]) + SQR(vtx_y[i % nvtx]));
        norm->push_back(osg::Vec3(vtx_x[i % nvtx], vtx_y[i % nvtx], nz));
        vert->push_back(osg::Vec3(vtx_x[i % nvtx], vtx_y[i % nvtx], h / 2.0));
        norm->push_back(osg::Vec3(vtx_x[i % nvtx], vtx_y[i % nvtx], -nz));
        vert->push_back(osg::Vec3(vtx_x[i % nvtx], vtx_y[i % nvtx], -h / 2.0));
    }
    primitivesqs->push_back(nvtx * 2);

    geom->addPrimitiveSet(primitivesqs);
    geom->addPrimitiveSet(primitives);

    geom->setNormalArray(norm);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(vert);
    geode->addDrawable(geom);
    setMaterial();
    applyTrans();
    trans->addChild(geode);
}

D_Extrude::~D_Extrude()
{
    if (vtxbuf != 0)
    {
        delete[] vtxbuf[0];
        delete[] vtxbuf;
    }
}

////////////////////////////////////////////////////////////////////////

D_Tetraeder::D_Tetraeder(parmblock &block)
    : D_Primitive(block)
{
    GLfloat dummy[6] = { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
    block.getvector(vtx1, 3, "Vertex1", dummy);
    block.getvector(vtx2, 3, "Vertex2", dummy + 3);
    block.getvector(vtx3, 3, "Vertex3", dummy + 2);
    block.getvector(vtx4, 3, "Vertex4", dummy + 1);

    osg::DrawArrayLengths *primitivestf = new osg::DrawArrayLengths(osg::PrimitiveSet::TRIANGLE_FAN, 0, 1);

    norm->push_back(osg::Vec3(0.0, 0.0, 1.0));
    vert->push_back(osg::Vec3(vtx1[0], vtx1[1], vtx1[2]));
    norm->push_back(osg::Vec3(-1.0, 0.0, (vtx1[0] - vtx2[0]) / (vtx1[2] - vtx2[2])));
    vert->push_back(osg::Vec3(vtx2[0], vtx2[1], vtx2[2]));
    norm->push_back(osg::Vec3(1.0, -1.0, 0.0));
    vert->push_back(osg::Vec3(vtx3[0], vtx3[1], vtx3[2]));
    norm->push_back(osg::Vec3(1.0, 1.0, 0.0));
    vert->push_back(osg::Vec3(vtx4[0], vtx4[1], vtx4[2]));
    norm->push_back(osg::Vec3(-1.0, 0.0, (vtx1[0] - vtx2[0]) / (vtx1[2] - vtx2[2])));
    vert->push_back(osg::Vec3(vtx2[0], vtx2[1], vtx2[2]));
    primitivestf->push_back(5);

    geom->addPrimitiveSet(primitivestf);

    geom->setNormalArray(norm);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(vert);
    //geom->setColor(color);
    geode->addDrawable(geom);
    setMaterial();
    applyTrans();
    trans->addChild(geode);
}

////////////////////////////////////////////////////////////////////////

D_Surface::D_Surface(parmblock &block)
    : D_Primitive(block)
{
    sname = block.getstring("SurfaceName");
    surfacename = new char[sname.length() + 1];
    std::strcpy(surfacename, sname.c_str());
    scale = block.getvalue("Scalefactor", 1.0);
    osg::MatrixTransform *localtrans = new osg::MatrixTransform;

    osg::Matrix translation = osg::Matrix::translate(shift[0], shift[1], shift[2]);
    osg::Matrix rotation = osg::Matrix::rotate(rotat[2], osg::Vec3(0, 0, 1), rotat[1], osg::Vec3(0, 1, 0), rotat[0], osg::Vec3(1, 0, 0));
    osg::Matrix transform = osg::Matrix::scale(scale, scale, scale) * rotation * translation;
    localtrans->setMatrix(transform);
    //opencover::coVRFileManager::instance()->loadFile(sname.c_str(), NULL,localtrans);
    std::string filename = DEMOAPlugin::instance()->path + "/" + sname;
    osg::Node *node = osgDB::readNodeFile(filename.c_str());
    if (node)
        localtrans->addChild(node);
    trans->addChild(localtrans);
}
////////////////////////////////////////////////////////////////////////

D_Muscle::D_Muscle(parmblock &block)
    : D_Primitive(block)
{
    myforcename = block.getstring("ForceName");
    linethickness = block.getvalue("LineThickness", 1.0);
    setMaterial();
    applyTrans();
}
/*
// drawing definition for a UserForce
void D_Muscle::Define()
{
	glDisable(GL_LIGHTING);
	glLineWidth(linethickness);

	glBegin(GL_LINES);
	glColor4fv(color);
	// origin of the line
	glVertex3f(triadfrom[0], triadfrom[1], triadfrom[2]);
	// ending point of the line
	glVertex3f(triadto[0], triadto[1], triadto[2]);
	glEnd();

	glEnable(GL_LIGHTING);
	glClearColor(0.0, 0.0, 0.0, 0.0);
}*/

////////////////////////////////////////////////////////////////////////

D_IVD::D_IVD(parmblock &block)
    : D_Primitive(block)
{
    myforcename = block.getstring("ForceName");
    myradius = block.getvalue("Radius", 1.0);
    setMaterial();
    applyTrans();
}
/*
// drawing definition of an ellipsoid
void D_IVD::Define()
{
	// calc middle point between two force triads
	GLfloat jointpoint[3] = {0.0, 0.0, 0.0};
	jointpoint[0] = (triadfrom[0] + triadto[0]) / 2;
	jointpoint[1] = (triadfrom[1] + triadto[1]) / 2;
	jointpoint[2] = (triadfrom[2] + triadto[2]) / 2;
	// add shift from model file to calculated mid point
	GLfloat shifthere[3] = {0.0, 0.0, 0.0};
	shifthere[0] = jointpoint[0] + shift[0];
	shifthere[1] = jointpoint[1] + shift[1];
	shifthere[2] = jointpoint[2] + shift[2];

	// debug
	// // calc change in orientation, here just rotation around x-axis
	// GLfloat rotx;
	// // unit vector of zy- plane (x=0)
	// GLfloat ux[3] = {0.0, 0.0, 1.0};
	// double dotprod = ux[0]*jointpoint[0] + ux[1]*jointpoint[1] + ux[2]*jointpoint[2];
	// double mod1 = std::sqrt(ux[0]*ux[0] + ux[1]*ux[1] + ux[2]*ux[2]);
	// double mod2 = std::sqrt(jointpoint[0]*jointpoint[0] + jointpoint[1]*jointpoint[1] + jointpoint[2]*jointpoint[2]);
	// rotx = std::acos(dotprod/mod1*mod2);
	// rotx = rotx*DEG;
	// std::cerr << "dotprod =  " << dotprod << std::endl;
	// std::cerr << "mod1 = " << mod1 << std::endl;
	// std::cerr << "mod2 = " << mod2 << std::endl;
	// std::cerr << "rotx = " << rotx << std::endl;

	//glPushMatrix();
	// give color
	glColor4fv(color);
	//transformations here...
	glTranslatef(shifthere[0], shifthere[1], shifthere[2]);
	glRotatef(rotat[0], 1.0, 0.0, 0.0);
	glRotatef(rotat[1], 0.0, 1.0, 0.0);
	glRotatef(rotat[2], 0.0, 0.0, 1.0);
	GLdouble dInnerRadius = myradius;
	GLdouble dOuterRadius = 2*myradius;
	GLint nSides = 20;
	GLint nRings = 20;
	glutSolidTorus(dInnerRadius, dOuterRadius, nSides, nRings);

	glShadeModel(GL_FLAT);
	glClearColor(0.0, 0.0, 0.0, 0.0);
}*/
