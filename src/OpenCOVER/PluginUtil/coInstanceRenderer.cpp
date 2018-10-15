/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coInstanceRenderer.h"
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Uniform>
#include <osg/Texture1D>
#include <osgDB/ReadFile>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRShader.h>

using namespace osg;
using namespace covise;
using namespace opencover;

coInstanceRenderer *coInstanceRenderer::instance_ = NULL;

coInstanceRenderer::coInstanceRenderer()
{
    objects.reserve(20);
    instanceObjects = new osg::Group();
    instanceObjects->setName("instanceObjects");
    cover->getObjectsRoot()->addChild(instanceObjects);
}

coInstanceRenderer *coInstanceRenderer::instance()
{
    static coInstanceRenderer *singleton = NULL;
    if (!singleton)
        singleton = new coInstanceRenderer;
    return singleton;
}

int coInstanceRenderer::addObject(std::string textureName, float width, float height)
{
    int id = objects.size();
    objects.push_back(new coInstanceObject(textureName, width, height));
    instanceObjects->addChild(objects[id]->getGeode());
    return id;
}

void coInstanceRenderer::addInstances(osg::Vec3Array &positions, int type)
{
    objects[type]->addInstances(positions);
}

coInstanceRenderer::~coInstanceRenderer()
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        delete objects[i];
    }
    objects.clear();
}

void coInstanceObject::createTwoQuadGeometry(float width, float height)
{

    osg::Vec3Array *v = new osg::Vec3Array;
    v->resize(8);
    geom->setVertexArray(v);

    // Geometry for a single quad.
    (*v)[0] = osg::Vec3(-width / 2.0, 0., 0);
    (*v)[1] = osg::Vec3(width / 2.0, 0., 0);
    (*v)[2] = osg::Vec3(width / 2.0, 0., height);
    (*v)[3] = osg::Vec3(-width / 2.0, 0., height);
    (*v)[4] = osg::Vec3(0., -width / 2.0, 0);
    (*v)[5] = osg::Vec3(0., width / 2.0, 0);
    (*v)[6] = osg::Vec3(0., width / 2.0, height);
    (*v)[7] = osg::Vec3(0., -width / 2.0, height);

    osg::Vec2Array *tc = new osg::Vec2Array;
    tc->resize(8);
    geom->setTexCoordArray(0, tc);

    (*tc)[0] = osg::Vec2(0, 0);
    (*tc)[1] = osg::Vec2(1, 0);
    (*tc)[2] = osg::Vec2(1, 1);
    (*tc)[3] = osg::Vec2(0, 1);
    (*tc)[4] = osg::Vec2(0, 0);
    (*tc)[5] = osg::Vec2(1, 0);
    (*tc)[6] = osg::Vec2(1, 1);
    (*tc)[7] = osg::Vec2(0, 1);
}

coInstanceObject::coInstanceObject(std::string textureName, float w, float h)
{
    width = w;
    height = h;
    geode = new osg::Geode;
    geom = new osg::Geometry;
    // Configure the Geometry for use with EXT_draw_arrays:
    // DL off and buffer objects on.
    geom->setSupportsDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    createTwoQuadGeometry(width, height);
    geode->addDrawable(geom.get());

    // Create a StateSet to render the instanced Geometry.
    stateSet = new osg::StateSet;

    osg::ref_ptr<osg::Image> objectImage = osgDB::readImageFile(coVRFileManager::instance()->getName(textureName.c_str()));
    if (!objectImage.valid())
    {
        osg::notify(osg::ALWAYS) << "Can't open image file" << textureName << std::endl;
    }
    else
    {
        osg::Texture2D *objectTexture = new osg::Texture2D(objectImage.get());
        objectTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        objectTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
        stateSet->setTextureAttribute(0, objectTexture);
        //stateSet->setUseModelViewAndProjectionUniforms(true);
        geode->setStateSet(stateSet);
        coVRShader *shader = coVRShaderList::instance()->get("instance");
        if (shader)
        {
            shader->apply(geode, geom);
        }
    }
    geode->setName(textureName);
}
coInstanceObject::~coInstanceObject()
{
}

void coInstanceObject::addInstances(osg::Vec3Array &positions)
{
    float maxX = -100000, maxY = -100000, maxZ = -100000, minX = 100000, minY = 100000, minZ = 100000;

    // Create texture and image
    osg::Texture *texture = new osg::Texture1D;
    osg::Image *image = new osg::Image();
    image->allocateImage(positions.size() * 4, 1, 1, GL_LUMINANCE, GL_FLOAT);
    texture->setInternalFormat(GL_LUMINANCE32F_ARB);
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
    texture->setImage(0, image);

    // Set data
    float *data = reinterpret_cast<float *>(image->data());
/* ... */

// Set texture to node
#define TEXTURE_UNIT_NUMBER 1
    stateSet->setTextureAttributeAndModes(TEXTURE_UNIT_NUMBER, texture);

    for (size_t i = 0; i < positions.size(); i++)
    {
        data[i * 4] = positions[i].x();
        data[i * 4 + 1] = positions[i].y();
        data[i * 4 + 2] = positions[i].z();
        data[i * 4 + 3] = 1;
        if (positions[i].x() < minX)
            minX = positions[i].x();
        if (positions[i].x() > maxX)
            maxX = positions[i].x();
        if (positions[i].y() < minY)
            minY = positions[i].y();
        if (positions[i].y() > maxY)
            maxY = positions[i].y();
        if (positions[i].z() < minZ)
            minZ = positions[i].z();
        if (positions[i].z() > maxZ)
            maxZ = positions[i].z();
    }
    minX -= width / 2.0;
    minY -= width / 2.0;
    maxX += width / 2.0;
    maxY += width / 2.0;
    maxZ += height;
    osg::BoundingBox bb(minX, minY, minZ, maxX, maxY, maxZ);
    geom->setInitialBound(bb);
}
