/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Logo.h"

#include <config/CoviseConfig.h>
#include <cover/coVRFileManager.h>
#include <osg/Geometry>
#include <osg/TexEnv>
#include <osgDB/ReaderWriter>
#include <osgDB/ReadFile>

Logo::Logo(std::string domain, osg::ref_ptr<osg::Camera> camera)
    : m_camera(camera)
    , m_isValid(true)
{
    int width = covise::coCoviseConfig::getInt("width", "COVER.Plugin.Logo" + domain, -1);
    int height = covise::coCoviseConfig::getInt("height", "COVER.Plugin.Logo" + domain, -1);
    int x = covise::coCoviseConfig::getInt("x", "COVER.Plugin.Logo" + domain, 10);
    int y = covise::coCoviseConfig::getInt("y", "COVER.Plugin.Logo" + domain, 10);
    float alpha = covise::coCoviseConfig::getFloat("alpha", "COVER.Plugin.Logo" + domain, 0.8);
    string logoFile = covise::coCoviseConfig::getEntry("file", "COVER.Plugin.Logo" + domain, "");

    osg::Geometry *geom = new osg::Geometry;
    osg::StateSet *stateset = geom->getOrCreateStateSet();
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    const char *logoName = opencover::coVRFileManager::instance()->getName(logoFile.c_str());
    osg::Image *image = NULL;
    if (logoName && (image = osgDB::readImageFile(logoName)) != NULL)
    {
        osg::Texture2D *texture = new osg::Texture2D;
        texture->setImage(image);

        osg::TexEnv *texenv = new osg::TexEnv;
        texenv->setMode(osg::TexEnv::REPLACE);

        stateset->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
        stateset->setTextureAttribute(0, texenv);
        if (width < 0)
            width = image->s();
        if (height < 0)
            height = image->t();
    }
    else
    {
        m_isValid = false;
    }

    if (width < 0)
        width = 100;
    if (height < 0)
        height = 100;

    osg::Vec3Array *vertices = new osg::Vec3Array;

    vertices->push_back(osg::Vec3(x, y + height, 0.001));
    vertices->push_back(osg::Vec3(x, y, 0.001));
    vertices->push_back(osg::Vec3(x + width, y, 0.001));
    vertices->push_back(osg::Vec3(x + width, y + height, 0.001));
    geom->setVertexArray(vertices);

    osg::Vec2Array *texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(0, 1));
    texcoords->push_back(osg::Vec2(0, 0));
    texcoords->push_back(osg::Vec2(1, 0));
    texcoords->push_back(osg::Vec2(1, 1));
    geom->setTexCoordArray(0, texcoords);

    osg::Vec3Array *normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1.0f, 1.0, 1.0f, alpha));
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

    m_geode = new osg::Geode();
    m_geode->addDrawable(geom);
}

Logo::~Logo()
{
}

void Logo::hide()
{
    if (!m_isValid)
    {
        return;
    }
    if (m_camera->containsNode(m_geode.get()))
    {
        m_camera->removeChild(m_geode.get());
    }
}

void Logo::show()
{
    if (!m_isValid)
    {
        return;
    }
    if (!m_camera->containsNode(m_geode.get()))
    {
        m_camera->addChild(m_geode.get());
    }
}

bool Logo::isValid()
{
    return m_isValid;
}
