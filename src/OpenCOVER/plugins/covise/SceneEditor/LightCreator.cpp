/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LightCreator.h"
#include "Light.h"

#include <QDir>
#include <iostream>

#include <osg/LightSource>
#include <osgShadow/ShadowedScene>
#include <osgShadow/ShadowTexture>
#include <osgShadow/StandardShadowMap>
#include <osgShadow/SoftShadowMap>
#include <osgShadow/MinimalShadowMap>
#include <osgShadow/SoftShadowMap>
#include <osgShadow/LightSpacePerspectiveShadowMap>

using namespace covise;
using namespace opencover;
LightCreator::LightCreator()
{
}

LightCreator::~LightCreator()
{
}

SceneObject *LightCreator::createFromXML(QDomElement *root)
{
    Light *light = new Light();
    if (!buildFromXML(light, root))
    {
        delete light;
        return NULL;
    }
    return light;
}

bool LightCreator::buildFromXML(SceneObject *so, QDomElement *root)
{
    if (!buildGeometryFromXML((Light *)so, root))
    {
        return false;
    }
    return SceneObjectCreator::buildFromXML(so, root);
}

bool LightCreator::buildGeometryFromXML(Light *light, QDomElement *root)
{
    osg::Light *l = new osg::Light();

    QDomElement geoElem = root->firstChildElement("geometry");
    if (!geoElem.isNull())
    {
        QDomElement pos = geoElem.firstChildElement("position");
        if (!pos.isNull())
        {
            float x, y, z = 100.f;
            x = pos.attribute("x").toFloat();
            y = pos.attribute("y").toFloat();
            z = pos.attribute("z").toFloat();
            l->setPosition(osg::Vec4(x, y, z, 1));
        }
        QDomElement dir = geoElem.firstChildElement("direction");
        if (!dir.isNull())
        {
            float x, y, z = 1.f;
            x = dir.attribute("x").toFloat();
            y = dir.attribute("y").toFloat();
            z = dir.attribute("z").toFloat();
            l->setDirection(osg::Vec3(x, y, z));
        }
        QDomElement amb = geoElem.firstChildElement("ambient");
        if (!amb.isNull())
        {
            float r, g, b, a = 1.f;
            r = amb.attribute("r").toFloat();
            g = amb.attribute("g").toFloat();
            b = amb.attribute("b").toFloat();
            a = amb.attribute("a").toFloat();
            l->setAmbient(osg::Vec4(r, g, b, a));
        }
        QDomElement dif = geoElem.firstChildElement("diffuse");
        if (!amb.isNull())
        {
            float r, g, b, a = 1.f;
            r = dif.attribute("r").toFloat();
            g = dif.attribute("g").toFloat();
            b = dif.attribute("b").toFloat();
            a = dif.attribute("a").toFloat();
            l->setDiffuse(osg::Vec4(r, g, b, a));
        }
        QDomElement spec = geoElem.firstChildElement("specular");
        if (!spec.isNull())
        {
            float r, g, b, a = 1.f;
            r = spec.attribute("r").toFloat();
            g = spec.attribute("g").toFloat();
            b = spec.attribute("b").toFloat();
            a = spec.attribute("a").toFloat();
            l->setSpecular(osg::Vec4(r, g, b, a));
        }
        QDomElement tec = geoElem.firstChildElement("technique");
        osgShadow::ShadowedScene *shadowedScene = dynamic_cast<osgShadow::ShadowedScene *>(opencover::cover->getObjectsRoot()->getParent(0));
        if (!shadowedScene)
        {
            return false;
        }
        if (!tec.isNull())
        {
            std::string technique = tec.attribute("value").toStdString();
            if (technique.compare("ShadowTexture") == 0)
            {
                osg::ref_ptr<osgShadow::ShadowTexture> st = new osgShadow::ShadowTexture;
                shadowedScene->setShadowTechnique(st.get());
            }
            else if (technique.compare("SoftShadowMap") == 0)
            {
                osg::ref_ptr<osgShadow::SoftShadowMap> sm = new osgShadow::SoftShadowMap;
                shadowedScene->setShadowTechnique(sm.get());
            }
            else if (technique.compare("StandardShadowMap") == 0)
            {
                osg::ref_ptr<osgShadow::StandardShadowMap> st = new osgShadow::StandardShadowMap;
                shadowedScene->setShadowTechnique(st.get());
            }
            //else if (technique.compare("ParallelSplitShadowMap")==0)
            //{
            //}
            else if (technique.compare("LightSpacePerspectiveShadowMapVB") == 0)
            {
                osg::ref_ptr<osgShadow::MinimalShadowMap> sm = new osgShadow::LightSpacePerspectiveShadowMapVB;
                /*unsigned int texSize = 2048;
            //float minLightMargin = 100000.f;
            //float maxFarPlane = 5;
            //unsigned int baseTexUnit = 0;
            unsigned int shadowTexUnit = 1;
            //sm->setMinLightMargin( minLightMargin );
            //sm->setMaxFarPlane( maxFarPlane );
            sm->setTextureSize( osg::Vec2s( texSize, texSize ) );
            sm->setShadowTextureCoordIndex( shadowTexUnit );
            sm->setShadowTextureUnit( shadowTexUnit );*/
                shadowedScene->setShadowTechnique(sm.get());
            }
            else if (technique.compare("LightSpacePerspectiveShadowMapCB") == 0)
            {
                osg::ref_ptr<osgShadow::MinimalShadowMap> sm = new osgShadow::LightSpacePerspectiveShadowMapCB;
                shadowedScene->setShadowTechnique(sm.get());
            }
            else if (technique.compare("LightSpacePerspectiveShadowMapDB") == 0)
            {
                osg::ref_ptr<osgShadow::MinimalShadowMap> sm = new osgShadow::LightSpacePerspectiveShadowMapDB;
                shadowedScene->setShadowTechnique(sm.get());
            }
            else
            {
                osg::ref_ptr<osgShadow::ShadowMap> sm = new osgShadow::ShadowMap;
                shadowedScene->setShadowTechnique(sm.get());
            }
        }
        else
        {
            osg::ref_ptr<osgShadow::ShadowMap> sm = new osgShadow::ShadowMap;
            shadowedScene->setShadowTechnique(sm.get());
        }
        QDomElement tu = geoElem.firstChildElement("textureUnit");
        if (!tu.isNull())
        {
            int textureUnit = tu.attribute("value").toInt();
            osgShadow::StandardShadowMap *sm = dynamic_cast<osgShadow::StandardShadowMap *>(shadowedScene->getShadowTechnique());
            if (sm)
            {
                sm->setShadowTextureCoordIndex(textureUnit);
                sm->setShadowTextureUnit(textureUnit);
            }
        }
        else
        {
            int textureUnit = 7;
            osgShadow::StandardShadowMap *sm = dynamic_cast<osgShadow::StandardShadowMap *>(shadowedScene->getShadowTechnique());
            if (sm)
            {
                sm->setShadowTextureCoordIndex(textureUnit);
                sm->setShadowTextureUnit(textureUnit);
            }
        }
        QDomElement ts = geoElem.firstChildElement("textureSize");
        if (!ts.isNull())
        {
            int textureSize = ts.attribute("value").toInt();
            osgShadow::StandardShadowMap *sm = dynamic_cast<osgShadow::StandardShadowMap *>(shadowedScene->getShadowTechnique());
            if (sm)
                sm->setTextureSize(osg::Vec2s(textureSize, textureSize));
        }
    }

    osg::LightSource *ls = new osg::LightSource();
    ls->setLight(l);

    light->setGeometryNode(ls);

    return true;
}
