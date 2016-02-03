/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneConf.h"
#include "KoordAxis.h"
#include <osg/Material>
#include <osg/LightModel>

SceneConf::SceneConf(void)
{
    confGroup = makeSceneConf();
}

SceneConf::~SceneConf(void)
{
    confGroup.release();
}

osg::Group *SceneConf::makeSceneConf()
{
    //Gruppe für Konfigurationsszene
    osg::ref_ptr<osg::Group> new_confGroup = new osg::Group();
    new_confGroup->setName("Konfiguration");

    osg::ref_ptr<osg::MatrixTransform> screenGeo = screen->draw();
    osg::StateSet *screenStateset = screenGeo->getOrCreateStateSet();

    if (screen->getStateMesh())
    {
        //Beleuchtung deaktivieren
        screenStateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        screenStateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    }
    else
    {
        //Beleuchtung aktivieren
        //----------------------
        screenStateset->setMode(GL_BLEND, osg::StateAttribute::ON);

        //Material dem Screen hinzufügen
        osg::ref_ptr<osg::Material> material = new osg::Material;
        material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
        material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 0.4));
        material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
        material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0, 0.0, 0.0, 0.4));
        material->setShininess(osg::Material::FRONT_AND_BACK, 10.0f);
        screenStateset->setAttributeAndModes(material.get(), osg::StateAttribute::ON);

        //beidseitige Beleuchtung
        osg::ref_ptr<osg::LightModel> screenLtModel = new osg::LightModel();
        screenLtModel->setTwoSided(true);
        screenStateset->setAttributeAndModes(screenLtModel.get(), osg::StateAttribute::ON);
        screenStateset->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    }
    //Radius der BoundingSphere um Screen
    osg::BoundingSphere boundScreen = screenGeo->getBound();
    float boundRadius = boundScreen.radius();

    //hinzufügen aller Geometrien, damit diese gerendert werden
    //- KoordinatenAchsen
    KoordAxis axis;
    new_confGroup->addChild(axis.createAxesGeometry(boundRadius / 15));

    //Screengeometrie
    new_confGroup->addChild(screenGeo.get());

    //- Projektor mit proj. Screen von jedem Channel
    new_confGroup->addChild(makeProjectorsGroup());

    return new_confGroup.release();
}

osg::Group *SceneConf::makeProjectorsGroup()
{
    osg::ref_ptr<osg::Group> projectorsGroup = new osg::Group();
    projectorsGroup->setName("Projektoren");
    for (unsigned int i = 0; i < projectors.size(); i++)
    {
        projectorsGroup->addChild(getProjector(i)->draw());
    }
    return projectorsGroup.release();
}

void SceneConf::updateScene(int num)
{
    osg::ref_ptr<osg::Group> new_confGroup = makeSceneConf();

    for (unsigned int i = 0; i < new_confGroup->getNumChildren(); i++)
    {
        confGroup->replaceChild(confGroup->getChild(i), new_confGroup->getChild(i));
    }
}
