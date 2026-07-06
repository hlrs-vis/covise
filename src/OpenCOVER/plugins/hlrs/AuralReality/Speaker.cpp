/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Speaker.h"

#include <osg/Material>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>

#include "AuralRealityPlugin.h"

using namespace opencover;

Speaker::Speaker(const std::string &id)
    : id(id)
    , interactor(osg::Matrix::identity(), 1000.0, vrui::coInteraction::ButtonA, "hand", "speakerInteractor", vrui::coInteraction::Medium)
{
    interactor.hide();
    interactor.disableIntersection();

    offset.makeTranslate(0.0, 0.0, 0);
    offset_i.invert(offset);

    transform = new osg::MatrixTransform;
    cover->getObjectsRoot()->addChild(transform);

    auto transform2 = new osg::MatrixTransform;
    osg::Matrix m;
    // m.makeScale(0.001, 0.001, 0.001);
    transform2->setMatrix(m);
    transform->addChild(transform2);

    // Attach the speaker icon to the transform
    auto icon = coVRFileManager::instance()->loadFile("share/covise/icons/speaker.glb", nullptr, transform2, "", true);

    // osg::StateSet *ss = icon->getOrCreateStateSet();
    // ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
    //
    // osg::ref_ptr<osg::Material> mat = new osg::Material;
    // mat->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    // ss->setAttributeAndModes(mat, osg::StateAttribute::ON);

    icon->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));

    sensor = new SelectableSensor(&(AuralRealityPlugin::instance()->selection()), this, transform.get());
}

Speaker::~Speaker()
{
    delete sensor;
}

void Speaker::preFrame()
{
    sensor->update();
    interactor.preFrame();

    if (interactor.isRunning())
    {
        osg::Matrix m = offset_i * interactor.getMatrix();
        transform->setMatrix(m);

        AuralRealityPlugin::instance()->pushSpeaker(id);
    }
}

const std::string &Speaker::getId() const
{
    return id;
}

void Speaker::setTransform(osg::Matrix transform)
{
    this->transform->setMatrix(transform);
    interactor.updateTransform(offset * transform);
}
osg::Matrix Speaker::getTransform() const
{
    return transform->getMatrix();
}

void Speaker::setProperties(SpeakerProperties properties)
{
    this->properties = properties;
}
const SpeakerProperties &Speaker::getProperties() const
{
    return properties;
}

void Speaker::updateSelection()
{
    if (isSelected())
    {
        interactor.show();
        interactor.enableIntersection();
    }
    else
    {
        interactor.hide();
        interactor.disableIntersection();
    }
}
