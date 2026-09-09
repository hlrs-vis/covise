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
#include "TmtEntity.h"

using namespace opencover;

Speaker::Speaker(const std::string &id)
    : TmtEntity(id)
{
    // Attach the speaker icon to the transform
    auto fileManager = coVRFileManager::instance();
    auto fileName = fileManager->getName("share/covise/icons/speaker.glb");
    if (fileName)
    {
        auto icon = coVRFileManager::instance()->loadFile(fileName, nullptr, transform, "", true);
        icon->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    }

    sensor = new SelectableSensor(&(AuralRealityPlugin::instance()->selection()), this, transform.get());
}

Speaker::~Speaker()
{
    delete sensor;
}

void Speaker::preFrame()
{
    sensor->update();

    if (checkTransformChanged())
    {
        AuralRealityPlugin::instance()->pushSpeaker(getId());
    }
}

void Speaker::updateSelection()
{
    showInteractor(isSelected());
}
