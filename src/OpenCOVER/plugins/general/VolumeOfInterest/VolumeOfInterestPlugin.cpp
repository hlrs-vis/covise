/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>

#include <cover/VRSceneGraph.h>

#include <osg/MatrixTransform>
#include <osg/Geode>

#include <PluginUtil/BoxSelection.h>

#include "VolumeOfInterestPlugin.h"
#include "VolumeOfInterestInteractor.h"
#include <cover/RenderObject.h>
#include <cover/ui/Button.h>

using namespace osg;

BoxSelection *VolumeOfInterestPlugin::s_boxSelection = NULL;

static VolumeOfInterestPlugin *plugin = NULL;

//-----------------------------------------------------------------------------

VolumeOfInterestPlugin::VolumeOfInterestPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("VolumeOfInterestPlugin", cover->ui)
, m_volumeOfInterestInteractor(new VolumeOfInterestInteractor(vrui::coInteraction::ButtonC, "BoxSelection", vrui::coInteraction::High))
, m_originalMatrix(osg::Matrix())
, m_destinationMatrix(osg::Matrix())
, m_originalScaleFactor(1.f)
, m_destinationScaleFactor(1.f)
, m_count(0)
, m_reset(false)
, m_volumeChanged(false)
, m_min(osg::Vec3(0, 0, 0))
, m_max(osg::Vec3(0, 0, 0))
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    new VolumeOfInterestPlugin\n");
    plugin = this;
}

bool VolumeOfInterestPlugin::init()
{
    m_volumeOfInterestInteractor->registerInteractionFinishedCallback(setToPreviousStateCallback);
    createMenuEntry();
    return true;
}

VolumeOfInterestPlugin::~VolumeOfInterestPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    delete VolumeOfInterestPlugin\n");

    deleteMenuEntry();
    m_volumeOfInterestInteractor->unregisterInteractionFinishedCallback();
    delete m_volumeOfInterestInteractor;
}

VolumeOfInterestPlugin *VolumeOfInterestPlugin::instance()
{
    return plugin;
}

void VolumeOfInterestPlugin::createMenuEntry()
{
    VolumeOfInterestPlugin::s_boxSelection = new BoxSelection(0, "Volume Of Interest", "Volume of Interest");
    VolumeOfInterestPlugin::s_boxSelection->registerInteractionFinishedCallback(defineVolumeCallback);
    s_boxSelection->getButton()->setCallback([this](bool state){
         if (state)
         {
            if (m_stateHistory.size() > 0)
                vrui::coInteractionManager::the()->registerInteraction(m_volumeOfInterestInteractor);
        }
        else
        {
            vrui::coInteractionManager::the()->unregisterInteraction(m_volumeOfInterestInteractor);
        }
    });
}

void VolumeOfInterestPlugin::deleteMenuEntry()
{
    delete VolumeOfInterestPlugin::s_boxSelection;
}

void VolumeOfInterestPlugin::preFrame()
{
    if (m_volumeChanged)
    {
        const int fps = 25;
        if (m_count < fps)
        {
            if (m_count == 0)
            {
                m_originalMatrix = VRSceneGraph::instance()->getTransform()->getMatrix();
                m_originalScaleFactor = VRSceneGraph::instance()->scaleFactor();
                if (!m_reset)
                {
                    MatrixState originalState;
                    originalState.matrix = m_originalMatrix;
                    originalState.scaleFactor = m_originalScaleFactor;
                    m_stateHistory.push_back(originalState);
                    vrui::coInteractionManager::the()->registerInteraction(m_volumeOfInterestInteractor);
                    BoundingBox bbox;
                    bbox.expandBy(m_min);
                    bbox.expandBy(m_max);
                    osg::BoundingSphere bsphere(bbox);
                    if (bsphere.radius() == 0.)
                        bsphere.radius() = 1.;
                    VRSceneGraph::instance()->boundingSphereToMatrices(bsphere, false,
                                                                       &m_destinationMatrix, &m_destinationScaleFactor);
                }
                else
                {
                    m_reset = false;
                }
            }
            float alpha = (m_count + 1.) / fps;
            setCurrentInterpolationState(alpha);
            ++m_count;
        }
        else
        {
            m_volumeChanged = false;
            m_count = 0;
        }
    }
}

void VolumeOfInterestPlugin::setCurrentInterpolationState(float alpha)
{
    cover->setScale(m_originalScaleFactor + (m_destinationScaleFactor - m_originalScaleFactor) * alpha);

    double *originalArray = m_originalMatrix.ptr();
    double *destinationArray = m_destinationMatrix.ptr();
    double *currentArray = new double[16];
    for (int i = 0; i < 16; ++i)
        currentArray[i] = originalArray[i] + (destinationArray[i] - originalArray[i]) * alpha;

    osg::Matrix currentMatrix(currentArray);
    cover->setXformMat(currentMatrix);
}

void VolumeOfInterestPlugin::defineVolumeCallback()
{
    VolumeOfInterestPlugin::instance()->defineVolume();
}

void VolumeOfInterestPlugin::defineVolume()
{
    float xmin, ymin, zmin;
    float xmax, ymax, zmax;
    VolumeOfInterestPlugin::s_boxSelection->getBox(xmin, ymin, zmin, xmax, ymax, zmax);
    printf("min: x: %f, y: %f, z: %f\n", xmin, ymin, zmin);
    printf("max: x: %f, y: %f, z: %f\n", xmax, ymax, zmax);
    m_min = osg::Vec3(xmin, ymin, zmin);
    m_max = osg::Vec3(xmax, ymax, zmax);

    m_volumeChanged = true;
}

void VolumeOfInterestPlugin::setToPreviousStateCallback()
{
    VolumeOfInterestPlugin::instance()->setToPreviousState();
}

void VolumeOfInterestPlugin::setToPreviousState()
{
    m_reset = true;
    m_volumeChanged = true;
    if (m_stateHistory.size() > 0)
    {
        const MatrixState previousState = m_stateHistory.back();
        m_stateHistory.pop_back();
        m_destinationMatrix = previousState.matrix;
        m_destinationScaleFactor = previousState.scaleFactor;
    }
    else
    {
        vrui::coInteractionManager::the()->unregisterInteraction(m_volumeOfInterestInteractor);
    }
}

COVERPLUGIN(VolumeOfInterestPlugin)
