/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2007 ZAIK  **
 **                                                                          **
 ** Description: Show Tracker Objects Plugin                                 **
 **         (shows an icon at the tracked object's position)                 **
 **                                                                          **
 ** Author: Hauke Fuehres                                                    **
 **                                                                          **
 \****************************************************************************/

#include <config/CoviseConfig.h>
#include "ShowTrackerObjectsPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/input/input.h>
#include <cover/coVRFileManager.h>
#include <osg/MatrixTransform>

ShowTrackerObjectsPlugin::ShowTrackerObjectsPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool ShowTrackerObjectsPlugin::init()
{
    if (!coCoviseConfig::isOn("COVER.Plugin.ShowTrackerObjects", false))
        return true;

    //check config for number of TrackerIcons to be checked
    checkStationNumbers = coCoviseConfig::getInt("COVER.Plugin.ShowTrackerObjects.CheckStations", 16);
    //check different stations w/ coresponding icons in config
    for (int i = 0; i <= checkStationNumbers; i++)
    {
        char stationEntry[255];
        sprintf(stationEntry, "COVER.Plugin.ShowTrackerObjects.Icon%i", i);
        std::string stationIcon = coCoviseConfig::getEntry(stationEntry);
        if (!stationIcon.empty())
        {
            char sizeEntry[255];
            sprintf(sizeEntry, "IconSize%i", i);
            float size = coCoviseConfig::getFloat(sizeEntry, "COVER.Plugin.ShowTrackerObjects", 1.0);
            trackerPosIcon[i] = loadTrackerPosIcon(stationIcon.c_str(), size);
            trackerPosIcon[i]->setNodeMask(trackerPosIcon[i]->getNodeMask() & ~Isect::Intersection);
            //Add Icon to Scene Graph
            cover->getScene()->addChild(trackerPosIcon[i].get());
            if (cover->debugLevel(4))
                fprintf(stderr, "ShowTrackerObjects::loadIcon%i\n", i);
        }
    }
    handIcon = coCoviseConfig::getEntry("COVER.Plugin.ShowTrackerObjects.HandIcon");
    float size;
    if (!handIcon.empty())
    {
        //vld1: VRTracker usage. Hand ststion number
        //handStationNr = VRTracker::instance()->getHandSensorStation();
        if (trackerPosIcon.find(handStationNr) != trackerPosIcon.end())
        {
            fprintf(stderr, "Warning ShowTrackerObjects:: Hand Icon already defined by Station Number %i\n",
                    handStationNr);
        }
        else
        {
            size = coCoviseConfig::getFloat("COVER.Plugin.ShowTrackerObjects.HandIconSize", 1.0);
            trackerPosIcon[handStationNr] = loadTrackerPosIcon(handIcon.c_str(), size);
            //no intersection with tracker Position Icon
            trackerPosIcon[handStationNr]->setNodeMask(trackerPosIcon[handStationNr]->getNodeMask() & ~Isect::Intersection);
            //Add Icon to Scene Graph
            cover->getScene()->addChild(trackerPosIcon[handStationNr].get());
            if (cover->debugLevel(4))
                fprintf(stderr, "ShowTrackerObjects::loadHandIcon\n");
        }
    }
    objectsIcon = coCoviseConfig::getEntry("COVER.Plugin.ShowTrackerObjects.ObjectsIcon");
    if (!objectsIcon.empty())
    {
        //vld1: VRTracker usage. World Station number
        //objectsStationNr = VRTracker::instance()->getWorldSensorStation();
        if (trackerPosIcon.find(objectsStationNr) != trackerPosIcon.end())
        {
            fprintf(stderr, "Warning ShowTrackerObjects:: Objects Icon already defined by Station Number %i\n",
                    objectsStationNr);
        }
        else
        {
            size = coCoviseConfig::getFloat("COVER.Plugin.ShowTrackerObjects.ObjectsIconSize", 1.0);
            trackerPosIcon[objectsStationNr] = loadTrackerPosIcon(objectsIcon.c_str(), size);
            //no intersection with tracker Position Icon
            trackerPosIcon[objectsStationNr]->setNodeMask(trackerPosIcon[objectsStationNr]->getNodeMask() & ~Isect::Intersection);
            //Add Icon to Scene Graph
            cover->getScene()->addChild(trackerPosIcon[objectsStationNr].get());
            if (cover->debugLevel(4))
                fprintf(stderr, "ShowTrackerObjects::loadObjectsIcon\n");
        }
    }

    return true;
}

ShowTrackerObjectsPlugin::~ShowTrackerObjectsPlugin()
{
    //delete the tracker Icons from the scenegraph
    stationMatrixMap::iterator iter;
    for (iter = trackerPosIcon.begin(); iter != trackerPosIcon.end(); iter++)
    {
        while (iter->second->getNumParents() > 0)
            iter->second->getParent(0)->removeChild(iter->second.get());
    }
}

void
ShowTrackerObjectsPlugin::preFrame()
{
    //update the position of the icons
    stationMatrixMap::iterator iter;
    for (iter = trackerPosIcon.begin(); iter != trackerPosIcon.end(); iter++)
    {
        //vld1: VRTracker usage: getStationMat
        //dynamic_cast<osg::MatrixTransform *>( iter->second->getChild(0))->setMatrix(VRTracker::instance()->getStationMat(iter->first));
    }
}

osg::MatrixTransform *
ShowTrackerObjectsPlugin::loadTrackerPosIcon(const char *name, float s)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "ShowTrackerObjects::loadTrackerPosIcon\n");
    osg::MatrixTransform *mt = new osg::MatrixTransform;
    osg::MatrixTransform *posTrans = new osg::MatrixTransform;
    osg::Node *icon;
    icon = coVRFileManager::instance()->loadIcon(name);
    if (icon == NULL)
    {
        fprintf(stderr, "failed to load TrackerObject icon: %s\n", name);
        icon = coVRFileManager::instance()->loadIcon("hlrsIcon");
    }
    posTrans->addChild(icon);
    mt->addChild(posTrans);
    mt->setMatrix(osg::Matrix::scale(s, s, s));

    return (mt);
}

COVERPLUGIN(ShowTrackerObjectsPlugin)
