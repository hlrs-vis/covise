/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2007 ZAIK  **
 **                                                                          **
 ** Description: Show Tracker Objects Plugin                                 **
 **         (shows an icon at the tracked object's position)                 **
 **         needed config entries to work:                                   **
 **          Cover.Plugin.ShowTrackerObjects "on" with following subitems:   **
 **          CheckStations: max number of stations to check for              **
 **                         config entries                                   **
 **          HandIcon: the icon file for the hand                            **
 **          HandIconSize: factor for the size of the hand icon              **
 **          ObjectsIcon: the icon file for the station which                **
 **          ObjectsIconSize: factor for the size of the objects icon        **
 **                          moves the object world                          **
 **          IconX:  Icon file for station X                                 **
**           IconSizeX: factor for the size of icon X                        **
 **                                                                          **
 ** Author: Hauke Fuehres                                                    **
 **                                                                          **
 \****************************************************************************/

#ifndef SHOWTRACKEROBJECTS_PLUGIN_H
#define SHOWTRACKEROBJECTS_PLUGIN_H
#include <map>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

typedef std::map<int, osg::ref_ptr<osg::MatrixTransform> > stationMatrixMap;
class ShowTrackerObjectsPlugin : public coVRPlugin
{
public:
    ShowTrackerObjectsPlugin();
    ~ShowTrackerObjectsPlugin();
    bool init();

    void preFrame();

private:
    //maximum number of Stations to be checked
    int checkStationNumbers;
    //stores an array with references to the different trackericons
    stationMatrixMap trackerPosIcon;
    int handStationNr;
    int objectsStationNr;
    std::string objectsIcon;
    std::string handIcon;
    //loads an icon file
    osg::MatrixTransform *loadTrackerPosIcon(const char *name, float s);
};
#endif
