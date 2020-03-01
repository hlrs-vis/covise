/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SUMOTRACI_H
#define SUMOTRACI_H

/****************************************************************************\ 
 **                                                            (C)2017 HLRS  **
 **                                                                          **
 ** Description: SumoTraCI - Traffic Control Interface client                **
 ** for traffic simulations with Sumo software - http://sumo.dlr.de          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>

#include <osg/ShapeDrawable>

#include <vector>
#include <random>
#include <TrafficSimulation/AgentVehicle.h>
#include <TrafficSimulation/PedestrianFactory.h>

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <QThreadPool>
#include <QElapsedTimer>

#include <algorithm>

#include "frameworkModules.h"
#include "configurationFiles.h"
#include "CoreFramework/CoreShare/callbacks.h"
#include "commandLineParser.h"
#include "configurationContainer.h"
#include "directories.h"
#include "frameworkModuleContainer.h"
#include "CoreFramework/CoreShare/log.h"
#include "runInstantiator.h"

namespace opencover
{
namespace ui {
class Slider;
class Label;
class Action;
class Button;
}
}

using namespace opencover;

class OpenPASS : public opencover::coVRPlugin , public ui::Owner
{
public:
    OpenPASS();
    ~OpenPASS();

    void preFrame();
    bool initConnection();

private:

    bool initUI();
    ui::Menu *openPASSMenu;
	ui::Button* pedestriansVisible;

    
        std::map<std::string, AgentVehicle *> vehicleMap;
        AgentVehicle *getAgentVehicle(const std::string &vehicleID, const std::string &vehicleClass, const std::string &vehicleType);

    osg::Group *vehicleGroup;
    std::string vehicleDirectory;

	osg::ref_ptr<osg::Switch> pedestrianGroup;
	osg::ref_ptr<osg::Switch> passengerGroup;
	osg::ref_ptr<osg::Switch> bicycleGroup;
	osg::ref_ptr<osg::Switch> busGroup;

    PedestrianFactory *pf;
    typedef std::map<std::string, coEntity *> EntityMap;
	EntityMap loadedEntities;
    
    PedestrianGeometry* createPedestrian(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID);
    double interpolateAngles(double lambda, double pastAngle, double futureAngle);
    //std::vector<pedestrianModel> pedestrianModels;
    //void getPedestriansFromConfig();
    void lineUpAllPedestrianModels();

    std::vector<std::string> vehicleClasses = {"passenger", "bus", "truck", "bicycle","escooter"};
    //std::map<std::string, std::vector<vehicleModel> *> vehicleModelMap;

    //void getVehiclesFromConfig();
    //void loadAllVehicles();
	bool connected;

};
#endif
