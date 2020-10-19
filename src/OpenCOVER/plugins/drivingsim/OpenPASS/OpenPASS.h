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
#include <cover/ui/Menu.h>

#include <osg/ShapeDrawable>

#include <vector>
#include <random>

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

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>


#if defined(OpenPASS_EXPORTS)
#  define OPENPASS_EXPORT COEXPORT   //! Export of the dll-functions
#else
#  define OPENPASS_EXPORT COIMPORT   //! Import of the dll-functions
#endif

namespace opencover
{
namespace ui {
class Slider;
class Label;
class Action;
class Button;
}
}

namespace TrafficSimulation
{
    class AgentVehicle;
    class PedestrianFactory;
    class PedestrianGeometry;
    class coEntity;
}
namespace OpenScenario
{
    class OpenScenarioBase;
}
class OpenPASS; 
namespace SimulationSlave
{
    class RunInstantiator;
    class FrameworkModuleContainer;
}
struct FrameworkModules;
struct ConfigurationFiles;
class Directories;
namespace Configuration
{
    class ConfigurationContainer;
}
namespace SimulationCommon
{
    class Callbacks;
}

using namespace opencover;

struct pedestrianModel
{
    std::string fileName;
    double scale;
    pedestrianModel(std::string, double);
};

struct vehicleModel
{
    std::string vehicleName;
    std::string fileName;
    vehicleModel(std::string, std::string);
    double scale = 1;
    vehicleModel(std::string t, std::string n, double s);
};

class simOpenPASS : public OpenThreads::Thread
{
public:
    simOpenPASS(OpenPASS* op);
    ~simOpenPASS();

    virtual void run(); 

private:
    OpenPASS* openPass;
    SimulationSlave::RunInstantiator* runInstantiator;
    Directories* directories;
    FrameworkModules* frameworkModules;
    ConfigurationFiles* configurationFiles;
    Configuration::ConfigurationContainer* configurationContainer;
    SimulationCommon::Callbacks* callbacks;
    SimulationSlave::FrameworkModuleContainer* frameworkModuleContainer;
};

class agentInfo
{
public:
    agentInfo(TrafficSimulation::AgentVehicle *a);
    ~agentInfo();
    TrafficSimulation::AgentVehicle *vehicle;
    bool used;
};

class OPENPASS_EXPORT OpenPASS : public opencover::coVRPlugin , public ui::Owner
{
public:
    OpenPASS();
    ~OpenPASS();

    bool update();
    static OpenPASS* instance() { return plugin; };
    void updateAgent(AgentInterface* agent);
    void updateTime(int time, WorldInterface* world); // time in ms;
    void addAgent(AgentInterface* agent); 
    void postRun(const RunResultInterface& runResult);
    void loadXosc(std::string file);
    OpenThreads::Barrier barrier;
    OpenThreads::Barrier barrierContinue;
    OpenThreads::Barrier endBarrier;

private:
    static OpenPASS* plugin;
    int oldTime;
    bool restartSim = false;

    simOpenPASS* sim;
    std::string xodrDirectory;
    std::string xoscDirectory;

    bool initUI();
	ui::Menu* openPASSMenu;
    ui::Action* startSim;
    ui::Button* pedestriansVisible;


	std::map<std::string, TrafficSimulation::AgentVehicle*> vehicleMap;
	TrafficSimulation::AgentVehicle* getAgentVehicle(int ID, const std::string& vehicleClass, AgentVehicleType vehicleType);
	TrafficSimulation::AgentVehicle* createVehicle(int ID, const std::string& vehicleModelType, AgentVehicleType vehicleType);

	OpenScenario::OpenScenarioBase* osdb;

    osg::Group *vehicleGroup;
    std::string vehicleDirectory;

	osg::ref_ptr<osg::Switch> pedestrianGroup;
	osg::ref_ptr<osg::Switch> passengerGroup;
	osg::ref_ptr<osg::Switch> bicycleGroup;
    osg::ref_ptr<osg::Switch> motorbikeGroup;
    osg::ref_ptr<osg::Switch> truckGroup;
    osg::ref_ptr<osg::Switch> busGroup;
    osg::ref_ptr<osg::Switch> escooterGroup;

    TrafficSimulation::PedestrianFactory *pf;
    typedef std::map<std::string, TrafficSimulation::coEntity *> EntityMap;
	EntityMap loadedEntities;
    
    TrafficSimulation::PedestrianGeometry* createPedestrian(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID);
    double interpolateAngles(double lambda, double pastAngle, double futureAngle);
    //std::vector<pedestrianModel> pedestrianModels;
    void getPedestriansFromConfig();
    void lineUpAllPedestrianModels();

    std::vector<std::string> vehicleClasses = {"passenger", "pedestrian", "motorbike", "bicycle", "truck", "bus","escooter"};
    std::map<std::string, std::vector<vehicleModel>*> vehicleModelMap;
    std::map<int, agentInfo*> agentInfos;
    std::vector<pedestrianModel> pedestrianModels;

    void getVehiclesFromConfig();
    void loadAllVehicles();
    double currentTime;
    double previousTime;
    double framedt;

};
#endif
