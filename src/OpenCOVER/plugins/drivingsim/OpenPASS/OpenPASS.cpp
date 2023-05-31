/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include "OpenPASS.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>
#include <TrafficSimulation/Vehicle.h>
#include <TrafficSimulation/CarGeometry.h>
#include <net/tokenbuffer.h>
#include <util/unixcompat.h>
#include <util/threadname.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <osg/Switch>

#include <TrafficSimulation/Vehicle.h>
#include <TrafficSimulation/AgentVehicle.h>
#include <TrafficSimulation/PedestrianFactory.h>
#include <OpenScenario/OpenScenarioBase.h>
#include <OpenScenario/schema/oscFileHeader.h>

///////////OpenPASS
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <QThreadPool>
#include <QElapsedTimer>

#include <algorithm>

#include "frameworkModules.h"
#include "configurationFiles.h"
#include "callbacks.h"
#include "commandLineParser.h"
#include "configurationContainer.h"
#include "directories.h"
#include "frameworkModuleContainer.h"
#include "log.h"
#include "runInstantiator.h"
#include <cover/coVRFileManager.h>
#include "egoAgentInterface.h"

int gPrecision;

using namespace opencover;
using namespace core;
using namespace openpass::core;

//-----------------------------------------------------------------------------
//! \brief SetupLogging Sets up the logging and print pending (buffered)
//! 	   logging messages
//! \param[in] logLevel 0 (none) to 5 (debug core)
//! \param[in] logFile  name of the logfile
//! \param[in] bufferedMessages messages recorded before creation of the logfile
//-----------------------------------------------------------------------------
static void SetupLogging(LogLevel logLevel, const std::string& logFile, const std::list<std::string>& bufferedMessages);

//-----------------------------------------------------------------------------
//! \brief  Check several parameters of a readable directory
//! \param  directory
//! \param  prefix   for readable logging in case of an error
//! \return true, if everything is allright
//-----------------------------------------------------------------------------
static bool CheckReadableDir(const std::string& directory, const std::string& prefix);

//-----------------------------------------------------------------------------
//! \brief  Check several parameters of a writeable directory
//! \param  directory
//! \param  prefix   for readable logging in case of an error
//! \return true, if everything is allright
//-----------------------------------------------------------------------------
static bool CheckWritableDir(const std::string& directory, const std::string& prefix);

//-----------------------------------------------------------------------------
//! \brief   CheckDirectories Checks the directories given by the command line parameters
//! \param   directories
//! \return  true, if everything is allright
//-----------------------------------------------------------------------------
static bool CheckDirectories(const Directories* directories);


agentInfo::agentInfo(TrafficSimulation::AgentVehicle* a)
{
	vehicle = a;
	used = true;
};
agentInfo::~agentInfo()
{
	delete vehicle;
};

openpass::common::RuntimeInformation runtimeInformation
{ {openpass::common::framework}, { "", "", "" } };;

simOpenPASS::simOpenPASS(OpenPASS* op)
{
    openPass = op;

    char* opHome = getenv("OPENPASS_HOME");
    if (opHome == nullptr)
    {
        opHome = "c:/src/simopenpass/build/OpenPASS";
    }
    char* covisedir = getenv("COVISEDIR");
    if (covisedir == nullptr)
    {
        covisedir = "c:/src/covise";
    }
    char* archsuffix = getenv("ARCHSUFFIX");
    if (archsuffix == nullptr)
    {
        archsuffix = "zebuopt";
    }

    int logLevel = 5;
    directories = new Directories(opHome,
        "lib",
        "configs",
        "results");
    if (!CheckDirectories(directories))
    {
        std::cerr << "CheckDirectories failed" << endl;
    }

    configurationFiles = new ConfigurationFiles( directories->configurationDir, "systemConfigBlueprint.xml", "slaveConfig.xml" );

    runtimeInformation.versions.framework = openpass::common::framework;
    runtimeInformation.directories.configuration = directories->configurationDir;
    runtimeInformation.directories.library = directories->outputDir;
    runtimeInformation.directories.output = directories->libraryDir;


    configurationContainer = new Configuration::ConfigurationContainer(*configurationFiles, runtimeInformation);
   
    if (!configurationContainer->ImportAllConfigurations())
    {
        std::cerr << "Failed to import all configurations";
        exit(EXIT_FAILURE);
    }

    auto& libraries = configurationContainer->GetSimulationConfig()->GetExperimentConfig().libraries;
    libraries["ObservationLibrary"] = "observationCOVER";
    ObservationInstanceCollection& oic = configurationContainer->GetSimulationConfig()->GetObservationConfig();
    ObservationInstance oi;
    oi.id = oic.size();
    oi.libraryName = std::string(covisedir) + "/" + archsuffix + "/lib/OpenCOVER/plugins/observationCOVER";
    oic.push_back(oi);

    frameworkModules = new FrameworkModules(
        logLevel,
        directories->libraryDir,
        libraries.at("DataStoreLibrary"),
        libraries.at("EventDetectorLibrary"),
        libraries.at("ManipulatorLibrary"),
        oic,
        libraries.at("StochasticsLibrary"),
        libraries.at("WorldLibrary"),
        configurationContainer->GetSimulationConfig()->GetSpawnPointsConfig()
    );
    callbacks = new SimulationCommon::Callbacks;
    frameworkModuleContainer = new core::FrameworkModuleContainer(*frameworkModules, configurationContainer, runtimeInformation, callbacks);

    configurationContainer->GetSimulationConfig()->GetExperimentConfig().numberOfInvocations = 100000000;
    openPass->loadXosc(configurationContainer->GetSimulationConfig()->GetScenarioConfig().scenarioPath.c_str());
    runInstantiator = nullptr;



}
simOpenPASS::~simOpenPASS()
{
    delete frameworkModuleContainer;
    delete callbacks;
    delete configurationContainer;
    delete configurationFiles;
    delete frameworkModules;
    delete directories;
    delete runInstantiator;
}

void simOpenPASS::run()
{
#ifdef WIN32
    HRESULT r;
    r = SetThreadDescription(
        GetCurrentThread(),
        L"simOpenPASSThread!"
    );
#else
    setThreadName("simOpenPASSThread");
#endif
    delete runInstantiator;
    runInstantiator = new RunInstantiator(*configurationContainer,
        *frameworkModuleContainer,
        *frameworkModules);

    if (runInstantiator->ExecuteRun())
    {
        LOG_INTERN(LogLevel::DebugCore) << "simulation finished successfully";
    }
    else
    {
        LOG_INTERN(LogLevel::Error) << "simulation failed";
    }
    openPass->barrier.release();
    openPass->barrierContinue.release();
    openPass->endBarrier.block(2);
}
OpenPASS::OpenPASS() : ui::Owner("OpenPASS", cover->ui)
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "OpenPASS::OpenPASS\n");
    plugin = this;
    initUI();
    const char *coviseDir = getenv("COVISEDIR");
    std::string defaultDir = std::string(coviseDir) + "/share/covise/vehicles";
    vehicleDirectory = covise::coCoviseConfig::getEntry("value","COVER.Plugin.OpenPASS.VehicleDirectory", defaultDir.c_str());
    //AgentVehicle *av = getAgentVehicle("veh_passenger","passenger","veh_passenger");
    //av = getAgentVehicle("truck","truck","truck_truck");
    pf = TrafficSimulation::PedestrianFactory::Instance();
    pedestrianGroup =  new osg::Switch;
    pedestrianGroup->setName("pedestrianGroup");
	cover->getObjectsRoot()->addChild(pedestrianGroup.get());
	passengerGroup = new osg::Switch;
	passengerGroup->setName("passengerGroup");
	cover->getObjectsRoot()->addChild(passengerGroup.get());
	bicycleGroup = new osg::Switch;
	bicycleGroup->setName("bicycleGroup");
	cover->getObjectsRoot()->addChild(bicycleGroup.get());
    motorbikeGroup = new osg::Switch;
    motorbikeGroup->setName("motorbikeGroup");
	cover->getObjectsRoot()->addChild(motorbikeGroup.get());
    truckGroup = new osg::Switch;
    truckGroup->setName("truckGroup");
    cover->getObjectsRoot()->addChild(truckGroup.get());
    busGroup = new osg::Switch;
    busGroup->setName("busGroup");
    cover->getObjectsRoot()->addChild(busGroup.get());
    escooterGroup = new osg::Switch;
    escooterGroup->setName("escooterGroup");
    cover->getObjectsRoot()->addChild(escooterGroup.get());
    pedestrianGroup->setNodeMask(pedestrianGroup->getNodeMask() & ~Isect::Update); // don't use the update traversal, tey are updated manually when in range
    pedestrianGroup->setAllChildrenOn();
    passengerGroup->setAllChildrenOn();
    bicycleGroup->setAllChildrenOn();
    motorbikeGroup->setAllChildrenOn();
    truckGroup->setAllChildrenOn();
    busGroup->setAllChildrenOn();
    escooterGroup->setAllChildrenOn();
	
    getPedestriansFromConfig();
    getVehiclesFromConfig();
    loadAllVehicles();
    lineUpAllPedestrianModels(); // preload pedestrians;


    osdb = new OpenScenario::OpenScenarioBase();
    osdb->setFullReadCatalogs(false);


    sim = new simOpenPASS(this);
    sim->start();
    oldTime = -1;

}


void OpenPASS::loadXosc(std::string file)
{
    osdb->setValidation(false); // don't validate, we might be on the road where we don't have axxess to the schema files
    if (osdb->loadFile(file, "OpenSCENARIO", "OpenSCENARIO") == false)
    {
        std::cerr << std::endl;
        std::cerr << "failed to load OpenSCENARIO from file " << file << std::endl;
        std::cerr << std::endl;
        delete osdb;
        osdb = nullptr;
    }

    //neu
    std::string filename(file);
    xoscDirectory.clear();
    if (filename[0] != '/' && filename[0] != '\\' && (!(filename[1] == ':' && (filename[2] == '/' || filename[2] == '\\'))))
    { // / or backslash or c:/
        char* workingDir = getcwd(NULL, 0);
        xoscDirectory.assign(workingDir);
        free(workingDir);
    }
    size_t lastSlashPos = filename.find_last_of('/');
    size_t lastSlashPos2 = filename.find_last_of('\\');
    if (lastSlashPos != filename.npos && (lastSlashPos2 == filename.npos || lastSlashPos2 < lastSlashPos))
    {
        if (!xoscDirectory.empty())
            xoscDirectory += "/";
        xoscDirectory.append(filename, 0, lastSlashPos);
    }
    if (lastSlashPos2 != filename.npos && (lastSlashPos == filename.npos || lastSlashPos < lastSlashPos2))
    {
        if (!xoscDirectory.empty())
            xoscDirectory += "\\";
        xoscDirectory.append(filename, 0, lastSlashPos2);
    }
    if (osdb->FileHeader.exists())
    {
        for (auto it = osdb->FileHeader->UserData.begin(); it != osdb->FileHeader->UserData.end(); it++)
        {

            
        }
    }
   

    //load xodr
    if (osdb->RoadNetwork.getObject() != NULL)
    {
        if (osdb->RoadNetwork->Logics.exists())
        {
            std::string xodrName_st = osdb->RoadNetwork->Logics->filepath.getValue();
            //loadRoadSystem(xodrName_st.c_str());
            coVRFileManager::instance()->loadFile((xoscDirectory+"/"+xodrName_st).c_str());
        }
        else if (osdb->RoadNetwork->LogicFile.exists())
        {
            std::string xodrName_st = osdb->RoadNetwork->LogicFile->filepath.getValue();
            //loadRoadSystem(xodrName_st.c_str());
            coVRFileManager::instance()->loadFile((xoscDirectory + "/" + xodrName_st).c_str());
        }
        if (osdb->RoadNetwork->SceneGraph.exists())
        {
            std::string geometryFile = osdb->RoadNetwork->SceneGraph->filepath.getValue();
            //load ScenGraph
            coVRFileManager::instance()->loadFile(geometryFile.c_str());
        }
    }
}

OpenPASS* OpenPASS::plugin=nullptr;


OpenPASS::~OpenPASS()
{
    fprintf(stderr, "OpenPASS::~OpenPASS\n");
    barrier.release();
    barrierContinue.release();
    sim->cancel();
    endBarrier.block(2);
    delete sim;
}

void OpenPASS::updateAgent(AgentInterface* agent)
{
    int ID = agent->GetId();
    agentInfo* ai = agentInfos[ID];
    if (ai == nullptr)
    {
        agentInfos[ID] = ai = new agentInfo(createVehicle(ID, agent->GetVehicleModelType(), agent->GetVehicleModelParameters().vehicleType));
    }
    if (ai != nullptr)
    {
        TrafficSimulation::AgentVehicle* av = ai->vehicle;
        ai->used = true;
        osg::Matrix rotOffset;
        rotOffset.makeIdentity();
        //fprintf(stderr, "%d: %f %f %f\n", ID, agent->GetPositionX(), agent->GetPositionY(), agent->GetRelativeYaw());
        osg::Matrix rmat, tmat;
        osg::Vec3 pos(agent->GetPositionX(), agent->GetPositionY(), 0);
        RoadPosition rp = agent->GetEgoAgent().GetMainLocatePosition().roadPosition;
        double ld = agent->GetEgoAgent().GetLaneDirection(0);
        osg::Quat orientation(-agent->GetEgoAgent().GetRelativeYaw()- ld, osg::Vec3d(0, 0, -1));
        rmat.makeRotate(orientation);
        tmat.makeTranslate(pos);
        av->setTransform(rotOffset * rmat * tmat);
        TrafficSimulation::VehicleState vs;
        vs.du = agent->GetVelocity();
        av->getCarGeometry()->updateCarParts(1, framedt, vs);
    }

}
void OpenPASS::updateTime(int time, WorldInterface* world) // tie in ms
{
    barrier.block(2);

    for (auto const& [key, val] : agentInfos)
    {
        val->used = false;
    }
    for (const auto& it : world->GetAgents()) {
       updateAgent(it.second);
    }

    for (auto it = agentInfos.begin(); it != agentInfos.end(); /* no increment */)
    {
        if (!it->second->used)
        {
            delete it->second;
            it = agentInfos.erase(it); 
        }
        else
        {
            ++it;
        }
    }
    barrierContinue.block(2);

}
void OpenPASS::lineUpAllPedestrianModels()
{
    /*for (int i = 0; i < pedestrianModels.size(); i++)
    {
        std::string ID = "pedestrianTest" + std::to_string(i);
        PedestrianAnimations a = PedestrianAnimations();

        int pedestrianIndex = i;

        pedestrianModel p = pedestrianModels[pedestrianIndex];
        PedestrianGeometry* pedgeom = new PedestrianGeometry(ID, p.fileName, p.scale, 40.0, a, pedestrianGroup);
        osg::Vec3d position = osg::Vec3d((double)i, 0.0, 50.0);

        osg::Quat orientation(osg::DegreesToRadians(0.0), osg::Vec3d(0, 0, -1));

        Transform trans = Transform(Vector3D(position.x(), position.y(), position.z()), Quaternion(orientation.w(), orientation.x(), orientation.y(), orientation.z()));
        pedgeom->setTransform(trans, M_PI);
    }*/
}

bool OpenPASS::initUI()
{
    openPASSMenu = new ui::Menu("OpenPASS", this);
    startSim = new ui::Action(openPASSMenu, "startSim");
    startSim->setCallback([this]() {
        if (!sim->isRunning())
        {
            sim->start();
        }
		});
    
    return true;
}
void OpenPASS::addAgent(AgentInterface* agent)
{
    //fprintf(stderr, "newAgent %d\n", agent->GetId());
}
void OpenPASS::postRun(const RunResultInterface& runResult)
{
    restartSim = true;
}

TrafficSimulation::AgentVehicle* OpenPASS::createVehicle(int ID, const std::string& vehicleModelType, AgentVehicleType vehicleType)
{
    TrafficSimulation::AgentVehicle* av = getAgentVehicle(ID, vehicleModelType, vehicleType);

    TrafficSimulation::VehicleParameters vp;
   /* if (vehicleType )
    {
        vp.rangeLOD = 400;
    }
    else*/
    {
        vp.rangeLOD = 1000;
    }
    std::string vehicleID = std::to_string(ID);
    return new TrafficSimulation::AgentVehicle(av, vehicleID, vp, NULL, 0.0, 0);
}

TrafficSimulation::AgentVehicle* OpenPASS::getAgentVehicle(int ID, const std::string& vehicleModelType, AgentVehicleType vehicleType)
{


    TrafficSimulation::AgentVehicle* av;
    auto avIt = vehicleMap.find(vehicleModelType);
    if (avIt != vehicleMap.end())
    {
        av = avIt->second;
    }
    else
    {
        auto vehiclesPair = vehicleModelMap.find(vehicleClasses[int(vehicleType)]);
        auto vehicles = vehiclesPair->second;
        int vehicleIndex = -1;
        for (int i = 0; i < vehicles->size(); i++)
        {
            if (vehicles->at(i).vehicleName == vehicleModelType)
            {
                vehicleIndex = i;
                break;
            }
        }
        if (vehicleIndex == -1)
        {

            auto avIt = vehicleMap.find(vehicles->at(0).vehicleName);
            if (avIt != vehicleMap.end())
            {
                return avIt->second;
            }
            vehicleIndex = 0;
        }

        osg::Group* parent = nullptr;
        if (vehicleType == AgentVehicleType::Truck)
            parent = truckGroup;
        if (vehicleType == AgentVehicleType::Bicycle)
            parent = bicycleGroup;
        if (vehicleType == AgentVehicleType::Motorbike)
            parent = motorbikeGroup;
        if (vehicleType == AgentVehicleType::Pedestrian)
            parent = pedestrianGroup;
        if (vehicleType == AgentVehicleType::Car)
            parent = passengerGroup;
        std::string vehicleID = std::to_string(ID);
        av = new TrafficSimulation::AgentVehicle(vehicleID, new TrafficSimulation::CarGeometry(vehicleID, vehicles->at(vehicleIndex).fileName, false, parent), 0, NULL, 0, 1, 0.0, 1);
        vehicleMap.insert(std::pair<std::string, TrafficSimulation::AgentVehicle*>(vehicles->at(vehicleIndex).vehicleName, av));
    }
    return av;
}
bool OpenPASS::update()
{
    previousTime = currentTime;
    currentTime = cover->frameTime();
    framedt = currentTime - previousTime;
    if(sim->isRunning())
    {
        if (restartSim)
        {
            for (auto const& [key, val] : agentInfos)
            {
                delete val;
            }
            agentInfos.clear();
            restartSim = false;
        }
		barrier.block(2);
        barrierContinue.block(2);
    }
    return true;
}

void SetupLogging(LogLevel logLevel, const std::string& logFile, const std::list<std::string>& bufferedMessages)
{
    QDir logFilePath(QString::fromStdString(logFile));
    LogOutputPolicy::SetFile(logFilePath.absolutePath().toStdString());

    LogFile::ReportingLevel() = logLevel;
    LOG_INTERN(LogLevel::DebugCore) << std::endl << std::endl << "### slave start ##";

    // delayed logging, since the log file was unknown before
    std::for_each(bufferedMessages.begin(), bufferedMessages.end(),
        [](const std::string& m) { LOG_INTERN(LogLevel::Warning) << m; });
}

bool CheckDirectories(const Directories* directories)
{
    LOG_INTERN(LogLevel::DebugCore) << "base path: " << directories->baseDir;
    LOG_INTERN(LogLevel::DebugCore) << "library path: " << directories->libraryDir;
    LOG_INTERN(LogLevel::DebugCore) << "configuration path: " << directories->configurationDir;
    LOG_INTERN(LogLevel::DebugCore) << "output path: " << directories->outputDir;

    if (!QDir(QString::fromStdString(directories->outputDir)).exists())
    {
        LOG_INTERN(LogLevel::DebugCore) << "create output folder " + directories->outputDir;
        QDir().mkpath(QString::fromStdString(directories->outputDir));
    }

    auto status_library = CheckReadableDir(directories->libraryDir, "Library");
    auto status_config = CheckReadableDir(directories->configurationDir, "Configuration");
    auto status_output = CheckWritableDir(directories->outputDir, "Output");

    // return after probing, so all directories are reported at once
    return status_config && status_library && status_output;
}

bool CheckReadableDir(const std::string& directory, const std::string& prefix)
{
    const QFileInfo dir(QString::fromStdString(directory));
    if (!dir.exists() || !dir.isDir() || !dir.isReadable())
    {
        LOG_INTERN(LogLevel::Error) << prefix << " directory "
            << directory << " does not exist, is not a directory, or is not readable";
        return false;
    }
    return true;
}

bool CheckWritableDir(const std::string& directory, const std::string& prefix)
{
    const QFileInfo dir(QString::fromStdString(directory));
    if (!dir.exists() || !dir.isDir() || !dir.isWritable())
    {
        LOG_INTERN(LogLevel::Error) << prefix << " directory "
            << directory << " does not exist, is not a directory, or is not writable";
        return false;
    }
    return true;
}

pedestrianModel::pedestrianModel(std::string n, double s) : fileName(n), scale(s) {}
vehicleModel::vehicleModel(std::string t, std::string n, double s) : vehicleName(t), fileName(n), scale(s) {}
vehicleModel::vehicleModel(std::string t, std::string n) : vehicleName(t), fileName(n) {}

void OpenPASS::getPedestriansFromConfig()
{
    covise::coCoviseConfig::ScopeEntries e = covise::coCoviseConfig::getScopeEntries("COVER.Plugin.SumoTraCI.Pedestrians");
    const char** entries = e.getValue();
    if (entries)
    {
        while (*entries)
        {
            const char* fileName = *entries;
            entries++;
            const char* scaleChar = *entries;
            entries++;
            double scale = atof(scaleChar);
            pedestrianModels.push_back(pedestrianModel(fileName, scale));
        }
    }
}

void OpenPASS::getVehiclesFromConfig()
{
    for (std::vector<std::string>::iterator itr = vehicleClasses.begin(); itr != vehicleClasses.end(); itr++)
    {
        std::vector<vehicleModel>* vehicles = new std::vector<vehicleModel>;
        vehicleModelMap[*itr] = vehicles;
        covise::coCoviseConfig::ScopeEntries e = covise::coCoviseConfig::getScopeEntries("COVER.Plugin.SumoTraCI.Vehicles." + *itr);
        const char** entries = e.getValue();
        if (entries)
        {
            while (*entries)
            {
                const char* vehicleName = *entries;
                entries++;
                const char* fileName = *entries;
                entries++;
                vehicleModel m = vehicleModel(vehicleName, fileName);
                vehicles->push_back(m);
            }
        }
        if (vehicles->size() == 0)
        {
            fprintf(stderr, "please add vehicle config %s\n", ("COVER.Plugin.SumoTraCI.Vehicles." + *itr).c_str());
        }
    }
}

void OpenPASS::loadAllVehicles()
{
    for (auto itr = vehicleModelMap.begin(); itr != vehicleModelMap.end(); itr++)
    {
        auto vehicles = itr->second;
        auto vehicleClass = itr->first;
        for (auto itr2 = vehicles->begin(); itr2 != vehicles->end(); itr2++)
        {
            osg::Group* parent = nullptr;
            if (vehicleClass == "bus")
                parent = busGroup;
            if (vehicleClass == "passenger")
                parent = passengerGroup;
            if (vehicleClass == "bicycle")
                parent = bicycleGroup;
            if (vehicleClass == "bicycle")
                parent = escooterGroup;
            if (vehicleClass == "motorbike")
                parent = motorbikeGroup;
            if (vehicleClass == "truck")
                parent = truckGroup;
            TrafficSimulation::AgentVehicle* av = new TrafficSimulation::AgentVehicle("test1", new TrafficSimulation::CarGeometry("test2", itr2->fileName, false, parent), 0, NULL, 0, 1, 0.0, 1);
            vehicleMap.insert(std::pair<std::string, TrafficSimulation::AgentVehicle*>(itr2->vehicleName, av));
        }
    }
}
COVERPLUGIN(OpenPASS)
