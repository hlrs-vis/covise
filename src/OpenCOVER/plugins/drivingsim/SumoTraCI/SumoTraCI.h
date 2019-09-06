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

#include <utils/traci/TraCIAPI.h>
#include <libsumo/TraCIDefs.h>

#include <vector>
#include <random>
#include <TrafficSimulation/AgentVehicle.h>
#include <TrafficSimulation/PedestrianFactory.h>

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

struct simData
{
    osg::Vec3d position;
    double  angle;
    std::string vehicleClass;
    std::string vehicleType;
    std::string vehicleID;
};

struct pedestrianModel
{
    std::string fileName;
    double scale;
    pedestrianModel(std::string, double);
};

struct modalSplit
{
    ui::Slider* modalSplitSlider;
    ui::Button* modalSplitButton;
    std::string modalSplitString;
};

pedestrianModel::pedestrianModel(std::string n, double s) : fileName(n), scale(s) {}

struct vehicleModel
{
    std::string vehicleName;
    std::string fileName;
    vehicleModel(std::string, std::string);
    double scale=1;
    vehicleModel(std::string t, std::string n, double s);
};

vehicleModel::vehicleModel(std::string t, std::string n, double s) : vehicleName(t), fileName(n),scale(s) {}
vehicleModel::vehicleModel(std::string t, std::string n) : vehicleName(t), fileName(n) {}

class SumoTraCI : public opencover::coVRPlugin , public ui::Owner
{
public:
    SumoTraCI();
    ~SumoTraCI();

    void preFrame();
    bool init();

private:
    TraCIAPI client;

    bool initUI();
    bool compareTAZending(std::string& TAZ, std::string ending);
    ui::Menu *traciMenu;
    ui::Button *pedestriansVisible;
    ui::Button *pauseUI;
    ui::Button* turboUI;
    ui::Button *addTrafficUI;
    ui::Slider *trafficRateUI;

    //Modal split
    ui::Slider *modalSplitPassengerUI;
    ui::Button *fixModalSplitPassengerUI;
    ui::Slider *modalSplitBusUI;
    ui::Button *fixModalSplitBusUI;
    ui::Slider *modalSplitCyclingUI;
    ui::Button *fixModalSplitCyclingUI;
    ui::Slider *modalSplitWalkingUI;
    ui::Button *fixModalSplitWalkingUI;
    
    std::vector<modalSplit> modalSplits;
    //std::map<std::string, ui::Slider*> modalSplits;
    
    
    bool m_pedestrianVisible = true;
    void setPedestriansVisible(bool);

    libsumo::SubscriptionResults simResults;
    libsumo::SubscriptionResults pedestrianSimResults;
    std::vector<simData> currentResults;
    std::vector<simData> previousResults;
        std::map<std::string, AgentVehicle *> vehicleMap;
        AgentVehicle *getAgentVehicle(const std::string &vehicleID, const std::string &vehicleClass, const std::string &vehicleType);

    osg::Group *vehicleGroup;
    std::string vehicleDirectory;

    osg::ref_ptr<osg::Group> pedestrianGroup;

    PedestrianFactory *pf;
    typedef std::map<std::string, PedestrianGeometry *> PedestrianMap;
    PedestrianMap loadedPedestrians;
    
    PedestrianGeometry* createPedestrian(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID);
    double interpolateAngles(double lambda, double pastAngle, double futureAngle);
    std::vector<pedestrianModel> pedestrianModels;
    void getPedestriansFromConfig();
    void lineUpAllPedestrianModels();

    std::vector<std::string> vehicleClasses = {"passenger", "bus", "truck", "bicycle","escooter"};
    std::map<std::string, std::vector<vehicleModel> *> vehicleModelMap;

    void getVehiclesFromConfig();
    void loadAllVehicles();

    double simTime=0.0;
    double nextSimTime=0.0;
    double TimeOffset = 0.0;
    double startStep = 0.0;
    double previousTime=0.0;
    double currentTime=0.0;
    double framedt;
    double lastParticipantStartedTime;
    double deltaT = 1.0;
    int timeStep=0;
    std::vector<int> variables;
    std::map<const std::string, AgentVehicle *> loadedVehicles;

    int uniqueIDValue = 0;

    void subscribeToSimulation();
    void insertMissingEntities();
    AgentVehicle* createVehicle(const std::string &vehicleClass, const std::string &vehicleType, const std::string &vehicleID);
    void interpolateVehiclePosition();
    osg::Vec3d interpolatePositions(double lambda, osg::Vec3d pastPosition, osg::Vec3d futurePosition);
void sendSimResults();
void readSimResults();

bool balanceModalSplits();

std::vector<std::string> sourceTAZs;
std::vector<std::string> sinkTAZs;
};
#endif
