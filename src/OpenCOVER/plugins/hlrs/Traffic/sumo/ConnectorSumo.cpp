#include "sumo/ConnectorSumo.h"

#include <cover/coVRMSController.h>

#include "sumo/socket.h"
#include "Traffic.h"

static const std::vector<int> variables = {VAR_POSITION3D, VAR_SPEED, VAR_ANGLE, VAR_VEHICLECLASS};

ConnectorSumo::ConnectorSumo() {
    connect();
}

ConnectorSumo::~ConnectorSumo() {
    client.close();
}

bool ConnectorSumo::isConnected() const {
    return connected;
}

void ConnectorSumo::connect() {
    if (connected) {
        return;
    }

    if (!opencover::coVRMSController::instance()->isMaster()) {
        return;
    }

    try {
        std::cout << "[Traffic/ConnectorSumo] Connecting to [localhost:1337]..." << std::endl;
        client.connect("localhost", 1337);
        connected = true;

        std::cout << "[Traffic/ConnectorSumo] Connected." << std::endl;

        // identifiers: 57, 64, 67, 73, 79
        // subscribeToSimulation();
    } catch (tcpip::SocketException &e) {
        std::cerr << "[Traffic/ConnectorSumo] Could not connect to [localhost:1337], retry in 2.0 seconds: " << e.what() << std::endl;
        tryReconnect = 2.0;
    }
}

bool ConnectorSumo::update(double deltaTime, double simulationDeltaTime) {
    bool updated = false;

    if (!connected) {
        if (tryReconnect <= 0.0) {
            connect();
        } else {
            tryReconnect -= deltaTime;
        }
    }

    if (simulationDeltaTime > 0 && connected) {
        simulationTime += simulationDeltaTime;

        try {
            // Call simulationStep() as often as required to update to
            // simulationStateTime to reach simulationTime (just never too often to
            // pass it).
            while (simulationStateTime <= simulationTime - simulationStepSize) {
                client.simulationStep();
                simulationStateTime += simulationStepSize;
                updated = true;
            }
        } catch (tcpip::SocketException& e) {
            std::cerr << "[Traffic/ConnectorSumo] Disconnected with TCP exception: " << e.what() << std::endl;
            connected = false;
        } catch (libsumo::TraCIException& e) {
            std::cerr << "[Traffic/ConnectorSumo] Disconnected with TraCI exception: " << e.what() << std::endl;
            connected = false;
        }
    }

    return updated;
}

void  ConnectorSumo::getSimulationState(SimulationState& state) {
    state.vehicles.clear();

    subscribeToSimulation();
    auto vehicles = client.vehicle.getAllSubscriptionResults();
    // auto persons = client.person.getAllSubscriptionResults();
    //
    for (const auto &[key, item]: vehicles) {
        const auto& pos = std::dynamic_pointer_cast<const libsumo::TraCIPosition>(item.at(VAR_POSITION3D));
        const double angle = std::dynamic_pointer_cast<const libsumo::TraCIDouble>(item.at(VAR_ANGLE))->value;

        state.vehicles.push_back(VehicleState{
            key,
            std::dynamic_pointer_cast<const libsumo::TraCIString>(item.at(VAR_VEHICLECLASS))->value,
            osg::Vec3d(pos->x, pos->y, pos->z),
            M_PI_2-osg::DegreesToRadians(angle),
            });
    }
    //
    // for (const auto &[key, item]: persons) {
    //     const auto& pos = std::dynamic_pointer_cast<const libsumo::TraCIPosition>(item.at(VAR_POSITION3D));
    //
    //     state.vehicles.push_back(VehicleState{
    //         key,
    //         std::dynamic_pointer_cast<const libsumo::TraCIString>(item.at(VAR_VEHICLECLASS))->value,
    //         osg::Vec3d(pos->x, pos->y, pos->z),
    //         std::dynamic_pointer_cast<const libsumo::TraCIDouble>(item.at(VAR_ANGLE))->value
    //     });
    // }
}

void ConnectorSumo::subscribeToSimulation() {
    if(opencover::coVRMSController::instance()->isMaster() && connected) {
        try {
            if (client.simulation.getMinExpectedNumber() > 0) {
                std::vector<std::string> departedIDList = client.simulation.getDepartedIDList();
                for (auto& id : departedIDList) {
                    client.vehicle.subscribe(id, variables, 0, 1000000ll);
                }

                std::vector<std::string> personIDList = client.person.getIDList();
                for (auto& id : personIDList) {
                    client.person.subscribe(id, variables, 0, 1000000ll);
                }
            } else {
                std::cout << "[Traffic/ConnectorSumo] No vehicles in simulation." << std::endl;
            }
        } catch (tcpip::SocketException& e) {
            std::cerr << "[Traffic/ConnectorSumo] Disconnected with TCP exception: " << e.what() << std::endl;
            connected = false;
        } catch (libsumo::TraCIException& e) {
            std::cerr << "[Traffic/ConnectorSumo] Disconnected with TraCI exception: " << e.what() << std::endl;
            connected = false;
        }
    }
}
