/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_CONNECTORZMQ_H
#define OPENCOVER_PLUGINS_TRAFFIC_CONNECTORZMQ_H

#include <vector>
#include <zmq.hpp>

#include "Connector.h"

struct VehicleState;

class ConnectorZmq : public Connector
{
public:
    ConnectorZmq();

    void connect();
    bool update(double deltaTime, double simulationDeltaTime);
    void getSimulationState(SimulationState &state);

    bool isConnected() const;

protected:
    /**
     * Sends commands to the server to ensure the simulation state is the same
     * as the configuration demands.
     */
    void sendCommands();

private:
    zmq::context_t zmq_context;
    zmq::socket_t zmq_command_socket;
    zmq::socket_t zmq_subscriber_socket;

    std::vector<VehicleState> vehicles;
};
#endif
