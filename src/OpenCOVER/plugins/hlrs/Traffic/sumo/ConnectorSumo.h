/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_CONNECTORSUMO_H
#define OPENCOVER_PLUGINS_TRAFFIC_CONNECTORSUMO_H

#include <vector>

#include "sumo/TraCIAPI.h"
#include "sumo/TraCIDefs.h"

struct SimulationState;

class Connector {
public:
    bool update(double deltaTime, double simulationDeltaTime); // returns whether new data is there
    void getSimulationState(SimulationState& state);
    bool isConnected() const;
};

class ConnectorSumo : public Connector {
  public:
    ConnectorSumo();
    ~ConnectorSumo();

    void connect();
    void subscribeToSimulation();
    bool update(double deltaTime, double simulationDeltaTime);
    void getSimulationState(SimulationState& state);

    bool isConnected() const;


  private:
    bool connected = false;
    double tryReconnect = 0.0;

    double simulationTime = 0.0;        // Desired target time
    double simulationStateTime = 0.0;   // SUMO state time
    double simulationStepSize = 0.2;    // TODO: fetch from sumo, if possible, or read from config

    TraCIAPI client;
};
#endif
