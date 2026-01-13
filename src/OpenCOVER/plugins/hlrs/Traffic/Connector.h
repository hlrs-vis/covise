/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_CONNECTOR_H
#define OPENCOVER_PLUGINS_TRAFFIC_CONNECTOR_H

#include <vector>

struct SimulationState;

class Connector
{
public:
    bool update(double deltaTime, double simulationDeltaTime); // returns whether new data is there
    void getSimulationState(SimulationState &state);
    bool isConnected() const;
};

#endif
