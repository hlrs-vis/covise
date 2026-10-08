/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_CONNECTOR_TRAFFIC_RECORDING_H
#define OPENCOVER_PLUGINS_TRAFFIC_CONNECTOR_TRAFFIC_RECORDING_H

#include <map>
#include <set>
#include <string>

#include "Connector.h"

// warning: vibecode ahead
#include <stdexcept>
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>

struct VehicleState;

class ConnectorTrafficRecording : public Connector
{
public:
    ConnectorTrafficRecording(const std::string &filename);

    bool update(double deltaTime, double simulationDeltaTime) override;
    void getSimulationState(SimulationState &state) override;
    bool isConnected() const override;
    bool isPrerecorded() const override { return true; }

    std::vector<VehicleState> at(double timestamp);

    double getTimeStep() const override;
    double getAnimationSpeed() const override;

private:
    template <typename T>
    T read()
    {
        T value { };
        file_.read(reinterpret_cast<char *>(&value), sizeof(value));

        if (!file_)
            throw std::runtime_error("Unexpected end of FCD file");

        return value;
    }

    std::ifstream file_;

    std::uint32_t version_ { };
    std::uint32_t vehicle_count_ { };
    std::uint32_t timestep_count_ { };
    std::uint64_t index_offset_ { };

    std::vector<std::string> ids_;
    std::vector<std::uint64_t> offsets_;
    std::vector<double> times_;

    std::map<double, SimulationState> m_simulationStates;
    std::vector<double> m_timesteps;

    double m_simulationTime = -1.0;
    double m_timeStep = 1.0; // determined when loading, if possible
    bool m_updated = false;
};
#endif
