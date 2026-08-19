/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENCOVER_PLUGINS_TRAFFIC_CONNECTORFCD_H
#define OPENCOVER_PLUGINS_TRAFFIC_CONNECTORFCD_H

#include <map>
#include <set>
#include <string>

#include <xercesc/sax2/DefaultHandler.hpp>

#include "Connector.h"

class ConnectorFcd : public Connector, public xercesc::DefaultHandler
{
public:
    ConnectorFcd(const std::string &filename);
    ~ConnectorFcd();

    bool update(double deltaTime, double simulationDeltaTime) override;
    void getSimulationState(SimulationState &state) override;
    bool isConnected() const override;

    void startElement(
        const XMLCh *const uri,
        const XMLCh *const localname,
        const XMLCh *const qname,
        const xercesc::Attributes &attrs) override;
    void fatalError(const xercesc::SAXParseException &) override;

private:
    double m_parseTimestep = -1.0;
    XMLCh *TAG_root;
    XMLCh *TAG_timestep;
    XMLCh *TAG_vehicle;
    XMLCh *ATTR_time;
    XMLCh *ATTR_id;
    XMLCh *ATTR_type;
    XMLCh *ATTR_x;
    XMLCh *ATTR_y;
    XMLCh *ATTR_z;
    XMLCh *ATTR_angle;
    XMLCh *ATTR_speed;

    std::map<double, SimulationState> m_simulationStates;
    std::set<double> m_timesteps;

    double m_simulationTime = 0.0;
    double m_lastSimulationTime = -1.0;
};
#endif
