/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-----------------------------------------------------------------------------
/*!
* \file  ObservationCOVERImplementation.h
* \brief Adds the RunStatistic information to the simulation output.
* \details  Writes the RunStatistic information into the simulation output.
*           Also manages the stop reasons of the simulation.
*/
//-----------------------------------------------------------------------------

#pragma once

#include <string>
#include <tuple>
#include <QFile>
#include <QTextStream>
#include "Interfaces/observationInterface.h"
#include "Interfaces/eventNetworkInterface.h"

//-----------------------------------------------------------------------------
/** \brief This class adds the RunStatistic information to the simulation output.
*   \details This class inherits the ObservationCOVERGeneric which creates the basic simulation output
*            and adds the RunStatistic information to the output.
*            This class also manages the stop reasons of the simulation.
*
*   \ingroup ObservationCOVER
*/
//-----------------------------------------------------------------------------
class ObservationCOVERImplementation : ObservationInterface
{
public:
    const std::string COMPONENTNAME = "ObservationCOVER";

    ObservationCOVERImplementation(SimulationSlave::EventNetworkInterface* eventNetwork,
                                 StochasticsInterface* stochastics,
                                 WorldInterface* world,
                                 const ParameterInterface* parameters,
                                 const CallbackInterface* callbacks);
    ObservationCOVERImplementation(const ObservationCOVERImplementation&) = delete;
    ObservationCOVERImplementation(ObservationCOVERImplementation&&) = delete;
    ObservationCOVERImplementation& operator=(const ObservationCOVERImplementation&) = delete;
    ObservationCOVERImplementation& operator=(ObservationCOVERImplementation&&) = delete;
    virtual ~ObservationCOVERImplementation() override = default;

    virtual void Insert(int time, int agentId, LoggingGroup group, const std::string& key, const std::string& value) override;
    virtual void InsertEvent(std::shared_ptr<EventInterface> event) override;
    virtual void SlavePreHook(const std::string& path) override;
    virtual void SlavePreRunHook() override;
    virtual void SlavePostRunHook(const RunResultInterface& runResult) override;
    virtual void SlaveUpdateHook(int, RunResultInterface&) override;
    virtual void MasterPreHook() override {}
    virtual void MasterPostHook(const std::string&) override {}
    virtual void SlavePostHook() override;

    //-----------------------------------------------------------------------------
    /*!
    * \brief Calculates the ego followers at simulation start
    */
    //-----------------------------------------------------------------------------
    virtual void GatherFollowers() override;

    //-----------------------------------------------------------------------------
    /*!
    * \brief Insert the id of the agent into the list of followers of it is behind the ego
    */
    //-----------------------------------------------------------------------------
    virtual void InformObserverOnSpawn(AgentInterface* agent) override;

    virtual const std::string SlaveResultFile() override
    {
        return "";
    }

private:
    SimulationSlave::EventNetworkInterface* eventNetwork;
};


