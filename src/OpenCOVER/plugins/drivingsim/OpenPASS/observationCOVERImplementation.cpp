/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-----------------------------------------------------------------------------
/** \file  ObservationCOVERImplementation */
//-----------------------------------------------------------------------------

#include <cassert>
#include <sstream>
#include <QDir>

#include "Interfaces/stochasticsInterface.h"
#include "Interfaces/worldInterface.h"

#include "observationCOVERImplementation.h"
#include "OpenPASS.h"

ObservationCOVERImplementation::ObservationCOVERImplementation(SimulationSlave::EventNetworkInterface* eventNetwork,
        StochasticsInterface* stochastics,
        WorldInterface* world,
        const ParameterInterface* parameters,
        const CallbackInterface* callbacks) :
    ObservationInterface(stochastics,
                         world,
                         parameters,
                         callbacks),
    eventNetwork(eventNetwork)
{
    
}

//-----------------------------------------------------------------------------
//! \brief Logs a key/value pair
//!
//! @param[in]     time      current time
//! @param[in]     agentId   agent identifier
//! @param[in]     group     LoggingGroup the key/value pair should be assigned to
//! @param[in]     key       Key of the value to log
//! @param[in]     value     Value to log
//-----------------------------------------------------------------------------
void ObservationCOVERImplementation::Insert(int time,
        int agentId,
        LoggingGroup group,
        const std::string& key,
        const std::string& value)
{
    
}

//-----------------------------------------------------------------------------
//! \brief Logs an event
//!
//! @param[in]     event     Shared pointer to the event to log
//-----------------------------------------------------------------------------
void ObservationCOVERImplementation::InsertEvent(std::shared_ptr<EventInterface> event)
{
    eventNetwork->InsertEvent(event);
}

void ObservationCOVERImplementation::SlavePreHook(const std::string& path)
{
  
}

void ObservationCOVERImplementation::SlavePreRunHook()
{
    
}

void ObservationCOVERImplementation::SlaveUpdateHook(int time, RunResultInterface& runResult)
{
    OpenPASS::instance()->updateTime(time, GetWorld());
}

void ObservationCOVERImplementation::SlavePostRunHook(const RunResultInterface& runResult)
{
    OpenPASS::instance()->postRun(runResult);
}

void ObservationCOVERImplementation::SlavePostHook()
{
}

void ObservationCOVERImplementation::GatherFollowers()
{
    
}

void ObservationCOVERImplementation::InformObserverOnSpawn(AgentInterface* agent)
{
    OpenPASS::instance()->addAgent(agent);
}
