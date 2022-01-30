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

#include "include/stochasticsInterface.h"
#include "include/worldInterface.h"

#include "observationCOVERImplementation.h"
#include "OpenPASS.h"

ObservationCOVERImplementation::ObservationCOVERImplementation(core::EventNetworkInterface* eventNetwork,
        StochasticsInterface* stochastics,
        WorldInterface* world,
        const ParameterInterface* parameters,
        const CallbackInterface* callbacks,
         DataBufferReadInterface* dataBuffer) :
    ObservationInterface(stochastics,
                         world,
                         parameters,
                         callbacks,
                         dataBuffer),
    eventNetwork(eventNetwork)
{
    
}


//-----------------------------------------------------------------------------
//! \brief Logs an event
//!
//! @param[in]     event     Shared pointer to the event to log
//-----------------------------------------------------------------------------
//void ObservationCOVERImplementation::InsertEvent(std::shared_ptr<EventInterface> event)
//{
//    eventNetwork->InsertEvent(event);
//}

void ObservationCOVERImplementation::OpSimulationPreHook()
{
  
}

void ObservationCOVERImplementation::OpSimulationPreRunHook()
{
    
}

void ObservationCOVERImplementation::OpSimulationUpdateHook(int time, RunResultInterface& runResult)
{
    OpenPASS::instance()->updateTime(time, GetWorld());
}

void ObservationCOVERImplementation::OpSimulationPostRunHook(const RunResultInterface& runResult)
{
    OpenPASS::instance()->postRun(runResult);
}

void ObservationCOVERImplementation::OpSimulationPostHook()
{
}

/*
void ObservationCOVERImplementation::InformObserverOnSpawn(AgentInterface* agent)
{
    OpenPASS::instance()->addAgent(agent);
}*/
