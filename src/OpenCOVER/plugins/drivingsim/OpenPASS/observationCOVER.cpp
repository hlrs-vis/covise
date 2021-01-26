/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-----------------------------------------------------------------------------
/** \file  ObservationLog.cpp */
//-----------------------------------------------------------------------------

#include "observationCOVER.h"
#include "observationCOVERImplementation.h"

const std::string Version = "1.0.0";    //!< The version of the current module - has to be incremented manually
static const CallbackInterface* Callbacks = nullptr;

//-----------------------------------------------------------------------------
//! dll-function to obtain the version of the current module
//!
//! @return                       Version of the current module
//-----------------------------------------------------------------------------
extern "C" OBSERVATION_COVER_SHARED_EXPORT const std::string& OpenPASS_GetVersion()
{
    return Version;
}

//-----------------------------------------------------------------------------
//! dll-function to create an instance of the module
//!
//! @param[in]     stochastics    Pointer to the stochastics class loaded by the framework
//! @param[in]     world          Pointer to the world
//! @param[in]     parameters     Pointer to the parameters of the module
//! @param[in]     callbacks      Pointer to the callbacks
//! @return                       A pointer to the created module instance
//-----------------------------------------------------------------------------
extern "C" OBSERVATION_COVER_SHARED_EXPORT ObservationInterface* OpenPASS_CreateInstance(
    StochasticsInterface* stochastics,
    WorldInterface* world,
    SimulationSlave::EventNetworkInterface* eventNetwork,
    const ParameterInterface* parameters,
    const CallbackInterface* callbacks,
    DataStoreReadInterface * dataStore)
{
    Callbacks = callbacks;

    try
    {
        return (ObservationInterface*)(new (std::nothrow) ObservationCOVERImplementation(eventNetwork,
                                       stochastics,
                                       world,
                                       parameters,
                                       callbacks,
                                        dataStore));
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return nullptr;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return nullptr;
    }
}

//-----------------------------------------------------------------------------
//! dll-function to destroy/delete an instance of the module
//!
//! @param[in]     implementation    The instance that should be freed
//-----------------------------------------------------------------------------
extern "C" OBSERVATION_COVER_SHARED_EXPORT void OpenPASS_DestroyInstance(ObservationInterface* implementation)
{
    delete implementation;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT bool OpenPASS_MasterPreHook(ObservationInterface* implementation)
{
    try
    {
        implementation->MasterPreHook();
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return false;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return false;
    }

    return true;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT bool OpenPASS_MasterPostHook(ObservationInterface* implementation,
        const std::string& filename)
{
    try
    {
        implementation->MasterPostHook(filename);
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return false;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return false;
    }

    return true;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT bool OpenPASS_SlavePreHook(ObservationInterface* implementation)
{
    try
    {
        implementation->SlavePreHook();
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return false;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return false;
    }

    return true;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT bool OpenPASS_SlavePreRunHook(ObservationInterface* implementation)
{
    try
    {
        implementation->SlavePreRunHook();
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return false;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return false;
    }

    return true;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT bool OpenPASS_SlaveUpdateHook(ObservationInterface* implementation,
        int time,
        RunResultInterface& runResult)
{
    try
    {
        implementation->SlaveUpdateHook(time, runResult);
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return false;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return false;
    }

    return true;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT bool OpenPASS_SlavePostRunHook(ObservationInterface* implementation,
        const RunResultInterface& runResult)
{
    try
    {
        implementation->SlavePostRunHook(runResult);
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return false;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return false;
    }

    return true;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT bool OpenPASS_SlavePostHook(ObservationInterface* implementation)
{
    try
    {
        implementation->SlavePostHook();
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return false;
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return false;
    }

    return true;
}

extern "C" OBSERVATION_COVER_SHARED_EXPORT const std::string OpenPASS_SlaveResultFile(
    ObservationInterface* implementation)
{
    try
    {
        return implementation->SlaveResultFile();
    }
    catch (const std::runtime_error& ex)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, ex.what());
        }

        return "";
    }
    catch (...)
    {
        if (Callbacks != nullptr)
        {
            Callbacks->Log(CbkLogLevel::Error, __FILE__, __LINE__, "unexpected exception");
        }

        return "";
    }
}
