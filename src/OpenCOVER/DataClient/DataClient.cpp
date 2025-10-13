#include "DataClient.h"
#include <cover/coVRMSController.h>

using namespace opencover::dataclient;

const char *opencover::dataclient::NoNodeName = "None";

void Client::queueUnregisterNode(size_t id)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_nodesToUnregister.push_back(id);
}

bool isNumerical(std::type_index t)
{
    return t == typeid(int8_t) || t == typeid(uint8_t) || t == typeid(int16_t) || t == typeid(uint16_t) || t == typeid(int32_t) || t == typeid(uint32_t) || t == typeid(double);
}

std::vector<std::string> Client::allAvailableScalars() const
{
    return getNodesWith(false, true);
}

std::vector<std::string> Client::availableNumericalScalars() const
{
    return getNodesWith(true, true);
}

std::vector<std::string> Client::allAvailableArrays() const
{
    return getNodesWith(false, false);
}

std::vector<std::string> Client::availableNumericalArrays() const
{
    return getNodesWith(true, false);
}

auto msController = opencover::coVRMSController::instance();

Client::StatusChange Client::statusChanged(void* caller)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    StatusChange retval = Unchanged;
    if(msController->isMaster())
    {
        auto obs = m_statusObservers.find(caller);
        if(obs == m_statusObservers.end())
        {
            obs = m_statusObservers.insert(obs, std::make_pair(caller, false));
        }
        if(!obs->second)
        {
            obs->second = true;
            retval = isConnected() ? Connected : Disconnected;
        }
    } 
    msController->syncData(&retval, sizeof(StatusChange));
    return retval;
}

void Client::statusChanged()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for(auto &obs : m_statusObservers)
        obs.second = false;
}

Client** Client::getClientReference(const ObserverHandle &handle)
{
    return &handle.m_deleter->m_client;
}