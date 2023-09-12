#include "observerHandle.h"
#include "opcuaClient.h"
using namespace opencover::opcua;

ObserverHandle::ObserverHandle(size_t id,  Client *client) : m_deleter(std::make_shared<Deleter>()){
    m_deleter->m_id = id;
    m_deleter->m_client = client;
}

bool ObserverHandle::operator==(size_t id) const
{
    return m_deleter->m_id == id;
}

bool ObserverHandle::operator==(const ObserverHandle &other) const
{
    return m_deleter->m_id == other.m_deleter->m_id;;
}

bool ObserverHandle::operator<(const ObserverHandle &other) const
{
    return m_deleter->m_id < other.m_deleter->m_id;;
}

ObserverHandle::Deleter::~Deleter()
{
    if(m_client)
        m_client->queueUnregisterNode(m_id);
}
