#include "RemoteClient.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <sstream>
#include <net/message_types.h>
#include <algorithm>
using namespace vrb;

RemoteClient::RemoteClient(UserType type)
    : m_id(-1)
    , m_isMaster(false)
    , m_sessionID()
    , m_name(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.UserName", covise::Host::getHostname()))
    , m_userInfo(type)
{
}

RemoteClient::RemoteClient(int id)
    : m_id(id), m_isMaster(false)
{
}

int RemoteClient::getID() const
{
    return m_id;
}

void RemoteClient::setID(int id)
{
    m_id = id;
}

const vrb::SessionID &RemoteClient::getSessionID() const
{
    return m_sessionID;
}

const std::string &RemoteClient::getName() const
{
    return m_name;
}

const std::string &RemoteClient::getEmail() const
{
    return m_userInfo.email;
}

const std::string &RemoteClient::getHostname() const
{
    return m_userInfo.hostName;
}

UserType RemoteClient::getUserType() const{
    return m_userInfo.userType;
}

void RemoteClient::setSession(const vrb::SessionID &g)
{
    m_sessionID = g;
}

void RemoteClient::setMaster(bool isMaster)
{
    m_isMaster = isMaster;
}

bool RemoteClient::isMaster() const
{
    return m_isMaster;
}

void RemoteClient::setInfo(covise::TokenBuffer &tb)
{
    tb >> m_name; 
    tb >> m_userInfo;
    tb >> m_sessionID;
    tb >> m_isMaster;
}

void RemoteClient::print() const
{
    std::cerr << "ID:       " << m_id << std::endl;
    std::cerr << "Name:     " << m_name << std::endl;
    std::cerr << m_userInfo << std::endl;
    std::cerr << "Group:    " << m_sessionID.toText() << std::endl;
    std::cerr << "Master:   " << m_isMaster << std::endl;
}

std::unique_ptr<covise::Message> RemoteClient::createHelloMessage()
{
    covise::TokenBuffer tb;
    tb << m_userInfo;

    auto msg = std::unique_ptr<covise::Message>(new covise::Message(tb));
    msg->type = covise::COVISE_MESSAGE_VRB_SET_USERINFO;
    return msg;
}
