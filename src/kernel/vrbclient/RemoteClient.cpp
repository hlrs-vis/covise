#include "RemoteClient.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <sstream>
#include <net/message_types.h>
using namespace vrb;

RemoteClient::RemoteClient()
    : m_id(-1), m_isMaster(false), m_sessionID(), m_hostname(covise::Host::getHostname()), m_address(covise::Host::getHostaddress()), m_name(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.UserName", covise::Host::getHostname())), m_email(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.Email", "covise-users@listserv.uni-stuttgart.de")), m_url(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.URL", "www.hlrs.de/covise"))
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
    return m_email;
}

const std::string &RemoteClient::getHostname() const
{
    return m_hostname;
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
    char *tmp, *tmp2;
    tb >> m_address;
    tb >> m_name; // name
    tb >> tmp;    // userInfo
    tb >> m_sessionID;
    tb >> m_isMaster;

    char *c = tmp;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    m_hostname = tmp2;
    c++;

    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    m_name = tmp2;
    c++;

    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    m_email = tmp2;
    c++;

    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    m_url = tmp2;
    c++;
}

void RemoteClient::print() const
{
    std::cerr << "ID:       " << m_id << std::endl;
    std::cerr << "HostName: " << m_hostname << std::endl;
    std::cerr << "Address:  " << m_address << std::endl;
    std::cerr << "Name:     " << m_name << std::endl;
    std::cerr << "Email:    " << m_email << std::endl;
    std::cerr << "URL:      " << m_url << std::endl;
    std::cerr << "Group:    " << m_sessionID.toText() << std::endl;
    std::cerr << "Master:   " << m_isMaster << std::endl;
}

std::unique_ptr<covise::Message> RemoteClient::createHelloMessage()
{
    std::stringstream str;
    str << "\"" << m_hostname << "\",\"" << m_name << "\",\"" << m_email << "\",\"" << m_url << "\"";

    covise::TokenBuffer tb;
    tb << str.str();

    auto msg = std::unique_ptr<covise::Message>(new covise::Message(tb));
    msg->type = covise::COVISE_MESSAGE_VRB_SET_USERINFO;
    return msg;
}
