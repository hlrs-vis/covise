#include "RemoteClient.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <sstream>
#include <net/message_types.h>
#include <algorithm>
using namespace vrb;

RemoteClient::RemoteClient(Program type, const std::string& sessionName)
    : m_userInfo(type)
    , m_session(0, sessionName)
{
}
namespace vrb{
namespace detail{
int getID(covise::TokenBuffer &tb){
    int i;
    tb >> i;
    return i;
}

SessionID getSid(covise::TokenBuffer &tb){
    SessionID sid;
    tb >> sid;
    return sid;
}
}//detail
}//vrb

RemoteClient::RemoteClient(covise::TokenBuffer &tb)
    : m_id(detail::getID(tb)) 
    , m_session(detail::getSid(tb))
    , m_userInfo(tb)
{
}

int RemoteClient::ID() const
{
    return m_id;
}

void RemoteClient::setID(int id)
{
    m_id = id;
}

const UserInfo &RemoteClient::userInfo() const{
    return m_userInfo;
}


const vrb::SessionID &RemoteClient::sessionID() const
{
    return m_session;
}

void RemoteClient::setSession(const vrb::SessionID &g)
{
    m_session = g;
}


void RemoteClient::setMaster(int clientID){
    m_session.setMaster(clientID);
}

bool RemoteClient::isMaster() const
{
    return m_session.master() == m_id;
}

void RemoteClient::print() const
{
    std::cerr << "ID: " << m_id << std::endl;
    std::cerr <<  m_userInfo << std::endl;
    std::cerr << m_session << std::endl;
}


covise::TokenBuffer &vrb::operator<<(covise::TokenBuffer &tb, const RemoteClient &rc){
    tb << rc.ID() << rc.sessionID() << rc.userInfo();
    return tb;
}

bool vrb::operator<(const RemoteClient &r1, const RemoteClient &r2){
    return r1.ID() < r2.ID();
}

