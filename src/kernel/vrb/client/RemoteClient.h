#ifndef VRB_CLIENT_REMOTE_CLIENT_H
#define VRB_CLIENT_REMOTE_CLIENT_H

#include <net/message.h>
#include <net/tokenbuffer.h>
#include <util/coExport.h>
#include <vrb/SessionID.h>
#include <vrb/UserInfo.h>

#include <string>
#include <memory>

namespace vrb
{

class VRBCLIENTEXPORT RemoteClient
{
public:
    RemoteClient(); //constructs a client with local information
    RemoteClient(int id); //constructs real remote client 
    virtual ~RemoteClient() = default;
    int getID() const;
    void setID(int id);
    const vrb::SessionID &getSessionID() const;
    const std::string &getName() const;
    const std::string &getEmail() const; 
    const std::string &getHostname() const; 

    void setSession(const vrb::SessionID &g);
    virtual void setMaster(bool m);
    bool isMaster() const;
    void setInfo(covise::TokenBuffer &tb);
    void print() const;
    std::unique_ptr<covise::Message> createHelloMessage();

protected:
    UserInfo m_userInfo;
    int m_id = -1;
    vrb::SessionID m_sessionID;
    std::string m_name;
    bool m_isMaster = false;
};

} // namespace vrb

#endif
