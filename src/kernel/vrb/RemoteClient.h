#ifndef VRB_CLIENT_REMOTE_CLIENT_H
#define VRB_CLIENT_REMOTE_CLIENT_H

#include <net/message.h>
#include <net/tokenbuffer.h>
#include <util/coExport.h>
#include <vrb/SessionID.h>
#include <net/userinfo.h>

#include <string>
#include <memory>

namespace vrb
{

class VRBEXPORT RemoteClient
{
public:
    RemoteClient(covise::Program type, const std::string& sessionName = ""); //constructs a client with local information
    RemoteClient(covise::TokenBuffer &tb); //constructs real remote client
    RemoteClient(RemoteClient &&other) = default;
    RemoteClient(const RemoteClient &other) = delete;
    virtual RemoteClient &operator=(RemoteClient &&other) = delete;
    virtual RemoteClient &operator=(const RemoteClient &other) = delete;

    virtual ~RemoteClient() = default;
    int ID() const;
    void setID(int id);
    virtual const vrb::SessionID &sessionID() const;
    virtual void setSession(const vrb::SessionID &g);
    const covise::UserInfo &userInfo() const;
    virtual void setMaster(int clientID);
    bool isMaster() const;

    void print() const;
protected:
    int m_id = 0;
    SessionID m_session;
    covise::UserInfo m_userInfo;

};

VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const RemoteClient &rc);
VRBEXPORT bool operator<(const RemoteClient &r1, const RemoteClient &r2);

} // namespace vrb

#endif
