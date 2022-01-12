#include "vrbClientInterface.h"
#include <vrb/client/VRBClient.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <vrb/VrbSetUserInfoMessage.h>
#include <vrb/client/VrbCredentials.h>
#include <set>
namespace vrbclient
{
namespace detail
{

class VrbClientInternals
{
public:
    VrbClientInternals(const std::string &ip, unsigned int port);
    VrbClientInternals(const VrbClientInternals &) = delete;
    VrbClientInternals(VrbClientInternals &&) = delete;
    VrbClientInternals &operator=(const VrbClientInternals &) = delete;
    VrbClientInternals &operator=(VrbClientInternals &&) = delete;
    ~VrbClientInternals() = default;

    bool isConnected();
    void update();
    std::vector<std::string> sessions() const;
    void joinSession(const std::string &name);
    std::vector<std::string> partner() const;

    std::vector<std::string> daemons() const;
    void addPartner(const std::string &name);

private:
    vrb::VRBClient m_client;
    std::set<vrb::RemoteClient> m_clientList;
};

VrbClientInternals::VrbClientInternals(const std::string &ip, unsigned int port)
    : m_client(covise::Program::external, vrb::VrbCredentials{ip, port, 50000})
{
    m_client.connectToServer();
}

bool VrbClientInternals::isConnected()
{
    return m_client.isConnected();
}

void VrbClientInternals::update()
{
    using namespace covise;
    covise::Message msg;
    while (m_client.poll(&msg))
    {
        switch (msg.type)
        {
        case COVISE_MESSAGE_VRB_SET_USERINFO:
        {

            vrb::UserInfoMessage uim(&msg);
            if (uim.hasMyInfo)
            {
                m_client.setID(uim.myClientID);
                m_client.setSession(uim.mySession);
            }
            for (auto &cl : uim.otherClients)
            {
                if (cl.userInfo().userType == covise::Program::external && cl.sessionID() == m_client.sessionID())
                {
                    //onPartnerJoinedCb
                }
                m_clientList.insert(std::move(cl));
            }
        }
        default:
            break;
        }
    }
}

std::vector<std::string> VrbClientInternals::sessions() const
{
    return std::vector<std::string>();
}

void VrbClientInternals::joinSession(const std::string &name)
{
    
}

std::vector<std::string> VrbClientInternals::partner() const
{
    return std::vector<std::string>();
    
}

std::vector<std::string> VrbClientInternals::daemons() const
{
    return std::vector<std::string>();
}

void VrbClientInternals::addPartner(const std::string &name)
{
    
}
} //detail
VrbClient::VrbClient(const std::string &ip, unsigned int port)
{
    internals = new detail::VrbClientInternals{ip, port};
}
VrbClient::~VrbClient()
{
    delete internals;
}

bool VrbClient::isConnected() const
{
    return internals->isConnected();
}

std::vector<std::string> VrbClient::sessions() const
{
    return internals->sessions();
}

void VrbClient::joinSession(const std::string &name)
{
    internals->joinSession(name);
}

std::vector<std::string> VrbClient::partner() const
{
    return internals->partner();
}

std::vector<std::string> VrbClient::daemons() const
{
    return internals->daemons();
}

void VrbClient::addPartner(const std::string &name)
{
    internals->addPartner(name);
}

} //vrbclient
