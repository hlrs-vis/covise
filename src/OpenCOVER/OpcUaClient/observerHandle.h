#ifndef COVER_OPCUA_REGISTER_ID_H
#define COVER_OPCUA_REGISTER_ID_H

#include <memory>

namespace opencover{namespace opcua{

class Client;
class ClientNode;

//automativally unobserves the node on deletion
class ObserverHandle{
    public:
        friend class Client;
        ObserverHandle() = default;
        ObserverHandle(size_t id,  Client *client);
        bool operator==(size_t id) const;
        bool operator==(const ObserverHandle &other) const;
        bool operator<(const ObserverHandle &other) const;
    private:
        struct Deleter{
            ~Deleter();
            size_t m_id = 0;
            Client *m_client = nullptr;
        };
        std::shared_ptr<Deleter> m_deleter;
        opencover::opcua::ClientNode *m_node = nullptr; 
    };

}}

#endif // COVER_OPCUA_REGISTER_ID_H