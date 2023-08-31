#ifndef OPENCOVER_OPCUA_CLIENT_H
#define OPENCOVER_OPCUA_CLIENT_H

#include "types.h"

#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <open62541/types.h>
#include <string>
#include <map>
#include "export.h"
class UA_Client;

namespace opencover{
namespace opcua
{
extern const char *NoNodeName;

class OPCUACLIENTEXPORT Client
{
public:
    Client(const std::string &name);
    ~Client();
    void onConnect(const std::function<void(void)> &cb);
    void onDisconnect(const std::function<void(void)> &cb);
    bool connect();
    bool disconnect();
    bool registerDouble(const std::string &name, const std::function<void(double)> &cb);
    void update();
    double readNumericValue(const std::string &name);

    template<typename T>
    T readValue(const std::string &name) //no typechecking after findNode
    {
        auto node = findNode(name, detail::getTypeId<T>());
        if(!node)
            return T();
        auto data = readData(*node, sizeof(T));
        return *(T*)data.data();
    }

    bool isConnected() const;
    std::vector<std::string> allAvailableScalars() const;
    std::vector<std::string> availableNumericalScalars() const;
    template<typename T>
    std::vector<std::string> availableScalars() const
    {
        return availableNodes<T>(m_scalarNodes);
    }
    std::vector<std::string> allAvailableArrays() const;
    std::vector<std::string> availableNumericalArrays() const;
    template<typename T>
    std::vector<std::string> availableArrays() const
    {
        return availableNodes<T>(m_arrayNodes);
    }
    typedef  std::map<int, std::map<std::string, UA_NodeId>> NodeMap;
private:
    
    void listVariables(UA_Client* client);
    void listVariablesInResponse(UA_Client* client, UA_BrowseResponse &bResp);
    bool connectMaster();
    const UA_NodeId *findNode(const std::string &name, int type, bool silent = false) const;
    std::vector<char> readData(const UA_NodeId &id, size_t size);

    std::vector<std::string> allAvailableNodes(const NodeMap &map) const;
    std::vector<std::string> availableNumericalNodes(const NodeMap &map) const;
    template<typename T>
    std::vector<std::string> availableNodes(const NodeMap &map) const
    {
        std::vector<std::string> v;
        v.push_back(NoNodeName);
        auto typeIt = map.find(detail::getTypeId<T>());
        if(typeIt == map.end())
            return v;
        for(const auto &node : typeIt->second)
            v.push_back(node.first);
        return v;
    }

    opencover::ui::Menu *m_menu;
    opencover::ui::Button *m_connect;
    std::unique_ptr<opencover::ui::FileBrowserConfigValue> m_certificate, m_key;
    std::unique_ptr<opencover::ui::EditFieldConfigValue> m_password, m_username, m_serverIp;
    std::unique_ptr<opencover::ui::SelectionListConfigValue> m_authentificationMode;
    const std::string m_name;
    UA_Client *client = nullptr;
    NodeMap m_scalarNodes, m_arrayNodes;
    std::function<void(void)> m_onConnect, m_onDisconnect;
};

}
}




#endif // OPENCOVER_OPCUA_CLIENT_H
