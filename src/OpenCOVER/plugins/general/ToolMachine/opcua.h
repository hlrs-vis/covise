#ifndef COVER_TOOL_MACHINE_OPCUA_H
#define COVER_TOOL_MACHINE_OPCUA_H

#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <open62541/types.h>
#include <string>
#include <map>
class UA_Client;

struct OpcUaField{
    std::string name;
    UA_UInt16 nameSpace = 0;
    UA_UInt32 nodeId = 0; 
};

class OpcUaClient
{
public:
    OpcUaClient(const std::string &name, opencover::ui::Menu *menu, opencover::config::File &config);
    ~OpcUaClient();
    void onConnect(const std::function<void(void)> &cb);
    void onDisconnect(const std::function<void(void)> &cb);
    bool connect();
    bool disconnect();
    bool registerDouble(const std::string &name, const std::function<void(double)> &cb);
    void update();
    double readValue(const std::string &name);
    bool isConnected() const;
private:
    void listVariables(UA_Client* client);


    opencover::ui::Menu *m_menu;
    opencover::ui::Button *m_connect;
    std::unique_ptr<opencover::ui::FileBrowserConfigValue> m_certificate, m_key;
    std::unique_ptr<opencover::ui::EditFieldConfigValue> m_password, m_username, m_serverIp;
    const std::string m_name;
    UA_Client *client = nullptr;
    std::map<std::string, OpcUaField> m_numericalNodes;
    std::function<void(void)> m_onConnect, m_onDisconnect;
};


#endif // COVER_TOOL_MACHINE_OPCUA_H
