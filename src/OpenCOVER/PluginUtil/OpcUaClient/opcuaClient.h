#ifndef OPENCOVER_OPCUA_CLIENT_H
#define OPENCOVER_OPCUA_CLIENT_H

#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <open62541/types.h>
#include <string>
#include <map>
#include <util/coExport.h>
class UA_Client;

namespace opencover{
namespace opcua
{

struct Field{
    std::string name;
    UA_UInt16 nameSpace = 0;
    UA_UInt32 nodeId = 0; 
};


class PLUGIN_UTILEXPORT Client
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
    double readValue(const std::string &name);
    bool isConnected() const;
    std::vector<std::string> availableFields() const;
private:
    void listVariables(UA_Client* client);

    bool connectMaster();

    opencover::ui::Menu *m_menu;
    opencover::ui::Button *m_connect;
    std::unique_ptr<opencover::ui::FileBrowserConfigValue> m_certificate, m_key;
    std::unique_ptr<opencover::ui::EditFieldConfigValue> m_password, m_username, m_serverIp;
    std::unique_ptr<opencover::ui::SelectionListConfigValue> m_authentificationMode;
    const std::string m_name;
    UA_Client *client = nullptr;
    std::map<std::string, Field> m_numericalNodes;
    std::function<void(void)> m_onConnect, m_onDisconnect;
};

}
}




#endif // OPENCOVER_OPCUA_CLIENT_H
