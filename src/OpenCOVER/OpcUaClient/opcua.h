#ifndef OPENCOVER_OPCUA_H
#define OPENCOVER_OPCUA_H

#include "opcuaClient.h"
#include <OpenConfig/file.h>
#include <cover/ui/Owner.h>
#include <util/coExport.h>

namespace opencover{
namespace ui{
class Action;
class Menu;
class SelectionList;
class EditField;
}

namespace opcua
{

namespace detail{

class Manager : public ui::Owner
{
    friend opencover::opcua::Client;

public:
    static Manager *instance();
    opencover::opcua::Client *getClient();
    void addOnClientConnectedCallback(const std::function<void(void)> &cb);
    void addOnClientDisconnectedCallback(const std::function<void(void)> &cb);
    void connect(const std::string &name);
private:
    Manager();
    void createClient(const std::string &name);
    static Manager* m_instance;
    std::unique_ptr<opencover::opcua::Client> m_client;
    std::unique_ptr<config::File> m_config;
    ui::Menu *m_menu;
    ui::SelectionList *m_configuredServersList;
    ui::Action *m_createBtn;
    ui::EditField *m_newClientName;
    std::vector<std::function<void(void)>> m_onConnectCbs, m_onDisconnectCbs;
};

}

OPCUACLIENTEXPORT Client * getClient();
OPCUACLIENTEXPORT void addOnClientConnectedCallback(const std::function<void(void)> &cb);
OPCUACLIENTEXPORT void addOnClientDisconnectedCallback(const std::function<void(void)> &cb);
OPCUACLIENTEXPORT void connect(const std::string &name);

}
}



#endif // OPENCOVER_OPCUA_H 
