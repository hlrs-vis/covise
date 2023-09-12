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
    opencover::opcua::Client *getClient(const std::string& name);
    Client *connect(const std::string &name);
private:
    Manager();
    void createClient(const std::string &name);
    static Manager* m_instance;
    std::map<std::string,std::unique_ptr<opencover::opcua::Client>> m_clients;
    std::unique_ptr<config::File> m_config;
    ui::Menu *m_menu;
    ui::SelectionList *m_configuredServersList;
    ui::Action *m_createBtn;
    ui::EditField *m_newClientName;
};

}

OPCUACLIENTEXPORT Client *getClient(const std::string &name);
OPCUACLIENTEXPORT Client *connect(const std::string &name);

}
}



#endif // OPENCOVER_OPCUA_H 
