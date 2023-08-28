#include "opcua.h"
#include <OpenConfig/access.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Action.h>
#include <cover/ui/EditField.h>
#include <cover/coVRPluginList.h>
#include <iostream>

using namespace opencover::opcua;
using namespace opencover::opcua::detail;

opencover::opcua::detail::Manager *opencover::opcua::detail::Manager::m_instance = nullptr;


Client *Manager::getClient()
{
    return m_client.get();
}

void Manager::addOnClientConnectedCallback(const std::function<void(void)> &cb)
{
    m_onConnectCbs.push_back(cb);
}

void Manager::addOnClientDisconnectedCallback(const std::function<void(void)> &cb)
{
    m_onDisconnectCbs.push_back(cb);
}

void Manager::connect(const std::string &name)
{
    if(name.empty())
        return;
    if(m_configuredServersList->selectedItem() == name && m_client)
    {
        if(m_client->isConnected())
            return;
        m_client->connect();
        return;
    }

    const auto &servers = m_configuredServersList->items();
    for (size_t i = 0; i < servers.size(); i++)
    {
        if(servers[i] == name)
        {
            m_configuredServersList->select(i);
            createClient(m_configuredServersList->selectedItem());
            m_client->connect();
            return;
        }
    }
    std::cerr << "can not find server " << name << " to connect to!" << std::endl;
}

Manager *Manager::instance()
{
    if(!m_instance)
        m_instance = new Manager();
    return m_instance;
}

Manager::Manager()
: ui::Owner("opcuaOwner", cover->ui)
, m_menu(new ui::Menu("opcua", this))
, m_configuredServersList(new ui::SelectionList(m_menu, "availableClients"))
, m_createBtn(new ui::Action(m_menu, "addClient"))
, m_config(cover->configFile("opcua"))
, m_newClientName(new ui::EditField(m_menu, "clientName"))
{
    std::cerr << "opcua menu has elemnt id " << m_menu->elementId() << std::endl;
    m_config->setSaveOnExit(true);
    m_configuredServersList->setList(m_config->sections());
    m_configuredServersList->setCallback([this](int index){
        createClient(m_configuredServersList->selectedItem());
    });
    m_createBtn->setCallback([this](){
        if(!m_newClientName->value().empty())
           createClient(m_newClientName->value());
    });

}

void Manager::createClient(const std::string &name)
{
    m_client = std::make_unique<Client>(name);
    m_client->onConnect([this](){
        for(const auto &cb : m_onConnectCbs)
        {
            if(cb)
                cb();
        }
    });
    m_client->onDisconnect([this](){
        for(const auto &cb : m_onDisconnectCbs)
        {
            if(cb)
                cb();
        }
    });
    
}

Client *opencover::opcua::getClient()
{
    return detail::Manager::instance()->getClient();
}

void opencover::opcua::addOnClientConnectedCallback(const std::function<void(void)> &cb)
{
    detail::Manager::instance()->addOnClientConnectedCallback(cb);
}

void opencover::opcua::addOnClientDisconnectedCallback(const std::function<void(void)> &cb)
{
    detail::Manager::instance()->addOnClientDisconnectedCallback(cb);
}

void opencover::opcua::connect(const std::string &name)
{
    detail::Manager::instance()->connect(name);
}
