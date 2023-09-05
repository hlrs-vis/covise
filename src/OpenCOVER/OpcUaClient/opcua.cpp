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


Client *Manager::getClient(const std::string &name)
{
    return m_clients[name].get();
}

void Manager::addOnClientConnectedCallback(const std::function<void(void)> &cb)
{
    m_onConnectCbs.push_back(cb);
}

void Manager::addOnClientDisconnectedCallback(const std::function<void(void)> &cb)
{
    m_onDisconnectCbs.push_back(cb);
}

Client *Manager::connect(const std::string &name)
{
    if(name.empty())
        return nullptr;
    if(m_clients.find(name)==m_clients.end())
    {
        createClient(name);
    }
    auto& m_client = m_clients[name];
    if (m_client->isConnected())
        return m_client.get();
    m_client->connect();

    const auto &servers = m_configuredServersList->items();
    for (size_t i = 0; i < servers.size(); i++)
    {
        if(servers[i] == name)
        {
            m_configuredServersList->select(i);
        }
    }
    if (m_client->isConnected())
    {
        return m_client.get();
    }
    else
    {
        return nullptr;
    }
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
    m_clients[name] = std::make_unique<Client>(name);
    m_clients[name]->onConnect([this](){
        for(const auto &cb : m_onConnectCbs)
        {
            if(cb)
                cb();
        }
    });
    m_clients[name]->onDisconnect([this](){
        for(const auto &cb : m_onDisconnectCbs)
        {
            if(cb)
                cb();
        }
    });
    
}

Client *opencover::opcua::getClient(const std::string &name)
{
    return detail::Manager::instance()->getClient(name);
}

void opencover::opcua::addOnClientConnectedCallback(const std::function<void(void)> &cb)
{
    detail::Manager::instance()->addOnClientConnectedCallback(cb);
}

void opencover::opcua::addOnClientDisconnectedCallback(const std::function<void(void)> &cb)
{
    detail::Manager::instance()->addOnClientDisconnectedCallback(cb);
}

Client* opencover::opcua::connect(const std::string &name)
{
    return detail::Manager::instance()->connect(name);
}
