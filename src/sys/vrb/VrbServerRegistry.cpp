/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "VrbServerRegistry.h"
#include <util/unixcompat.h>
#include <net/tokenbuffer.h>
#include <VRBClientList.h>
#include <iostream>

#include <assert.h>


using namespace covise;

VrbServerRegistry *VrbServerRegistry::instance = NULL;

VrbServerRegistry::VrbServerRegistry()
{
    assert(!instance);
    instance = this;
}

VrbServerRegistry::~VrbServerRegistry()
{
    // delete entry list including all entries!
    assert(instance == this);
    instance = NULL;
}

std::map<int, std::shared_ptr<serverRegClass>> VrbServerRegistry::getClasses(const std::string &name)
{
    std::map<int, std::shared_ptr<serverRegClass>> clientsClass;
    for (size_t i = 0; i < clients.num(); i++)
    {
        int id = clients.item(i)->getID();
        if (clientsClasses[id].find(name) == clientsClasses[id].end())
        {
            continue;
        }
        clientsClass[id] = clientsClasses[id][name];
    }
    return clientsClass;
}

serverRegClass *VrbServerRegistry::getClass(int ID, const std::string & name)
{
    if (clientsClasses.find(ID) == clientsClasses.end())//to check if there is a id 0 class and if so copy their observers
    {
        return nullptr;
    }
    if (clientsClasses[ID].find(name) == clientsClasses[ID].end())
    {
        return nullptr;
    }
    return clientsClasses[ID][name].get();
}
/// set a Value or create new Entry
void VrbServerRegistry::setVar(int ID, const std::string &className, const std::string &name, covise::TokenBuffer &value, int senderID, bool s)
{
    serverRegClass *rc = getClass(ID, className);
    if (!rc)
    {
        serverRegClass *grc = getClass(0, className);
        rc = new serverRegClass(className, ID);
        clientsClasses[ID][className].reset(rc);
        if (grc)
        {
            // we have a generic observer for this class name, copy observers
            for (const auto client : grc->observers)
            {
                rc->observers.insert(client);
            }
        }
    }
    serverRegVar *rv = rc->getVar(name);
    auto cl = currentVariables.find(className);
    if (rv)
    {
        rv->setValue(value);
        rv->setLastEditor(senderID);
    }
    else 
    {
        rv = new serverRegVar(rc, name, value);
        rc->append(rv);
        rv->setLastEditor(senderID);
    }

    //call observers
    std::set<int> collectiveObservers = *rc->getOList();
    auto &client = clientsClasses[0];
    auto clx = client.find(className);
    if (clx != client.end())
    {
        collectiveObservers.insert(clx->second->getOList()->begin(), clx->second->getOList()->end());
    }
    collectiveObservers.insert(rv->getOList()->begin(), rv->getOList()->end());
    sendVariableChange(rv, collectiveObservers);
    updateUI(rv);


}

void VrbServerRegistry::updateUI(serverRegVar *rv)
{

}
/// create new Entry
void VrbServerRegistry::create(int ID, const std::string &className, const std::string &name, covise::TokenBuffer &value, bool s)
{
    serverRegClass *rc = getClass(ID, className);
    if (rc)
    {
        serverRegVar *rv = rc->getVar(name);
        if (rv)
        {
            return;
        }
    }
    setVar(ID, className, name, value, s);
}
/// get a boolean Variable
int VrbServerRegistry::isTrue(int ID, const std::string &className, const std::string &name, int def)
{
    serverRegClass *rc = getClass(ID, className);
    if (rc)
    {
        serverRegVar *rv = rc->getVar(name);
        if (rv)
        {
            bool b;
            rv->getValue() >> b;
            rv->getValue().rewind();
            return b;
        }
        return def;
    }
    return def;
}

void VrbServerRegistry::deleteEntry(int ID, const std::string &className, const std::string &name)
{
    if (currentVariables[className][name] == ID) //dont delete the entry if it is the current one *FIXME: delete it when sb else becomes current
    {
        return;
    }
    clientsClasses[ID][className]->deleteVar(name);

}

void VrbServerRegistry::deleteEntry(int modID)
{
    if (!modID)
    {
        return;
    }
    for (const auto cl : clientsClasses[modID])
    {
        cl.second->deleteAllNonStaticVars();
    }
}

void VrbServerRegistry::unObserve(int recvID)
{
    for (const auto client : clientsClasses)
    {
        for (const auto cl : client.second)
        {
            cl.second->unObserve(recvID);
        }
    }
}

void VrbServerRegistry::sendVariableChange(serverRegVar * rv, std::set<int> observers)
{
    covise::TokenBuffer sb;
    sb << rv->getClass()->getID(); //sender ID
    sb << rv->getClass()->getName();
    sb << rv->getName();
    sb << rv->getValue();

    for (const auto client : observers)
    {
        VRBSClient *cl = clients.get(client);
        if (cl)
        {
            clients.sendMessageToID(sb, client, COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED);
        }
    }
}

void VrbServerRegistry::observeVar(int ID, const std::string &className, int recvID, const std::string &variableName, covise::TokenBuffer &value)
{
    if (ID == 0) //observe all clients class with className
    {
        for (auto it = clientsClasses.begin();
            it != clientsClasses.end(); ++it)
        {
            auto found = it->second.find(className);
            if (found != it->second.end())
            {
                found->second->observeVar(recvID, variableName, value);
            }
        }
        return;
    }

    auto idIt = clientsClasses.find(ID);
    if (idIt == clientsClasses.end())
    {
        idIt = clientsClasses.emplace(ID, std::map<const std::string, std::shared_ptr<serverRegClass>>()).first;
    }
    auto classIt = idIt->second.find(className);
    if(classIt == idIt->second.end()) //if class does not exists create it
    {
        auto rc = std::make_shared<serverRegClass>(className, ID);
        classIt = idIt->second.emplace(className, rc).first;
    }
    classIt->second->observeVar(recvID, variableName, value);
}

void VrbServerRegistry::observeClass(int ID, const std::string &className, int recvID)
{
    if (ID == 0) //observe all clients class with className
    {
        for (auto it = clientsClasses.begin();
            it != clientsClasses.end(); ++it)
        {
            auto found = it->second.find(className);
            if (found != it->second.end())
            {
                found->second->observe(recvID);
            }
        }
        return;
    }

    auto idIt = clientsClasses.find(ID);
    if (idIt == clientsClasses.end())
    {
        idIt = clientsClasses.emplace(ID, std::map<const std::string, std::shared_ptr<serverRegClass>>()).first;
    }
    auto classIt = idIt->second.find(className);
    if (classIt == idIt->second.end()) //if class does not exists create it
    {
        auto rc = std::make_shared<serverRegClass>(className, ID);
        classIt = idIt->second.emplace(className, rc).first;
    }
    classIt->second->observe(recvID);
}

void VrbServerRegistry::unObserve(int ID, const std::string &className, int recvID, const std::string &variableName)
{
    if (className.empty())
    {
        return;
    }
    int foundOne = 0;
    if (ID == 0)
    {
        for (auto it = clientsClasses.begin(); it != clientsClasses.end(); ++it)
        {
            auto cl = it->second.find(className);
            if (cl != it->second.end())
            {
                cl->second->unObserveVar(recvID, variableName);
                foundOne = true;
            }
        }
    }
    clientsClasses[ID][className]->unObserveVar(recvID, variableName);
    if (!foundOne)
    {
        if (!variableName.empty())
            std::cerr << "Variable " << variableName << " not found in class " << className << " ID: " << ID << std::endl;
        else
            std::cerr << "Class " << className << " ID: " << ID << " not found" << std::endl;
    }
}

/////////////SERVERREGVAR/////////////////////////////////////////////////
serverRegVar::~serverRegVar()
{
    informDeleteObservers();
}

void serverRegVar::update(int recvID)
{
    covise::TokenBuffer sb;
    sb << myClass->getID();
    sb << myClass->getName();
    sb << getName();

    sb << getValue();
    clients.sendMessageToID(sb, recvID, COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED);
}

void serverRegVar::informDeleteObservers()
{

    covise::TokenBuffer sb;
    sb << getClass()->getID();
    sb << getClass()->getName();
    sb << getName();
    sb << getValue();
    std::set<int> combinedObservers;
    combinedObservers = observers;
    for (const int obs : getClass()->observers)
    {
        combinedObservers.insert(obs);
    }
    for (const int obs : combinedObservers)
    {
        clients.sendMessageToID(sb, obs, COVISE_MESSAGE_VRB_REGISTRY_ENTRY_DELETED);
    }
}

void serverRegVar::setLastEditor(int id)
{
    lastEditor = id;
}

int serverRegVar::getLastEditor()
{
    return lastEditor;
}

/////////////SERVERREGCLASS/////////////////////////////////////////////////
void serverRegClass::observe(int recvID)
{
    observers.insert(recvID);
    for (const auto var : myVariables)
    {
        var.second->observe(recvID);
    }
}

void serverRegClass::observeVar(int recvID, const std::string &variableName, covise::TokenBuffer &value)
{
    serverRegVar *rv = getVar(variableName);
    if (!rv)
    {
        rv = new serverRegVar(this, variableName, value);
        append(rv);
    }
    rv->observe(recvID);

}

void serverRegClass::unObserveVar(int recvID, const std::string &variableName)
{
    serverRegVar *rv = getVar(variableName);
    if (rv)
    {
        rv->unObserve(recvID);
    }
}
void serverRegClass::unObserve(int recvID)
{
    observers.erase(recvID);
    for (const auto &var : myVariables)
    {
        var.second->unObserve(recvID);
    }
}
   
