/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "regClass.h"
#include <net/message.h>
#include <net/message_types.h>
#include <vrbclient/VrbClientRegistry.h>

using namespace covise;
/////////////CLIENTREGVAR//////////////////////////////////////////////////
void clientRegVar::notifyLocalObserver()
{
    if (_observer)
    {
        _observer->update(this);
    }
}

void clientRegVar::subscribe(regVarObserver * ob, int id)
{
    myClass->setLastEditor(myClass->getID());
    lastEditor = myClass->getID();
    _observer = ob;
    TokenBuffer tb;
    // compose message
    tb << myClass->getID();
    tb << myClass->getName();
    tb << name;
    tb << id;
    tb << value;
    // inform controller about creation
    if (myClass->getID() >= 0 && myClass->getRegistryClient())
        myClass->getRegistryClient()->sendMessage(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE);
}

/////////////CLIENTREGCLASS/////////////////////////////////////////////////

clientRegClass::clientRegClass(const std::string & n, int ID, VrbClientRegistry * reg)
    : regClass(n, ID)
    , registry(reg)
{
}

void clientRegClass::setLastEditor(int lastEditor)
{
    this->lastEditor = lastEditor;
}

void clientRegClass::notifyLocalObserver()
{
    if (_observer)
    {
        _observer->update(this);
    }
}

void clientRegClass::resubscribe(int oldID)
{
    if (classID == oldID)//update ID to new clientID
    {
        classID = registry->getID();
    }
    if (getRegistryClient() && _observer)
    {
        if (myVariables.size() == 0)
        {
            subscribe(_observer, registry->getID());
        }
        else
        {
            for (const auto var : myVariables)
            {
                var.second->subscribe(var.second->getLocalObserver(), registry->getID());
            }
        }
    }

}

void clientRegClass::subscribe(regClassObserver *obs, int id)
{
    lastEditor = classID;
    _observer = obs; //maybe inform old observer
    TokenBuffer tb;
    // compose message
    tb << classID;
    tb << name;
    tb << id; 
    // inform controller about creation
    if (classID >= 0 && getRegistryClient())
        getRegistryClient()->sendMessage(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS);
}
clientRegClass::VariableMap &clientRegClass::getAllVariables()
{
    return myVariables;
}
covise::VRBClient * clientRegClass::getRegistryClient()
{
    return registry->getVrbc();
}
