/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "regClass.h"
#include <net/message.h>
#include <net/message_types.h>
#include <vrbclient/VrbClientRegistry.h>
#include <vrbclient/VRBClient.h>

using namespace covise;
namespace vrb
{
/////////////CLIENTREGVAR//////////////////////////////////////////////////
void clientRegVar::notifyLocalObserver()
{
    if (_observer)
    {
        _observer->update(this);
    }
}

void clientRegVar::subscribe(regVarObserver * ob, int sessionID)
{
    myClass->setLastEditor(myClass->getID());
    lastEditor = myClass->getID();
    _observer = ob;
    TokenBuffer tb;
    // compose message
    tb << sessionID;
    tb << myClass->getID();
    tb << myClass->getName();
    tb << name;
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

void clientRegClass::resubscribe(int sessionID)
{
    classID = registry->getID();
    if (getRegistryClient() && sessionID != 0)
    {
        if (myVariables.size() == 0 && _observer)
        {
            subscribe(_observer, sessionID);
        }
        else
        {
            for (const auto var : myVariables)
            {
                if (var.second->getLocalObserver())
                {
                    var.second->subscribe(var.second->getLocalObserver(), sessionID);
                }
            }
        }
    }

}

void clientRegClass::subscribe(regClassObserver *obs, int sessionID)
{   
    lastEditor = classID;
    _observer = obs; //maybe inform old observer
    TokenBuffer tb;
    // compose message
    tb << sessionID;
    tb << classID;
    tb << name;

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
}
