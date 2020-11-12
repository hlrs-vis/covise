/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ClientRegistryClass.h"
#include "VrbClientRegistry.h"
#include "ClientRegistryVariable.h"

#include <net/dataHandle.h>
#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer_serializer.h>
#include <vrb/SessionID.h>


using namespace covise;
namespace vrb
{
	
/////////////CLIENTREGCLASS/////////////////////////////////////////////////

void clientRegClass::sendMsg(covise::TokenBuffer & tb, covise::covise_msg_type type)
{
    registry->sendMsg(tb, type);
}

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

void clientRegClass::resubscribe(const SessionID &sessionID)
{
    m_classID = registry->getID();
    if (m_variables.size() == 0 && _observer)
    {
        subscribe(_observer, sessionID);
    }
    else
    {
        for (const auto var : m_variables)
        {
			std::shared_ptr<clientRegVar> v = std::dynamic_pointer_cast<clientRegVar>(var.second);
			if (v->getLocalObserver())
            {
                v->subscribe(v->getLocalObserver(), sessionID);
            }
        }
    }

}

void clientRegClass::subscribe(regClassObserver *obs, const SessionID &sessionID)
{   
    _observer = obs; //maybe inform old observer
    TokenBuffer tb;
    // compose message
    tb << sessionID;
    tb << m_classID;
    tb << m_name;

    // inform controller about creation
    if (m_classID >= 0)
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS);
}
clientRegClass::Variables &clientRegClass::getAllVariables()
{
    return m_variables;
}

std::shared_ptr<regVar> clientRegClass::createVar(const std::string &name, const DataHandle &value)
{
    return std::shared_ptr<clientRegVar>(new clientRegVar(this, name, value));
}
}
