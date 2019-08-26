/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <assert.h>

#include "VRBClient.h"
#include "VrbMessageSenderInterface.h"
#include "SharedStateSerializer.h"
#include "VrbClientRegistry.h"

#include <net/message.h>
#include <net/dataHandle.h>
#include <net/message_types.h>




using namespace covise;
namespace vrb
{
VrbClientRegistry *VrbClientRegistry::instance = NULL;
//==========================================================================
//
// class VrbClientRegistry
//
//==========================================================================
VrbClientRegistry::VrbClientRegistry(int ID)
    :clientID(ID)
	,m_sender(nullptr)
{
    assert(!instance);
    instance = this;
}
void VrbClientRegistry::registerSender(VrbMessageSenderInterface* sender)
{
	m_sender = sender;
}

VrbClientRegistry::~VrbClientRegistry()
{
    // delete entry list including all entries!
    instance = NULL;
}

void VrbClientRegistry::setID(int clID, const SessionID &session)
{
    if (sessionID == session && clientID == clID)
    {
        //reconect to old session
    }
    else
    {
        sessionID = session;
        clientID = clID;
        for (const auto cl : myClasses)
        {
            cl.second->setID(clientID);
        }
    }
}

void VrbClientRegistry::resubscribe(const SessionID &sessionID, const SessionID &oldSession)
{
    if (!oldSession.isPrivate()) //unobserve old public session
    {
		if (!m_sender)
		{
			return;
		}
		covise::TokenBuffer tb;
        tb << oldSession;
        tb << clientID;

        m_sender->sendMessage(tb, COVISE_MESSAGE_VRBC_UNOBSERVE_SESSION);
    }
    // resubscribe all registry entries on reconnect
    for (const auto cl : myClasses)
    {
        if (cl.first != "SharedState")
        {
           std::dynamic_pointer_cast<vrb::clientRegClass>(cl.second)->resubscribe(sessionID);
        }
    }
}

void VrbClientRegistry::sendMsg(TokenBuffer &tb, covise::covise_msg_type type)
{
	if (!m_sender)
	{
		return;
	}
	if (clientID != -1)
    {
        m_sender->sendMessage(tb, type);
    }
}

clientRegClass *VrbClientRegistry::subscribeClass(const SessionID &sessionID, const std::string &cl, regClassObserver *ob)
{
    clientRegClass *rc = getClass(cl);
    if (!rc) //class does not exist
    {
        rc = new clientRegClass(cl, clientID, this);
        myClasses[cl].reset(rc);
    }
    rc->subscribe(ob, sessionID);
    return rc;
}

clientRegVar *VrbClientRegistry::subscribeVar(const SessionID &sessionID, const std::string &cl, const std::string &var, const DataHandle &value, regVarObserver *ob)
{
    // attach to the list
    if (var == "VRVMenue_testTest")
    {
        std::cerr << "debug" << var << std::endl;
    }
    clientRegClass *rc = getClass(cl);
    if (!rc) //class does not exist
    {
        rc = new clientRegClass(cl, clientID, this);
        myClasses[cl].reset(rc);
    }
    clientRegVar *rv = dynamic_cast<clientRegVar*>(rc->getVar(var));
    if (!rv)
    {
        rv = new clientRegVar(rc, var, value);
        rc->append(rv);
    }
    //maybe inform old observer here
    rv->subscribe(ob, sessionID);
    return rv;
}

void VrbClientRegistry::unsubscribeClass(const SessionID &sessionID, const std::string &cl)
{
    clientRegClass *rc = getClass(cl);
    if (rc)
    {
        TokenBuffer tb;
        // compose message
        tb << sessionID;
        tb << clientID;
        tb << cl;
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS);
        myClasses.erase(cl);

    }
}

void VrbClientRegistry::unsubscribeVar(const std::string &cl, const std::string &var, bool unsubscribeServerOnly)
{
    clientRegClass *rc = getClass(cl);
    if (rc)
    {
        if (rc->getVar(var))
        {
            //found
            TokenBuffer tb;
            // compose message
            tb << clientID;
            tb << cl;
            tb << var;
            sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE);
            if (!unsubscribeServerOnly)
            {
                rc->deleteVar(var);
            }
        }
    }
}

void VrbClientRegistry::createVar(const SessionID sessionID, const std::string &cl, const std::string &var, const covise::DataHandle& value, bool isStatic)
{

    // compose message
    TokenBuffer tb;
    tb << sessionID;
    tb << clientID;
    tb << cl;
    tb << var;
    serialize(tb, value);
    tb << isStatic;
    sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY);
}

void VrbClientRegistry::setVar(const SessionID sessionID, const std::string &cl, const std::string &var, const DataHandle  &value, bool muted)
{
    // attach to the list
    clientRegClass *rc = getClass(cl);
    if (!rc)
    {
        return; //maybe create class
    }
    rc->setLastEditor(clientID);
    clientRegVar *rv = dynamic_cast<clientRegVar*>(rc->getVar(var));
    if (!rv)
    {
        rv = new clientRegVar(rc, var, value);
    }
    else
    {
        rv->setValue(value);
        rv->setLastEditor(clientID);
    }
    if (muted)
    {
        return;
    }
    // compose message
    TokenBuffer tb;
    tb << sessionID;
    tb << clientID;// local client ID
    tb << cl;
    tb << var;
	serialize(tb, value);

    sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE);
}

void VrbClientRegistry::destroyVar(const SessionID sessionID, const std::string &cl, const std::string &var)
{
    clientRegClass * rc = getClass(cl);
    if (rc)
    {
        clientRegVar *rv = dynamic_cast<clientRegVar*>(rc->getVar(var));
        if (rv)
        {
            rc->deleteVar(var);
            // compose message
            TokenBuffer tb;
            tb << sessionID;
            tb << cl;
            tb << var;
            sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY);
        }
    }
}

clientRegClass *VrbClientRegistry::getClass(const std::string &name)
{
    auto it = myClasses.find(name);
    if (it == myClasses.end())
    {
        return (NULL);
    }
    return std::dynamic_pointer_cast<vrb::clientRegClass>(it->second).get();
}
// called by controller if registry entry has changed
void VrbClientRegistry::update(TokenBuffer &tb, int reason)
{
    int senderID;
    std::string cl;
    std::string var;
    clientRegClass *rc;
    clientRegVar *rv;
    tb >> senderID;
    tb >> cl;
    tb >> var;
    switch (reason)
    {

    case COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED:
	{

		DataHandle  valueData;
		deserialize(tb, valueData);

		// call all class specific observers
		rc = getClass(cl);
		if (rc)
		{
			rc->setDeleted(false);
			rv = dynamic_cast<clientRegVar*>(rc->getVar(var));
			if (rv)
			{
				//inform var observer if not receiving my own message
				rv->setDeleted(false);
				if (!(rv->getLastEditor() == clientID && senderID == clientID))
				{
					rv->setLastEditor(senderID);
					rv->setValue(valueData);
					rv->notifyLocalObserver();
				}
			}
			//inform class observer if not receiving my own message
			if (!(rc->getLastEditor() == clientID && senderID == clientID))
			{
				rc->notifyLocalObserver();
			}
		}
	}
        break;

    case COVISE_MESSAGE_VRB_REGISTRY_ENTRY_DELETED:

        rc = getClass(cl);
        if (rc)
        {
            if (var.empty())//delete class if no variable is submitted
            {
                rc->setDeleted();
                rc->notifyLocalObserver();
                return;
            }
            rv = dynamic_cast<clientRegVar*>(rc->getVar(var));
            if (rv)
            {
                rv->setDeleted();
                rc->notifyLocalObserver();
                rv->notifyLocalObserver();
            }

        }
        break;

    } // switch
}


std::shared_ptr<regClass> VrbClientRegistry::createClass(const std::string &name, int id)
{
    return std::shared_ptr<clientRegClass>(new clientRegClass(name, id, this));
}
}

