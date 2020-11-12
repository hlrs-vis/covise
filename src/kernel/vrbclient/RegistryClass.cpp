/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RegistryClass.h"
#include "SessionID.h"
#include "SharedStateSerializer.h"
#include <net/dataHandle.h>
#include <net/message.h>
#include <net/message_types.h>
#include <vrbclient/VRBClient.h>
#include <vrbclient/VrbClientRegistry.h>


using namespace covise;
namespace vrb
{
	regClass::regClass(const std::string& name, int ID)
		: m_name(name)
		, m_classID(ID)
	{
	}

    regClass::Iter regClass::begin(){
		return m_variables.begin();
	}
	regClass::Iter regClass::end(){
		return m_variables.end();
	}

	/// get Class ID
	int regClass::getID()
	{
		return m_classID;
	}
	void regClass::setID(int id)
	{
		m_classID = id;
	}
	const std::string& regClass::name() const
	{
		return m_name;
	}

	bool regClass::isMap() const{
		return m_name == sharedMapName;
	}
	///creates a  a regvar entry  in the map
	void regClass::append(regVar* var)
	{
		m_variables[var->name()].reset(var);
	}
	/// getVariableEntry, returns NULL if not found
	regVar* regClass::getVar(const std::string& n)
	{
		auto it = m_variables.find(n);
		if (it == m_variables.end())
		{
			return nullptr;
		}
		return it->second.get();
	}
	/// remove a Variable
	void regClass::deleteVar(const std::string& n)
	{
		m_variables.erase(n);
	}
	/// remove some Variables
	void regClass::deleteAllNonStaticVars()
	{
		typename Variables::iterator it = m_variables.begin();
		while (it != m_variables.end())
		{
			if (it->second->isStatic())
			{
				it = m_variables.erase(it);
			}
			else
			{
				++it;
			}
		}
	}
	bool regClass::isDeleted()
	{
		return m_isDel;
	}
	void regClass::setDeleted(bool s)
	{
		m_isDel = s;
		for (const auto var : m_variables)
		{
			var.second->setDeleted(s);
		}
	}

	void regClass::serialize(covise::TokenBuffer& tb) const
	{
		tb << m_name;
		tb << m_classID;
		tb << m_isDel;

		tb << (uint32_t)m_variables.size();
		for (const auto var : m_variables)
		{
			vrb::serialize(tb, *var.second);
		}
	};
	

	void regClass::deserialize(covise::TokenBuffer& tb){
		tb >> m_name;
		tb >> m_classID;
		tb >> m_isDel;

		uint32_t size;
		tb >> size;
		for (uint32_t i = 0; i < size; i++)
		{
			auto var = createVar("", DataHandle{});
			vrb::deserialize(tb, *var);
			m_variables[var->name()] = var;
		}
	}


template<>
void serialize(covise::TokenBuffer& tb, const regClass& value)
{
    value.serialize(tb);
}

template<>
void deserialize(covise::TokenBuffer& tb, regClass& value)
{
    value.deserialize(tb);
}
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
