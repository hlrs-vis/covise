/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "regClass.h"
#include "SessionID.h"
#include <net/message.h>
#include <net/message_types.h>
#include <vrbclient/VrbClientRegistry.h>
#include <vrbclient/VRBClient.h>

#include "SharedState.h"
#include "SharedStateSerializer.h"

using namespace covise;
namespace vrb
{
	regClass::regClass(const std::string& n, int ID)
		: name(n)
		, classID(ID)
		, isDel(false)
	{
	}
	/// get Class ID
	int regClass::getID()
	{
		return (classID);
	}
	void regClass::setID(int id)
	{
		classID = id;
	}
	const std::string& regClass::getName()
	{
		return (name);
	}
	///creates a  a regvar entry  in the map
	void regClass::append(regVar* var)
	{
		myVariables[var->getName()].reset(var);
	}
	/// getVariableEntry, returns NULL if not found
	regVar* regClass::getVar(const std::string& n)
	{
		auto it = myVariables.find(n);
		if (it == myVariables.end())
		{
			return (NULL);
		}
		return it->second.get();
	}
	/// remove a Variable
	void regClass::deleteVar(const std::string& n)
	{
		myVariables.erase(n);
	}
	/// remove some Variables
	void regClass::deleteAllNonStaticVars()
	{
		typename VariableMap::iterator it = myVariables.begin();
		while (it != myVariables.end())
		{
			if (it->second->isStatic())
			{
				it = myVariables.erase(it);
			}
			else
			{
				++it;
			}
		}
	}
	bool regClass::isDeleted()
	{
		return isDel;
	}
	void regClass::setDeleted(bool s)
	{
		isDel = s;
		for (const auto var : myVariables)
		{
			var.second->setDeleted(s);
		}
	}
	///write the classname and all variables in a .vrbreg file
	void regClass::writeClass(std::ofstream& file) {
		file << name;
		file << "\n";
		file << "{";
		file << "\n";
		for (const auto var : myVariables)
		{
			var.second->writeVar(file);
			file << "\n";
		}
		file << "}";

	}
	///reads the name and value out of stream, return false if class has no variable
	void regClass::readVar(std::ifstream& file)
	{

		while (true)
		{
			std::string varName = "invalid";
			int valueSize = -1;
			file >> varName;
			if (varName == "}")
			{
				return;
			}
			varName.pop_back();
			file >> valueSize;
			char* value = new char[valueSize];
			file.read(value, valueSize);
			covise::TokenBuffer tb(value, valueSize);
			myVariables[varName] = createVar(varName, std::move(tb));
			delete[] value; //createVar did copy the tokenbuffer
		}

	};
	////////////////////////////////REGVAR//////////////////////////
	regVar::regVar(regClass* c, const std::string& n, covise::TokenBuffer& v, bool s)
	{
		myClass = c;
		name = n;
		staticVar = s;
		setValue(v);
		isDel = false;
		lastPos = changedEtries.begin();
	}
	regVar::~regVar()
	{
		value.delete_data();
	}
	/// returns the value
	covise::TokenBuffer& regVar::getValue()
	{
		value.rewind();
		return (value);
	};
	/// returns the class of this variable
	regClass* regVar::getClass()
	{
		return (myClass);
	};
	/// set value
	inline void regVar::setValue(const covise::TokenBuffer& v)
	{
		if (myClass->getName() == "SharedMap")
		{
			std::shared_ptr<covise::TokenBuffer> tb;
			tb->copy(v);

			int type, pos;
			value >> type;
			switch ((ChangeType)type)
			{
			case vrb::WHOLE:
				value.copy(v);
				break;
			case vrb::ADD_ENTRY:
			{
				*tb.get() >> pos;
				int i = 0;
				int k = 0;
				EntryMap newChangedEntries;
				for (auto it = changedEtries.begin(); it != changedEtries.end(); ++it)
				{
					if (i = pos)
					{
						newChangedEntries[i] = tb;
						++i;
					}
					newChangedEntries[i] = changedEtries[k];
					++i;
					++k;
				}
			}
				break;
			case vrb::ENTRY_CHANGE:
			{
				*tb.get() >> pos;
				if (!lastPos->first == pos)
				{
					lastPos = changedEtries.find(pos);
				}
				lastPos->second = tb;
			}
				break;
			case vrb::ROMOVE_ENRY:
			{

			}
				break;
			default:
				std::cerr << "unexpected SharedMap change type: " << type << std::endl;
				break;
			}
		}
		else
		{
			value.copy(v);
		}


	}
	/// returns true if this Var is static
	int regVar::isStatic()
	{
		return (staticVar);
	};
	/// returns the Name
	const std::string& regVar::getName()
	{
		return (name);
	};

	bool regVar::isDeleted()
	{
		return isDel;
	}
	void regVar::setDeleted(bool isdeleted)
	{
		isDel = isdeleted;
	}
	void regVar::writeVar(std::ofstream& file)
	{
		file << "    " << name << "; ";
		int length = value.get_length();
		file << length;
		file.write(value.get_data(), value.get_length());
	}
	
	
	
	
	
	
	
	
	
	/////////////CLIENTREGVAR//////////////////////////////////////////////////
void clientRegVar::notifyLocalObserver()
{
    if (_observer)
    {
        _observer->update(this);
    }
}

void clientRegVar::subscribe(regVarObserver * ob, const SessionID &sessionID)
{
    _observer = ob;
    TokenBuffer tb;
    // compose message
    tb << sessionID;
    tb << myClass->getID();
    tb << myClass->getName();
    tb << name;
    tb << value;
    // inform vrb about creation
    dynamic_cast<clientRegClass *>(myClass)->sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE);
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
    classID = registry->getID();
    if (myVariables.size() == 0 && _observer)
    {
        subscribe(_observer, sessionID);
    }
    else
    {
        for (const auto var : myVariables)
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
    tb << classID;
    tb << name;

    // inform controller about creation
    if (classID >= 0)
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS);
}
clientRegClass::VariableMap &clientRegClass::getAllVariables()
{
    return myVariables;
}

std::shared_ptr<regVar> clientRegClass::createVar(const std::string &name, covise::TokenBuffer &&value)
{
    return std::shared_ptr<clientRegVar>(new clientRegVar(this, name, value));
}
}
