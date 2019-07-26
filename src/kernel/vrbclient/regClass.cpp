/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "regClass.h"
#include "SessionID.h"
#include <net/message.h>
#include <net/message_types.h>
#include <net/dataHandle.h>
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

    regClass::~regClass()
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
			if (name == sharedMapName)
			{
				//nur fertiges datahandle übergeben
				covise::TokenBuffer outerTb;
				outerTb << WHOLE;
				//read serialized map data
				file >> valueSize;
				char* value = new char[valueSize];
				file.read(value, valueSize);
				covise::TokenBuffer innerTb(value, valueSize);
				outerTb << innerTb;
				//read changes of map
				file >> valueSize;
				std::map<int, DataHandle> changes;
				for (size_t i = 0; i < valueSize; i++)
				{
					int pos, size;
					file >> pos;
					file >> size;
					char* v = new char[size];
					file.read(v, size);
					changes[pos] = DataHandle(v, size);
				}
				serialize(outerTb, changes);
                auto l = outerTb.get_length();
                myVariables[varName] = createVar(varName, DataHandle(outerTb.take_data(), l));
				delete[] value;
			}
			else
			{
			file >> valueSize;
			char* value = new char[valueSize];
			file.read(value, valueSize);
			DataHandle valueData(value, valueSize);
			myVariables[varName] = createVar(varName, valueData);
			}

		}

	};
	void regVar::sendValueChange(covise::TokenBuffer& tb)
	{
		serialize(tb, value);
	}
	void regVar::sendValue(covise::TokenBuffer& tb)
	{

		if (myClass->getName() == sharedMapName)
		{
			covise::TokenBuffer v;
			covise::TokenBuffer serializedMap(wholeMap.data(), wholeMap.length());
			v << (int)vrb::WHOLE;
			v << serializedMap;
			serialize(v, m_changedEtries);
			tb << v;
		}
		else
		{
			sendValueChange(tb);
		}
	}
	////////////////////////////////REGVAR//////////////////////////
	regVar::regVar(regClass* c, const std::string& n, const DataHandle & v, bool s)
	{
		myClass = c;
		name = n;
		staticVar = s;
		setValue(v);
		isDel = false;

		//std::string t("test string");
		//covise::TokenBuffer tb1;
		//tb1 << t;
		//DataHandle dh(tb1);
		//covise::TokenBuffer tb2(dh.data(), dh.length());
		//t = "";
		//tb2 >> t;
		//covise::TokenBuffer tb3;
		//serialize(tb3, dh);
		//tb3.rewind();
		//deserialize(tb3, dh);
		//t = "";
		//covise::TokenBuffer tb4(dh.data(), dh.length());
		//tb4 >> t;
		
	}
	regVar::~regVar()
	{
	}
	/// returns the value
	const DataHandle& regVar::getValue() const
	{
		return value;
	};
	/// returns the class of this variable
	regClass* regVar::getClass()
	{
		return (myClass);
	};
	/// set value
	void regVar::setValue(const DataHandle& v)
	{
		value = v;
		if (myClass->getName() == sharedMapName)
		{
			covise::TokenBuffer  tb(v.data(), v.length());
			int type, pos;
			tb >> type;
			switch ((ChangeType)type)
			{
			case vrb::WHOLE:
			{
				covise::TokenBuffer m;
				tb >> m;
                auto l = m.get_length();
                wholeMap = DataHandle(m.take_data(), l);
				m_changedEtries.clear();
				deserialize(tb, m_changedEtries); //should be empty after complete map was send from ckient, may be filled after session was loaded from file
				break;
			}
			case vrb::ENTRY_CHANGE:
			{
				tb >> pos;
				m_changedEtries[pos] = v;
			}
				break;
			default:
				std::cerr << "unexpected SharedMap change type: " << type << std::endl;
				break;
			}
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
		if (myClass->getName() == sharedMapName)
		{
			file << wholeMap.length();
			file.write(wholeMap.data(), wholeMap.length());
			file << (int)m_changedEtries.size();
			for (auto change : m_changedEtries)
			{
				file << change.first;
				file << change.second.length();
				file.write(change.second.data(), change.second.length());
			}
		}
		else
		{

			int length = value.length();
			file << length;
			file.write(value.data(), value.length());
		}
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
    sendValue(tb);
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

std::shared_ptr<regVar> clientRegClass::createVar(const std::string &name, const DataHandle &value)
{
    return std::shared_ptr<clientRegVar>(new clientRegVar(this, name, value));
}
}
