/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <map>
#include <memory>

#include <net/tokenbuffer.h>

#include <vrbclient/VrbRegistry.h>
#include <vrbclient/regClass.h>
#include <VrbClientList.h>
#include <vrbclient/SessionID.h>



#ifndef VrbClientRegistry_H
#define VrbClientRegistry_H


namespace covise
{
class TokenBuffer;
class DataHandle;
}

namespace vrb
{
class serverRegVar;

class VrbServerRegistry : public VrbRegistry
{
public:
    /// constructor initializes Variables with values from yac.config:regVariables
    VrbServerRegistry(SessionID &session);


    /// set a Value or create new Entry, s for isStatic
    void setVar(int ID, const std::string &className, const std::string &name, const covise::DataHandle &value, bool s = false);
    /// create new Entry
    void create(int ID, const std::string &className, const std::string &name, const covise::DataHandle &value, bool s);
    /// remove an Entry
    void deleteEntry(const std::string &className, const std::string &name);
    /// remove all Entries from one Module
    void deleteEntry();
    ///add sender sa observer to every vaiable and every class that has no variables
    void observe(int sender);
    /// add a new observer to a specific variable and provide a default value
    void observeVar(int ID, const std::string &className, const std::string &variableName, const covise::DataHandle &value);
    ///add a observer to a class an all its variables
    void observeClass(int ID, const std::string &className);
    /// remove an observer
    void unObserveVar(int ID, const std::string &className, const std::string &variableName);
    ///remove observer from class and all its variables
    void unObserveClass(int ID, const std::string &className);
    /// remove all observers for this ID
    void unObserve(int recvID);
    ///informs the observers about a variable change
    void sendVariableChange(serverRegVar *rv, std::set<int> observers);
    ///Updates the <ui of the vrb server
    void updateUI(serverRegVar *rv);
    /// get a boolean Variable
    int isTrue(int ID, const std::string &className, const std::string &name, int def = 0);
    /**
       * add Registry to Script
       */
    void saveNetwork(coCharBuffer &cb);
    void setOwner(int id);
    int getOwner();

    int getID() override
    {
        return -1;
    }
    std::shared_ptr<regClass> createClass(const std::string &name, int id) override;
private:
    SessionID sessionID;
    int owner;
};

class serverRegVar : public regVar
{
private:
    std::set<int> observers;
public:

    using regVar::regVar;
    ~serverRegVar();
    /// send Value to recvID
    void update(int recvID);
	///updatafunction for SharedMaps
	void updateMap(int recvID);
    /// send Value UIs depending on UI variable RegistryMode
    void updateUIs();
    /// add an observer to my list
    void observe(int recvID)
    {
        observers.insert(recvID);
    };
    /// remove an observer to my list
    void unObserve(int recvID)
    {
        observers.erase(recvID);
    };
    /// get list of Observers
    std::set<int> &getOList()
    {
        return observers;
    };
    void informDeleteObservers();

};

class serverRegClass : public regClass
{
private:
    std::set<int> observers; // clients
public:

    using regClass::regClass;
    /// observe all variables or this clas if it has o variables
    void observeAllVars(int sender);
    /// add a new observer to this class and all of its variables
    void observe(int recvID);
    ///add Observer to a specific variable
    void observeVar(int recvID, const std::string &variableName, const covise::DataHandle &value);
    /// remove an observer from this class and variable of this class. 
    void unObserveVar(int recvID, const std::string &variableName);
    ///remove the observer from all variables
    void unObserve(int recvID);
    /// get list of Observers
	std::set<int>& getOList()
	{
		return observers;
	};
    void informDeleteObservers();
    std::shared_ptr<regVar> createVar(const std::string &name, const covise::DataHandle &value);

};
}

#endif
