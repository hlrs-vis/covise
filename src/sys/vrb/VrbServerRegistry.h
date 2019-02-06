/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <net/tokenbuffer.h>
#include <map>
#include <vrbclient/VrbRegistry.h>
#include <vrbclient/regClass.h>
#include <VRBClientList.h>


#ifndef VrbClientRegistry_H
#define VrbClientRegistry_H

class coVrbRegEntryObserver;
class coCharBuffer;
namespace covise
{
class TokenBuffer;
}


class VrbServerRegistry 
{
   public:
       static VrbServerRegistry *instance;
       /// constructor initializes Variables with values from yac.config:regVariables
        VrbServerRegistry(int session);
        ~VrbServerRegistry();
        int regMode;


    std::map<int, std::shared_ptr<serverRegClass>> getClasses(const std::string &name);
    /// get a map with an entry of the specified classes of all clients if the have that class
///int : client id, regClass : regClass with name that belongs to that client
    serverRegClass *getClass(int ID, const std::string &name);
    /// set a Value or create new Entry, s for isStatic
    void setVar(int ID, const std::string &className, const std::string &name, covise::TokenBuffer &value, bool s = false);
    /// create new Entry
    void create(int ID, const std::string &className, const std::string &name, covise::TokenBuffer &value, bool s);
    /// remove an Entry
    void deleteEntry(int ID, const std::string &className, const std::string &name);
    /// remove all Entries from one Module
    void deleteEntry(int moduleID);
    /// add a new observer to a specific variable and provide a default value
    void observeVar(int ID, const std::string &className, int recvID, const std::string &variableName, covise::TokenBuffer &value);
    ///add a observer to a class an all its variables
    void observeClass(int ID, const std::string &className, int recvID);
    /// remove an observer
    void unObserve(int ID, const std::string &className, int recvID, const std::string &variableName = NULL);
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
    int getSessionID()
    {
        return sessionID;
    }
    void setSessionID(int newID)
    {
        sessionID = newID;
    }
private:
    int sessionID;
    VRBClientList clients;
    std::map<const int, std::map<const std::string, std::shared_ptr<serverRegClass>>> clientsClasses;
};

class serverRegVar : public regVar<serverRegClass>
{
private:

public:
    std::set<int> observers;
    using regVar::regVar;
    ~serverRegVar();
    /// send Value to recvID
    void update(int recvID);
    /// send Value UIs depending on UI variable RegistryMode
    void updateUIs();
    /// add an observer to my list
    void observe(int recvID)
    {
        observers.insert(recvID);
        update(recvID);
    };
    /// remove an observer to my list
    void unObserve(int recvID)
    {
        observers.erase(recvID);
    };
    /// get list of Observers
    std::set<int> *getOList()
    {
        return (&observers);
    };
    void informDeleteObservers();
};

class serverRegClass : public regClass<serverRegVar>
{
private:
public:
    std::set<int> observers; //other clients
    using regClass::regClass;
    /// add a new observer to this class and all of its variables
    void observe(int recvID);
    ///add Observer to a specific variable
    void observeVar(int recvID, const std::string &variableName, covise::TokenBuffer &value);
    /// remove an observer from this class and variable of this class. 
    void unObserveVar(int recvID, const std::string &variableName);
    ///remove the observer from all variables
    void unObserve(int recvID);
    /// get list of Observers
    std::set<int> *getOList()
    {
        return (&observers);
    }
    void informDeleteObservers();
};


#endif
