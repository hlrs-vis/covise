/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REGCLASS_H
#define REGCLASS_H

#include "dataHandle.h"

#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include <util/coExport.h>
#include <fstream>

class coCharBuffer;
namespace covise
{
class VRBClient;
}

namespace vrb
{
class regVar;
class regClassObserver;
class regVarObserver;
class VrbClientRegistry;
class SessionID;


class VRBEXPORT regClass
{
public:
    typedef std::map<const std::string, std::shared_ptr<regVar>> VariableMap;
	regClass(const std::string& n, int ID);
    virtual ~regClass();

    /// get Class ID
	int getID();

	void setID(int id);

	const std::string& getName();

    ///creates a  a regvar entry  in the map
	void append(regVar* var);

    /// getVariableEntry, returns NULL if not found
	regVar* getVar(const std::string& n);

    /// remove a Variable
	void deleteVar(const std::string& n);

    /// remove some Variables
	void deleteAllNonStaticVars();

	bool isDeleted();

	void setDeleted(bool isdeleted = true);

    ///write the classname and all variables in a .vrbreg file
	void writeClass(std::ofstream& file);

    ///reads the name and value out of stream, return false if class has no variable
	void readVar(std::ifstream& file);

    virtual std::shared_ptr<regVar> createVar(const std::string &name, DataHandle &value) = 0;

protected:
    std::string name;
    int classID = -1;
    bool isDel;
    VariableMap myVariables;
};


class VRBEXPORT regVar
{
protected:


    std::string name;
    regClass *myClass;
    bool staticVar;
    bool isDel;
	///writes value to tb
	void sendValueChange(covise::TokenBuffer& tb);
	///writes value to tb, in case of SahredMap also writes all changes
	void sendValue(covise::TokenBuffer& tb);

	DataHandle value;

public:
	//for SahredMaps
	typedef std::map<int, DataHandle> EntryMap;
	DataHandle wholeMap;
	EntryMap m_changedEtries;

	regVar(regClass* c, const std::string& n, DataHandle & v, bool s = 1);

	virtual ~ regVar();

	/// returns the value
	const DataHandle& getValue() const;

    /// returns the class of this variable
	regClass* getClass();

	/// set value
	void setValue(const DataHandle& v);

    /// returns true if this Var is static
	int isStatic();

    /// returns the Name
	const std::string& getName();


	bool isDeleted();

	void setDeleted(bool isdeleted = true);

	void writeVar(std::ofstream& file);

};


class VRBEXPORT clientRegClass : public regClass
{
private:
    regClassObserver *_observer = nullptr; //local observer class
    int lastEditor;
    VrbClientRegistry *registry;
public:
    void sendMsg(covise::TokenBuffer &tb, covise::covise_msg_type type);
    clientRegClass(const std::string &n, int ID, VrbClientRegistry *reg);
    regClassObserver *getLocalObserver()
    {
        return _observer;
    }
    ///attach a observer to the regClass
    void attach(regClassObserver *ob)
    {
        _observer = ob;
    }
    int getLastEditor()
    {
        return lastEditor;
    }
    void setLastEditor(int lastEditor);
    void notifyLocalObserver();
    void resubscribe(const SessionID &sessionID);
    void subscribe(regClassObserver *obs, const SessionID &sessionID);
    VariableMap &getAllVariables();
    std::shared_ptr<regVar> createVar(const std::string &name, DataHandle &value) override;
};
class VRBEXPORT clientRegVar : public regVar
{
private:
    regVarObserver *_observer;
    int lastEditor;
public:
    using regVar::regVar;
    ///returns the clent side observer
    regVarObserver * getLocalObserver()
    {
        return _observer;
    }
    void notifyLocalObserver();
    void subscribe(regVarObserver *ob, const SessionID &sessionID);

    //void attach(regVarObserver *ob)
    //{
    //    _observer = ob;
    //}
    int getLastEditor()
    {
        return lastEditor;
    }
    void setLastEditor(int lastEditor)
    {
        this->lastEditor = lastEditor;
    }
};




class VRBEXPORT regClassObserver
{
public:
    virtual void update(clientRegClass *theChangedClass) = 0;
};
class VRBEXPORT regVarObserver
{
public:
    virtual void update(clientRegVar *theChangedVar) = 0;
};
}
#endif
