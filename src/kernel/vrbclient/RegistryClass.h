/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REGCLASS_H
#define REGCLASS_H

#include "RegistryVariable.h"
#include "SharedStateSerializer.h"

#include <net/dataHandle.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <util/coExport.h>

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <vector>

class coCharBuffer;
namespace covise
{
class VRBClient;
class DataHandle;
}

namespace vrb
{
class regClassObserver;
class VrbClientRegistry;
class SessionID;

constexpr char sharedMapName[] = "SharedMap";

class VRBEXPORT regClass
{
public:
    typedef std::map<const std::string, std::shared_ptr<regVar>> Variables;
    typedef Variables::const_iterator Iter;
    regClass(const std::string &name = "", int ID = -1);
    virtual ~regClass() = default;

    Iter begin();
    Iter end();

    /// get Class ID
	int getID();

	void setID(int id);

    const std::string &name() const;
    bool isMap() const;
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


	void serialize(covise::TokenBuffer& file) const;
	void deserialize(covise::TokenBuffer& file);

    virtual std::shared_ptr<regVar> createVar(const std::string &m_name, const covise::DataHandle &value) = 0;

protected:
    std::string m_name;
    int m_classID = -1;
    bool m_isDel = false;
    Variables m_variables;
};

template <>
void serialize(covise::TokenBuffer &tb, const regClass &value);

template <>
void deserialize(covise::TokenBuffer &tb, regClass &value);

class VRBEXPORT clientRegClass : public regClass
{

public:
    clientRegClass(const std::string &name, int ID, VrbClientRegistry *reg);
    void sendMsg(covise::TokenBuffer &tb, covise::covise_msg_type type);
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
    Variables &getAllVariables();
    std::shared_ptr<regVar> createVar(const std::string &m_name, const covise::DataHandle &value) override;
private:
    regClassObserver *_observer = nullptr; //local observer class
    int lastEditor = -1;
    VrbClientRegistry *registry = nullptr;
};





class VRBEXPORT regClassObserver
{
public:
    virtual void update(clientRegClass *theChangedClass) = 0;
};

}
#endif
