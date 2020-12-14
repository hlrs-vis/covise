/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_CLIENT_REGCLASS_H
#define VRB_CLIENT_REGCLASS_H

#include <vrb/RegistryClass.h>
#include <net/message_types.h>
namespace covise
{
class DataHandle;
}

namespace vrb
{
class VRBClient;
class regClassObserver;
class VrbClientRegistry;
class SessionID;



class VRBCLIENTEXPORT clientRegClass : public regClass
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





class VRBCLIENTEXPORT regClassObserver
{
public:
    virtual void update(clientRegClass *theChangedClass) = 0;
};

}
#endif
