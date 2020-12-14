/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VrbClientRegistry_H
#define VrbClientRegistry_H

#include "ClientRegistryClass.h"
#include "ClientRegistryVariable.h"

#include <net/tokenbuffer.h>
#include <util/coExport.h>
#include <vrb/Registry.h>
#include <vrb/SessionID.h>

#include <map>

namespace covise
{
class DataHandle;
class MessageSenderInterface;
}
namespace vrb
{
class VRBClient;
class VRBCLIENTEXPORT VrbClientRegistry : public VrbRegistry
{
public:
    static VrbClientRegistry *instance;
    /// construct a registry access path to the controller
    VrbClientRegistry(int id);
	void registerSender(covise::MessageSenderInterface* sender);
    ///gets id from server
    void setID(int clID, const SessionID &session);
    ///unsubscribe all clases and variables from old session and subscribe to the new one (ignore sharedStates, they resubscribe them selves)
    void resubscribe(const SessionID &sessionID, const SessionID &oldSession = SessionID());

    /**
       *  Subscribe to all variables in a registry class
       *
       *  @cl    registry class
       *  @ob    observer cl to be attached for updates
       *  @id    use clientID if id is not set
       */
    clientRegClass *subscribeClass(const SessionID &sessionID, const std::string &clName, regClassObserver *ob);

    /**
       *  Subscribe to a specific variable of a registry cl
       *
       *  @cl    registry class
       *  @var      variable in registry cl
       *  @ob       observer cl to be attached for updates
       */
    clientRegVar *subscribeVar(const SessionID &sessionID, const std::string &cl, const std::string &var, const covise::DataHandle &value, regVarObserver *ob);

    /**
       *  Unsubscribe from a registry cl (previously subscribed with subscribecl)
       *
       *  @cl    registry cl
       *
       */
    void unsubscribeClass(const SessionID &sessionID, const std::string &cl);

    /**
       *  Unsubscribe from a specific variable of a registry cl
       *  Stop observing this variable on the VRB
       *  @cl      registry class
       *  @var        registry variable belonging to the cl
       *  @unsubscribeServerOnly    if true, the variable entry is kept in the local registry and only the observation on the server ends
       */
    void unsubscribeVar(const std::string &cl, const std::string &var, bool unsubscribeServerOnly = false);

    /**
       *  Create a specific class variable in the registry. If the class
       *  variable is already existing the operation is ignored. If the class
       *  for the given variable is not existing, the cl is created in the registry.
       *
       *  @cl  registry class
       *  @var    registry variable belonging to the cl
       *  @flag   flag=0: session local variable, flag=1: global variable surviving a session
       */
    void createVar(const SessionID sessionID, const std::string &cl, const std::string &var, const covise::DataHandle &value, bool isStatic = false);

    /**
       *  Sets a specific variable value in the registry. The Vrb server
       *  is informed about the change
       *
       *  @cl  registry class
       *  @var    registry variable belonging to the cl
       *  @val    current variable value to be set in the registry
       */
    void setVar(const SessionID sessionID, const std::string &cl, const std::string &var, const covise::DataHandle &val, bool muted = false);

    /**
       *  Destroys a specific variable in the registry. All observers attached
       *  to this variable will be notified immediately about deletion.
       *  Note: It is normally not necessary to destroy variables no longer used!
       *
       *  @cl  registry class
       *  @var    registry variable belonging to a cl
       */
    void destroyVar(const SessionID sessionID, const std::string &cl, const std::string &var);

    /**
       *  if a VRB connected, resend local variables and subscriptions.
       */

///returns  the regClass with name or nullptr
    clientRegClass *getClass(const std::string &name);

    void update(covise::TokenBuffer &tb, int reason);

    virtual ~VrbClientRegistry();
    void sendMsg(covise::TokenBuffer &tb, covise::covise_msg_type type);

    int getID() override
    {
        return clientID;
    }
    SessionID getSession()
    {
        return sessionID;
    }

    std::shared_ptr<regClass> createClass(const std::string &name, int id) override;
private:
    int clientID = -1;
    SessionID sessionID;
    covise::MessageSenderInterface *m_sender = nullptr;

};
}


#endif
