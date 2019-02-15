/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VrbClientRegistry_H
#define VrbClientRegistry_H

#include <net/tokenbuffer.h>
#include <map>
#include "regClass.h"
#include <util/coExport.h>
namespace covise
{
class VRBClient;
}
class VRBEXPORT VrbClientRegistry
{
public:
    static VrbClientRegistry *instance;
    /// construct a registry access path to the controller
    VrbClientRegistry(int id, covise::VRBClient *vrbc);
    ///gets id from server
    void setID(int clID, int session);
    void resubscribe(int odID);

    /**
       *  Subscribe to all variables in a registry class
       *
       *  @cl    registry class
       *  @ob    observer cl to be attached for updates
       *  @id    use clientID if id is not set
       */
    clientRegClass *subscribeClass(const std::string &clName, regClassObserver *ob, int id = -1);

    /**
       *  Subscribe to a specific variable of a registry cl
       *
       *  @cl    registry class
       *  @var      variable in registry cl
       *  @ob       observer cl to be attached for updates
       */
    clientRegVar *subscribeVar(const std::string &cl, const std::string &var, covise::TokenBuffer &&value, regVarObserver *ob, int id);

    /**
       *  Unsubscribe from a registry cl (previously subscribed with subscribecl)
       *
       *  @cl    registry cl
       *
       */
    void unsubscribeClass(const std::string &cl);

    /**
       *  Unsubscribe from a specific variable of a registry cl
       *
       *  @cl      registry class
       *  @var        registry variable belonging to the cl
       *
       */
    void unsubscribeVar(const std::string &cl, const std::string &var);

    /**
       *  Create a specific class variable in the registry. If the class
       *  variable is already existing the operation is ignored. If the class
       *  for the given variable is not existing, the cl is created in the registry.
       *
       *  @cl  registry class
       *  @var    registry variable belonging to the cl
       *  @flag   flag=0: session local variable, flag=1: global variable surviving a session
       */
    void createVar(const std::string &cl, const std::string &var, covise::TokenBuffer &value, bool isStatic = false);

    /**
       *  Sets a specific variable value in the registry. The Vrb server
       *  is informed about the change
       *
       *  @cl  registry class
       *  @var    registry variable belonging to the cl
       *  @val    current variable value to be set in the registry
       */
    void setVar(int ID, const std::string &cl, const std::string &var, covise::TokenBuffer &&val);

    /**
       *  Destroys a specific variable in the registry. All observers attached
       *  to this variable will be notified immediately about deletion.
       *  Note: It is normally not necessary to destroy variables no longer used!
       *
       *  @cl  registry class
       *  @var    registry variable belonging to a cl
       */
    void destroyVar(const std::string &cl, const std::string &var);

    /**
       *  if a VRB connected, resend local variables and subscriptions.
       */
    void updateVRB();

    clientRegClass *getClass(const std::string &name);

    void update(covise::TokenBuffer &tb, int reason);

    virtual ~VrbClientRegistry();
    void sendMsg(covise::TokenBuffer &tb, int message_type);

    int getID()
    {
        return clientID;
    }
    int getSession()
    {
        return sessionID;
    }
    covise::VRBClient *getVrbc();
    void setVrbc(covise::VRBClient *client);

private:
    std::map<const std::string, std::shared_ptr<clientRegClass>> myClasses;
    int clientID;
    int sessionID;
    covise::VRBClient *vrbc;

};



#endif
