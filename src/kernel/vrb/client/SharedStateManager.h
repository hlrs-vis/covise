/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHAREDSTATEMANAGER_H
#define SHAREDSTATEMANAGER_H

#include "SharedState.h"

#include <vrb/SessionID.h>

#include <set>

namespace vrb
{
class VRBClient;
class VrbClientRegistry;
 ///Manages the behaviour of all sharedStates depending on their sharedStateType
class VRBCLIENTEXPORT SharedStateManager
{
public:
    SharedStateManager(VrbClientRegistry *reg);
    ~SharedStateManager();

    VrbClientRegistry *getRegistry();
    static SharedStateManager *instance();
    ///adds the sharedState to a list depending on its type; Returns the current sessionID and the mute state
    std::pair<SessionID, bool> add(SharedStateBase *base, SharedStateType mode);
    ///removes the sharedState from the list it is in
    void remove(SharedStateBase *base);
    ///Updates the IDs to which the SharedStates send and from which they receive updates. 
    ///muted sharedStates will update the local registry but will not send information to vrb
    ///If force = true all SharedStates resubscribe, no matter if one of the IDs has changed  
    void update(SessionID &privateSessionID, SessionID & publicSessionID, bool muted, bool force = false);
    ///unmutes all sharedStates and makes them send their current value to vrb
    void becomeMaster();
    ///lets sharedStates send their value if it has changed and their syncInterval allows it
    void frame(double time);
private:
    static SharedStateManager *s_instance;
    std::set<SharedStateBase *> useCouplingMode, alwaysShare, neverShare, shareWithAll;
    SessionID m_privateSessionID;
    SessionID m_publicSessionID;
    bool m_muted = false;
    VrbClientRegistry *registry = nullptr;
};
}
#endif