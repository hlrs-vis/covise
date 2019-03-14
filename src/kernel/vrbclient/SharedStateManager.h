/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHAREDSTATEMANAGER_H
#define SHAREDSTATEMANAGER_H

#include "SharedState.h"
#include "SessionID.h"
#include <set>



namespace covise
{
class VRBClient;
}
namespace vrb
{
class VrbClientRegistry;
 
class VRBEXPORT SharedStateManager
{
public:
    SharedStateManager(VrbClientRegistry *reg);
    ~SharedStateManager();

    VrbClientRegistry *getRegistry();
    static SharedStateManager *instance();
    SessionID &add(SharedStateBase *base, SharedStateType mode);
    void remove(SharedStateBase *base);
    ///Updates the IDs to which the SharedStates send and from which they receive updates. 
    ///If force = true all SharedStates resubscribe, no matter if one of the IDs has changed  
    void update(SessionID &privateSessionID, SessionID & publicSessionID, SessionID & useCouplingModeSessionID, SessionID & sesisonToSubscribe, bool force = false);
    void frame(double time);
private:
    static SharedStateManager *s_instance;
    std::set<SharedStateBase *> useCouplingMode, alwaysShare, neverShare, shareWithAll;
    SessionID m_privateSessionID;
    SessionID m_publicSessionID;
    SessionID m_useCouplingModeSessionID;
    VrbClientRegistry *registry;
};
}
#endif