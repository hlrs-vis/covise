/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHAREDSTATEMANAGER_H
#define SHAREDSTATEMANAGER_H

#include "SharedState.h"

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
    int add(SharedStateBase *base, SharedStateType mode);
    void remove(SharedStateBase *base);
    void update(int privateSessionID, int publicSessionID, int useCouplingModeSessionID, int sesisonToSubscribe);
private:
    static SharedStateManager *s_instance;
    std::set<SharedStateBase *> useCouplingMode, alwaysShare, neverShare;
    int m_privateSessionID, m_publicSessionID, m_useCouplingModeSessionID = 0;
    VrbClientRegistry *registry;
};
}
#endif