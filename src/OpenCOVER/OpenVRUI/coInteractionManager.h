/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_INTERACTIONMANAGER
#define CO_INTERACTIONMANAGER

#include <map>
#include <memory>
#include <list>

#include <OpenVRUI/coInteraction.h>

#include <vrb/client/SharedState.h>

namespace vrui
{

class OPENVRUIEXPORT coInteractionManager
{
public:
    explicit coInteractionManager();
	coInteractionManager(const coInteractionManager&) = delete;
	coInteractionManager& operator=(const coInteractionManager&) = delete;
	~coInteractionManager();
    //virtual ~coInteractionManager();

    bool update();
	//initialize the sheredState for remote locking this group if neccecary
	void registerGroup(int group);
    void registerInteraction(coInteraction *);
    void unregisterInteraction(coInteraction *);
    bool isOneActive(coInteraction::InteractionType type);
    bool isOneActive(coInteraction::InteractionGroup group);

    static coInteractionManager *the();
	void resetLocks(int id);
	void doRemoteLock(int groupId);
	void doRemoteUnLock(int groupId);
	bool isNaviagationBlockedByme();
private:
    // list of registered interactions
    std::list<coInteraction *> interactionStack[coInteraction::NumInteractorTypes];
    // list of active but unregistered interactions
    std::list<coInteraction *> activeUnregisteredInteractions[coInteraction::NumInteractorTypes];
	//store the client id that locked a interaction group. -1 if not locked
	std::map<int, std::unique_ptr<vrb::SharedState<int>>> remoteLocks;
	//setup SharedState for this group
	void initializeRemoteLock(int group);
	bool naviagationBlockedByme = false;
	
protected:
    static coInteractionManager *im;
};
}
#endif
