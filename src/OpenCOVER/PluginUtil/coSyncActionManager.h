/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SYNCACTIONMANAGER_H
#define CO_SYNCACTIONMANAGER_H

#include "coSynchronizedAction.h"

namespace opencover
{
/** Allow all clients to performe a synchronized Action
 */
class COVEREXPORT coSyncActionManager
{
protected:
    covise::DLinkList<coSynchronizedAction *> remoteActions;
    covise::DLinkList<coSynchronizedAction *> localActions;
    coSyncActionManager();
    static coSyncActionManager *instance;

public:
    static coSyncActionManager *instance();
    virtual ~coSyncActionManager();
    void handleRemoteActions(int len, const char *message);
    coSynchronizedAction *findLocalAction(int host, int type, int sequenceNumber);
    coSynchronizedAction *findRemoteAction(int host, int type, int sequenceNumber);
    void initiateAction(coSynchronizedAction *newAction, int type, bool blocking = false);
};
}
#endif
