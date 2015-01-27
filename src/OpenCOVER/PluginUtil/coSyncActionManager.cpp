/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coVRCommunication.h"
#include "coVRPluginSupport.h"
#include "coSyncActionManager.h"

using namespace covise;
using namespace opencover;
coSyncActionManager *coSyncActionManager::instance = NULL;

coSyncActionManager *coSyncActionManager::instance()
{
    if (instance == NULL)
        instance = new coSyncActionManager();
    return instance;
}

coSyncActionManager::coSyncActionManager()
{
    remoteActions.noDelete = true;
    localActions.noDelete = true;
}

coSyncActionManager::~coSyncActionManager()
{
}

coSynchronizedAction *coSyncActionManager::findRemoteAction(int host, int type, int sequenceNumber)
{
    coSynchronizedAction *c;
    remoteActions.reset();
    while ((c = remoteActions.current()))
    {
        if (c->host == host && c->type == type && c->mySequenceNumber == sequenceNumber)
        {
            return c;
        }
        remoteActions.next();
    }

    cerr << "coSyncActionManager::findRemoteAction: hier muss man nochmal nachschauen" << endl;
    return 0;
}

coSynchronizedAction *coSyncActionManager::findLocalAction(int host, int type, int sequenceNumber)
{
    coSynchronizedAction *c;
    localActions.reset();
    while ((c = localActions.current()))
    {
        if (c->host == host && c->type == type && c->mySequenceNumber == sequenceNumber)
        {
            return c;
        }
        localActions.next();
    }

    cerr << "coSyncActionManager::findLocalAction: hier muss man nochmal nachschauen" << endl;
    return 0;
}

void coSyncActionManager::handleRemoteActions(int /*len*/, const char *message)
{
    int host, type, sequenceNumber, blocking;
    if (strncmp(message, "Init", 4) == 0)
    {
        if (sscanf(message, "Init %d %d %d %d", &host, &type, &sequenceNumber, &blocking) != 4)
        {
            cerr << "coSyncActionManager::handleRemoteActions: sscanf1 failed" << endl;
        }

        if (host != coVRCommunication::instance()->getID())
        {
            char buf[1000];
            snprintf(buf, 1000, "Confirm %d %d %d %d %d", coVRCommunication::instance()->getID(), host, type, sequenceNumber, blocking);

            Message *message = new Message();

            message->data = buf;
            message->type = COVISE_MESSAGE_SYNCHRONIZED_ACTION; // should be a real type
            message->length = strlen(message->data) + 1;

            cover->sendVrbMessage(message);
            delete[] message -> data;
            delete message;
            coSynchronizedAction *newAction = new coSynchronizedAction(host, type, sequenceNumber, blocking);
            remoteActions.append(newAction);

            while (blocking && remoteActions.find(newAction))
            {
                Message *m = coVRCommunication::instance()->waitForMessage(COVISE_MESSAGE_SYNCHRONIZED_ACTION);
                if (m)
                {
                    handleRemoteActions(m->length, m->data);
                }
            }

            if (remoteActions.find(newAction))
                remoteActions.remove();
        }
    }
    if (strncmp(message, "Confirm", 7) == 0)
    {
        int senderHost;
        if (sscanf(message, "Confirm %d %d %d %d %d", &senderHost, &host, &type, &sequenceNumber, &blocking) != 5)
        {
            cerr << "coSyncActionManager::handleRemoteActions: sscanf2 failed" << endl;
        }

        coSynchronizedAction *action = findLocalAction(host, type, sequenceNumber);
        action->numberOfConfirmations++;
        if (action && action->numberOfConfirmations >= coVRCommunication::instance()->getNumberOfPartners())
        {
            char buf[1000];
            snprintf(buf, 1000, "Release %d %d %d %d %d", coVRCommunication::instance()->getID(), host, type, sequenceNumber, blocking);

            Message *message = new Message();

            message->data = buf;
            message->type = COVISE_MESSAGE_SYNCHRONIZED_ACTION; // should be a real type
            message->length = strlen(message->data) + 1;

            cover->sendVrbMessage(message);

            delete[] message -> data;
            delete message;
            if (localActions.find(action))
            {
                action->fireAction(type);
                localActions.remove();
            }
            else
            {
                fprintf(stderr, "Did not find local action to release, sending host: %d\n", senderHost);
            }
        }
    }
    if (strncmp(message, "Release", 7) == 0)
    {
        int senderHost;
        if (sscanf(message, "Release %d %d %d %d %d", &senderHost, &host, &type, &sequenceNumber, &blocking) != 5)
        {
            cerr << "coSyncActionManager::handleRemoteActions: sscanf3 failed" << endl;
        }

        coSynchronizedAction *action = findRemoteAction(host, type, sequenceNumber);
        if (action && action->numberOfConfirmations >= coVRCommunication::instance()->getNumberOfPartners())
        {
            if (remoteActions.find(action))
            {
                remoteActions.remove();
            }
            else
            {
                fprintf(stderr, "Did not find local action to release, sending host: %d\n", senderHost);
            }
        }
    }
}

void coSyncActionManager::initiateAction(coSynchronizedAction *newAction, int type, bool blocking)
{
    localActions.append(newAction);
    newAction->type = type;
    newAction->blocking = blocking;
    newAction->host = coVRCommunication::instance()->getID();

    char buf[1000];
    snprintf(buf, 1000, "Init %d %d %d %d", coVRCommunication::instance()->getID(), type, newAction->mySequenceNumber, (int)blocking);

    Message *message = new Message();

    message->data = buf;
    message->type = COVISE_MESSAGE_SYNCHRONIZED_ACTION; // should be a real type
    message->length = strlen(message->data) + 1;

    cover->sendVrbMessage(message);

    delete[] message -> data;
    delete message;

    while (blocking && localActions.find(newAction))
    {
        Message *m = coVRCommunication::instance()->waitForMessage(COVISE_MESSAGE_SYNCHRONIZED_ACTION);
        {
            handleRemoteActions(m->length, m->data);
        }
    }

    if (localActions.find(newAction))
        localActions.remove();
}
