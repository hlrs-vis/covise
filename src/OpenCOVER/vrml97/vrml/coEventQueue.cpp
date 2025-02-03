/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <list>
using std::list;

#include <iostream>
using std::cerr;
using std::endl;

#include "coEventQueue.h"
#include "System.h"
#include "Viewer.h"
#include "VrmlNode.h"
#include "VrmlScene.h"
#include "VrmlNamespace.h"
#include "VrmlNodeAnchor.h"
#include "VrmlNodeProximitySensor.h"
#include "VrmlNodeCOVER.h"
#include "Byteswap.h"

using namespace vrml;

char *coEventQueue::queuedURL = NULL;

list<const coEventType *> coEventQueue::eventHandlerList;

coEventSourceData::coEventSourceData()
{
    nodeName = NULL;
    namespaceNum = NamespaceNum(0, 0);
}

coEventSourceData::~coEventSourceData()
{
    delete[] nodeName;
}

void coEventSourceData::setName(const char *name)
{
    delete[] nodeName;
    int len = (int)strlen(name) + 1;
    nodeName = new char[len];
    strcpy(nodeName, name);
    bufferSize = 16 + 4 + len + 8 - (len % 8);
}

void coEventSourceData::addToMsg(VrmlMessage *msg)
{
    msg->append((char *)&node, sizeof(uint64_t));
    msg->append(namespaceNum.first);
	msg->append(namespaceNum.second);
    msg->append(nodeName, strlen(nodeName) + 1);
    char dummy[8 + 8];
    msg->append(dummy, 8 + 8 - ((strlen(nodeName) + 1) % 8));
}

char *coEventSourceData::lastNodeName = NULL;
VrmlNode *coEventSourceData::lastNode = NULL;

char *coEventSourceData::readFromBuf(char *buf)
{
    node = NULL;
    remoteNode = (VrmlNode *)(*((uint64_t *)buf));
    buf += sizeof(uint64_t);
	int p = *((int *)buf);
	buf += sizeof(int);
	int n = *((int*)buf);
	buf += sizeof(int);

#ifdef BYTESWAP
    byteSwap(p);
	byteSwap(n);
#endif
	namespaceNum = NamespaceNum(p, n);

    setName(buf);

    // find node * for remoteNode
    if ((lastNodeName) && (strcmp(lastNodeName, nodeName) == 0))
    {
        node = lastNode;
    }
    else
    {

        if (strncmp(nodeName, "Anchor ", 7) == 0)
        {
            // do this later, after all Events are processed!!!!vrmlScene->load(nodeName+7);
            delete[] coEventQueue::queuedURL;
            coEventQueue::queuedURL = new char[strlen(nodeName + 7) + 1];
            strcpy(coEventQueue::queuedURL, nodeName + 7);
        }
        else
        {
            node = VrmlNamespace::findNode(nodeName, namespaceNum);
            if (node == NULL)
            {
                cerr << "Node not Found: " << nodeName << " Namespace: " << namespaceNum.first << " | " << namespaceNum.second << endl;
            }
        }

        lastNode = node;
        delete[] lastNodeName;
        lastNodeName = new char[strlen(nodeName) + 1];
        strcpy(lastNodeName, nodeName);
    }

    return (buf + bufferSize - (3 * sizeof(int)));
}

void coEventData::addToMsg(VrmlMessage *msg)
{
    msg->append((char *)&node, sizeof(uint64_t));
    msg->append(timeStamp);
    for (int i = 0; i < 4; i++)
        msg->append(point[i]);
    char flags = 0;
    if (isOver)
        flags |= 1;
    if (isActive)
        flags |= 2;
    msg->append(&flags, 1);
}

char *coEventData::readFromBuf(char *buf)
{

    memcpy(&node, buf, sizeof(uint64_t));
    buf += sizeof(uint64_t);
    memcpy(&timeStamp, buf, sizeof(double));
    buf += sizeof(double);
    memcpy(&point[0], buf, sizeof(float));
    buf += sizeof(float);
    memcpy(&point[1], buf, sizeof(float));
    buf += sizeof(float);
    memcpy(&point[2], buf, sizeof(float));
    buf += sizeof(float);
    memcpy(&point[3], buf, sizeof(float));
    buf += sizeof(float);
    isOver = *buf & 1;
    isActive = (*buf & 2) != 0;
    buf++;
#ifdef BYTESWAP
    byteSwap(timeStamp);
    byteSwap(point[0]);
    byteSwap(point[1]);
    byteSwap(point[2]);
    byteSwap(point[3]);
#endif
    //cerr << "readFromBuf" << point[0] << "|"  << point[1] << "|" << point[2] << endl;
    return (buf);
}

coEventQueue::coEventQueue(VrmlScene *vrmlscene)
    : vrmlScene(vrmlscene)
{
    numQueuedEvents = 0;
    numEvents = 0;
    numEventSources = 0;
    numQueuedEventSources = 0;
    lastUpdate = 0.0;
    eventSourceSize = 0;
}

coEventQueue::~coEventQueue()
{
    int i;
    for (i = 0; i < numEvents; i++)
    {
        delete events[i];
    }
    for (i = 0; i < numEventSources; i++)
    {
        delete eventSources[i];
    }
}

void coEventQueue::addEvent(VrmlNode *node, double timeStamp,
                            bool isOver, bool isActive,
                            double *point)
{
    if (!System::the->hasRemoteConnection())
        return; // do nothing, if no remote connection

    // check, if we have queued to may events
    if ((numQueuedEventSources == MAX_EVENTSOURCES) || (numQueuedEvents == MAX_EVENTS))
    {
        sendEvents();
    }
    // check, if this event is the same as the last event,
    // if this is the case, forget it

    if (numQueuedEvents > 0)
    {
        coEventData *le = events[numQueuedEvents - 1];
        if ((le->node == node) && (le->isOver == isOver) && (le->isActive == isActive))
        {
            le->point[0] = (float)point[0]; // just update last event
            le->point[1] = (float)point[1];
            le->point[2] = (float)point[2];
            le->timeStamp = timeStamp - lastUpdate; // relative time is stored in this structure
            return;
        }
    }

    // this is a new Event, so add it

    if (numQueuedEvents == numEvents)
    {
        events[numEvents] = new coEventData();
        numEvents++;
    }
    events[numQueuedEvents]->node = node;
    // relative time is stored in this structure
    events[numQueuedEvents]->timeStamp = timeStamp - lastUpdate;
    events[numQueuedEvents]->isOver = isOver;
    events[numQueuedEvents]->isActive = isActive;
    events[numQueuedEvents]->point[0] = (float)point[0];
    events[numQueuedEvents]->point[1] = (float)point[1];
    events[numQueuedEvents]->point[2] = (float)point[2];
    addEventSource(node);
    numQueuedEvents++;
}

void coEventQueue::sendEvents()
{
    lastUpdate = currentTime;
    if (numQueuedEvents)
    {
        //cerr << "sending " << numQueuedEvents << " events, " << numQueuedEventSources << " eventSources"<< endl;
        int buflen = 2 * sizeof(int);
        buflen += 2 * sizeof(int);
        buflen += eventSourceSize;
        buflen += numQueuedEvents * sizeof(coEventData);

        VrmlMessage *msg = System::the->newMessage(buflen);

        int idummy;
        msg->append(idummy = SENSITIVE_EVENTS);
        msg->append(numQueuedEventSources);
        msg->append(numQueuedEvents);

        int i;
        for (i = 0; i < numQueuedEventSources; i++)
            eventSources[i]->addToMsg(msg);
        for (i = 0; i < numQueuedEvents; i++)
            events[i]->addToMsg(msg);

        System::the->sendAndDeleteMessage(msg);
    }
    numQueuedEvents = 0;
    numQueuedEventSources = 0;
    eventSourceSize = 0;
}

// send a key Event
void coEventQueue::sendKeyEvent(int type, const char *key)
{
    int buflen = 2 * sizeof(int);
    buflen += 1 * sizeof(int);
    buflen += (int)strlen(key) + 1;

    VrmlMessage *msg = System::the->newMessage(buflen);
    int idummy;
    msg->append(idummy = KEY_EVENTS);
    msg->append(type);
    msg->append(key, strlen(key) + 1);
    System::the->sendAndDeleteMessage(msg);
}

void coEventQueue::removeNodeFromCache(VrmlNode *) // remove this Node from the Nodecache
{
    coEventSourceData::lastNodeName = NULL;
    coEventSourceData::lastNode = NULL;
}

void coEventQueue::receiveKeyMessage(int /*len*/, const void *buf)
{
    currentTime = System::the->time();
    const char *currentPos = (const char *)buf;
    int type = *((int *)currentPos);
#ifdef BYTESWAP
    byteSwap(type);
#endif
    currentPos += sizeof(int);
    if (theCOVER)
        theCOVER->remoteKeyEvent((VrmlNodeCOVER::KeyEventType)type, currentPos);
}

void coEventQueue::receiveMessage(int type, int len, const void *buf)
{
    if (type == SENSITIVE_EVENTS)
    {
        currentTime = System::the->time();
        char *currentPos = (char *)buf;
        numQueuedEventSources = *((int *)currentPos);
#ifdef BYTESWAP
        byteSwap(numQueuedEventSources);
#endif
        currentPos += sizeof(int);
        numQueuedEvents = *((int *)currentPos);
#ifdef BYTESWAP
        byteSwap(numQueuedEvents);
#endif
        currentPos += sizeof(int);
        int i, n;
        for (i = 0; i < numQueuedEventSources; i++)
        {
            if (i == numEventSources)
            {
                eventSources[numEventSources] = new coEventSourceData();
                numEventSources++;
            }
            currentPos = eventSources[i]->readFromBuf(currentPos);
        }
        coEventData event;
        for (i = 0; i < numQueuedEvents; i++)
        {
            currentPos = event.readFromBuf(currentPos);
            for (n = 0; n < numQueuedEventSources; n++)
            {
                if (event.node == eventSources[n]->remoteNode)
                {
                    postEvent(eventSources[n]->node, event);
                }
            }
        }

        // do this now, after all Events are processed!!!!
        if (coEventQueue::queuedURL)
        {
            vrmlScene->load(coEventQueue::queuedURL);
            delete[] coEventQueue::queuedURL;
            coEventQueue::queuedURL = NULL;
        }
    }
    else if (type == KEY_EVENTS)
    {
        vrmlScene->getIncomingSensorEventQueue()->receiveKeyMessage(len, buf);
    }
    else
    {
        const coEventType *eventType = findEventType(type);
        if (eventType && eventType->eventHandler)
            (*eventType->eventHandler)(type, len, buf);
    }
}

// post remote event locally
void coEventQueue::postEvent(VrmlNode *node, coEventData &event)
{
    //cerr << "post " << event.point[0] << "|"  << event.point[1] << "|" << event.point[2] << endl;
    if (vrmlScene && node)
    {
        if (node->as<VrmlNodeProximitySensor>())
        {
            VrmlNodeProximitySensor *proxi = node->as<VrmlNodeProximitySensor>();
            proxi->remoteEvent(currentTime + event.timeStamp, event.isOver, event.isActive, event.point);
        }
        else
        {
            double dpoint[3];
            dpoint[0] = event.point[0];
            dpoint[1] = event.point[1];
            dpoint[2] = event.point[2];
            vrmlScene->remoteSensitiveEvent(node, currentTime + event.timeStamp, event.isOver, event.isActive, dpoint, NULL);
        }
    }
}

void coEventQueue::addEventSource(VrmlNode *node)
{
    //check, if we already have this node in our list
    int i;
    for (i = 0; i < numQueuedEventSources; i++)
    {
        if (eventSources[i]->node == node)
        {
            return;
        }
    }

    // not found, so add
    if (numQueuedEventSources == numEventSources)
    {
        eventSources[numEventSources] = new coEventSourceData();
        numEventSources++;
    }
    //const char *name = node->name();
    if (node->name()[0] != '\0')
    {
        eventSources[numQueuedEventSources]->setName(node->name());
    }
    else
    {
        VrmlNodeAnchor *a = node->as<VrmlNodeAnchor>();
        if (a && a->url())
        {
            int len = (int)strlen(a->url());
            len += 10;
            char *name = new char[len];
            strcpy(name, "Anchor ");
            strcat(name, a->url());
            eventSources[numQueuedEventSources]->setName(name);
        }
        else
        {
            cerr << "ERROR: Noname Event Source" << endl;
            return;
        }
    }
    eventSources[numQueuedEventSources]->node = node;
    VrmlNamespace *tmp = node->getNamespace();
    if (NULL != tmp)
    {
        eventSources[numQueuedEventSources]->namespaceNum = tmp->getNumber();
    }
    eventSourceSize += eventSources[numQueuedEventSources]->bufferSize;
    numQueuedEventSources++;
}

void coEventQueue::update()
{
    currentTime = System::the->time();
    if (currentTime > lastUpdate + System::the->getSyncInterval())
    {
        sendEvents();
    }
}

const coEventType *coEventQueue::findEventType(int type)
{
    for (list<const coEventType *>::const_iterator it = eventHandlerList.begin();
         it != eventHandlerList.end();
         it++)
    {
        if ((*it)->type == type)
            return *it;
    }

    return NULL;
}

int coEventQueue::registerEventType(const coEventType *event)
{
    if (findEventType(event->type))
        return -1;

    eventHandlerList.push_back(event);
    return 0;
}

int coEventQueue::unregisterEventType(const coEventType *event)
{
    const coEventType *p = findEventType(event->type);
    if (p == NULL)
        return -1;

    if (p->eventHandler == event->eventHandler)
    {
        eventHandlerList.remove(p);

        return 0;
    }
    return -1;
}
