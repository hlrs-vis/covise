/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_EVENT_QUEUE_H
#define CO_EVENT_QUEUE_H

#define MAX_EVENTS 1000
#define MAX_EVENTSOURCES 50

#define SENSITIVE_EVENTS 1
#define KEY_EVENTS 2
#define AR_EVENTS 3
#define MULTI_TOUCH_EVENTS 4
#include "config.h"
#include <list>
#include <utility>
// maximum number of events queued

namespace vrml
{

class VrmlScene;
class VrmlNode;
class VrmlMessage;

struct VRMLEXPORT coEventType
{
    int type;
    void (*eventHandler)(int type, int len, const void *buf);
};

class VRMLEXPORT coEventData
{
public:
    VrmlNode *node;
    double timeStamp; // relative time is stored here
    bool isOver, isActive;
    float point[4]; // point or orientation
    void addToMsg(VrmlMessage *msg);
    char *readFromBuf(char *buf);
};

class VRMLEXPORT coEventSourceData
{
public:
    coEventSourceData();
    ~coEventSourceData();
    void setName(const char *name);

    VrmlNode *node;
    VrmlNode *remoteNode;
    int bufferSize;
    std::pair<int, int> namespaceNum;
    char *nodeName;
    void addToMsg(VrmlMessage *msg);
    char *readFromBuf(char *buf);
    static char *lastNodeName;
    static VrmlNode *lastNode;
};

class VRMLEXPORT coEventQueue
{
    friend class coEventSourceData;

public:
    coEventQueue(VrmlScene *vrmlscene);
    ~coEventQueue(); // does not send eventually queued events
    void update(); // called each frame to eventually pack and send data
    void sendKeyEvent(int type, const char *key); // send a key Event
    void addEvent(VrmlNode *node, double timeStamp,
                  bool isOver, bool isActive,
                  double *point); // add a sensitive Event to the queue
    void sendEvents(); // packs and sends datavoid

    // post remote event locally
    void postEvent(VrmlNode *node, coEventData &event);
    // parse incoming events and send them as local events
    void receiveMessage(int type, int len, const void *buf);
    // parse incoming key events
    void receiveKeyMessage(int len, const void *buf);
    void removeNodeFromCache(VrmlNode *node); // remove this Node from the Nodecache

    static const coEventType *findEventType(int type);
    static int registerEventType(const coEventType *event);
    static int unregisterEventType(const coEventType *event);

private:
    void addEventSource(VrmlNode *node);
    double lastUpdate;
    double currentTime;
    int numQueuedEvents;
    int numQueuedEventSources;
    int numEvents;
    int numEventSources;
    int eventSourceSize;
    coEventData *events[MAX_EVENTS];
    coEventSourceData *eventSources[MAX_EVENTS];
    VrmlScene *vrmlScene;

    static char *queuedURL;
    static std::list<const coEventType *> eventHandlerList;
};
}
#endif
