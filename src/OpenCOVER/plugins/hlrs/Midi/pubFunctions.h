//
// Created by leoei on 15.02.2022.
//

#ifndef INC_02_PUBSUB_SERVER_PUBFUNCTIONS_H
#define INC_02_PUBSUB_SERVER_PUBFUNCTIONS_H

#include "open62541.h"
#include "pubFunctions.h"
#include <signal.h>

extern UA_NodeId connectionIdent, publishedDataSetIdent, writerGroupIdent;

void
addPubSubConnection(UA_Server *server, UA_String *transportProfile,
                    UA_NetworkAddressUrlDataType *networkAddressUrl);

void
addPublishedDataSet(UA_Server *server);

void
addDataSetField(UA_Server *server, UA_NodeId pubVariantId, UA_NodeId IdObject);

void
addWriterGroup(UA_Server *server);

void
addDataSetWriter(UA_Server *server);


#endif //INC_02_PUBSUB_SERVER_PUBFUNCTIONS_H
