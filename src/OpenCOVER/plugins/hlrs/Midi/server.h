//
// Created by Lukas Koberg on 28.02.2022.
//

#include "open62541.h"
#ifndef INC_04_BANDSERVER_SERVER_H
#define INC_04_BANDSERVER_SERVER_H


extern UA_NodeId b_pubDataObj_Id;
extern UA_NodeId b_pubDataID_Id;
extern UA_Byte pubData_ID;


struct PrevValueIDAndValue{
    int nodeID;
    UA_Byte pubDataValues;
};

extern int numberPubDataBytes;

extern UA_Byte pubData_arr[8];
extern UA_VariableAttributes pubDataAttr[8];
extern UA_NodeId b_pubData_Id[8];

extern UA_Byte pubDataValues[8];

UA_StatusCode addPublishObj(UA_Server *server);
UA_StatusCode addSongJobObject(UA_Server *server, int numberSongObject);

#endif //INC_04_BANDSERVER_SERVER_H