//
// Created by Lukas Koberg on 28.02.2022.
//

#include "open62541.h"
#include "server.h"
#include "midi.h"


// Initialise Variables:

int numberPubDataBytes = 8;

UA_Byte pubData_arr[8];
UA_VariableAttributes pubDataAttr[8];
UA_NodeId b_pubData_Id[8];
UA_Byte pubDataValues[8];

UA_NodeId b_pubDataObj_Id;
UA_NodeId b_pubDataID_Id;
UA_Byte pubData_ID;

/*
 * Play Song
 *
 * @Lukas
 */
UA_StatusCode playSongCallback(UA_Server *server,
                               const UA_NodeId *sessionId, void *sessionHandle,
                               const UA_NodeId *methodId, int *methodContext,
                               const UA_NodeId *objectId, void *objectContext,
                               size_t inputSize, const UA_Variant *input,
                               size_t outputSize, UA_Variant *output) {
    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "Hier wird nun Song%i abgespielt", methodContext);
    //Song 1 Pfad: "../Songfiles/Song1.mid", hier muss midiparsing gestartet werden
    //setPubData(server, 20,20,20,20,20);
    if (!midiParserReady) {
        midiParserReady = run_midi(server, methodContext);
        play = true;
    }else{
        play = true;
    }


    return UA_STATUSCODE_GOOD;
}

UA_StatusCode pauseSongCallback(UA_Server *server,
                               const UA_NodeId *sessionId, void *sessionHandle,
                               const UA_NodeId *methodId, int *methodContext,
                               const UA_NodeId *objectId, void *objectContext,
                               size_t inputSize, const UA_Variant *input,
                               size_t outputSize, UA_Variant *output) {
    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "Hier wird nun Song%i pausiert",  methodContext);
    play = false;
    return UA_STATUSCODE_GOOD;
}

UA_StatusCode stopSongCallback(UA_Server *server,
                                const UA_NodeId *sessionId, void *sessionHandle,
                                const UA_NodeId *methodId, int *methodContext,
                                const UA_NodeId *objectId, void *objectContext,
                                size_t inputSize, const UA_Variant *input,
                                size_t outputSize, UA_Variant *output) {
    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_SERVER, "Hier wird nun Song%i gestopt",  methodContext);
    midiParserReady = false;
    return UA_STATUSCODE_GOOD;
}



/*
 * Add Song Job
 *
 * Used to add a songjob, which generates the job, holds the file.
 * @Lukas
 */

UA_StatusCode addSongJobObject(UA_Server *server, int numberSongObject) {
    char end[10] = ".mid";

    UA_NodeId b_SongJob_Id[numberSongObject];
    UA_ObjectAttributes sjAttr[numberSongObject];

    UA_NodeId b_SongJob_File[numberSongObject];
    UA_ObjectAttributes sfAttr[numberSongObject];

    UA_MethodAttributes playAttr[numberSongObject];
    UA_MethodAttributes pauseAttr[numberSongObject];
    UA_MethodAttributes stopAttr[numberSongObject];

    for (int i = 0; i<numberSongObject; i++) {
        char SongJobName[20]  = "";
        sprintf(SongJobName, "SongJob_%d", i);
        char filePathString[30]  = "";
        sprintf(filePathString, "../Songfiles/Song_%d", i);
        strcat(filePathString,end);


       // printf(fileObjectSongJobID);

        sjAttr[i] = UA_ObjectAttributes_default;
        UA_Server_addObjectNode(server, UA_NODEID_NUMERIC(2, 10*i+2000),
                                UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
                                UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
                                UA_QUALIFIEDNAME(2, SongJobName), UA_NODEID_NUMERIC(0, UA_NS0ID_BASEOBJECTTYPE),
                                sjAttr[i], NULL, &b_SongJob_Id[i]);


        sfAttr[i] = UA_ObjectAttributes_default;
        sfAttr[i].displayName = UA_LOCALIZEDTEXT("de-DE", "SongJobFile"); // for persistence files comment line 31173-31174 in open62541.c in function UA_Server_addFile
        UA_String filePath = UA_STRING(filePathString); //create in current directory
        UA_Server_addFile(server, UA_NODEID_NUMERIC(2, 10*i+2001), //UA_Server_addFile
                          b_SongJob_Id[i], UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
                          UA_QUALIFIEDNAME(1, "Sample File"), sfAttr[i],
                          filePath, NULL, &b_SongJob_File[i]);

        // Add Play Method
        playAttr[i] = UA_MethodAttributes_default;
         playAttr[i].description = UA_LOCALIZEDTEXT("en-US", "Play a Song");
         playAttr[i].displayName = UA_LOCALIZEDTEXT("en-US", "Play");
         playAttr[i].executable = true;
         playAttr[i].userExecutable = true;
        UA_Server_addMethodNode(server, UA_NODEID_NUMERIC(2, 10*i+2002),
                                b_SongJob_Id[i],
                                UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                                UA_QUALIFIEDNAME(1, "Play"),
                                playAttr[i], &playSongCallback,
                                0, NULL, 0, NULL, i, NULL);

        // Add Pause Method
        pauseAttr[i] = UA_MethodAttributes_default;
         pauseAttr[i].description = UA_LOCALIZEDTEXT("en-US", "Pause a Song");
         pauseAttr[i].displayName = UA_LOCALIZEDTEXT("en-US", "Pause");
         pauseAttr[i].executable = true;
         pauseAttr[i].userExecutable = true;
        UA_Server_addMethodNode(server, UA_NODEID_NUMERIC(2, 10*i+2003),
                                b_SongJob_Id[i],
                                UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                                UA_QUALIFIEDNAME(1, "Pause"),
                                pauseAttr[i], &pauseSongCallback,
                                0, NULL, 0, NULL, i, NULL);
        // Add Stop Method
        stopAttr[i] = UA_MethodAttributes_default;
        stopAttr[i].description = UA_LOCALIZEDTEXT("en-US", "Stop a Song");
        stopAttr[i].displayName = UA_LOCALIZEDTEXT("en-US", "Stop");
        stopAttr[i].executable = true;
        stopAttr[i].userExecutable = true;
        UA_Server_addMethodNode(server, UA_NODEID_NUMERIC(2, 10*i+2004),
                                b_SongJob_Id[i],
                                UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                                UA_QUALIFIEDNAME(1, "Stop"),
                                stopAttr[i], &stopSongCallback,
                                0, NULL, 0, NULL, i, NULL);
        //Möglichkeit file einfach per string einlesen! (File öffnen)
    }

    return UA_STATUSCODE_GOOD;
}



void beforeReadDataValue(UA_Server *server,
const UA_NodeId *sessionId, void *sessionContext,
const UA_NodeId *nodeId, void *nodeContext,
        UA_Boolean sourceTimeStamp, const UA_NumericRange *range,
        UA_DataValue *dataValue){

    struct PrevValueIDAndValue* instance =(struct PrevValueIDAndValue* )(nodeContext);
    UA_Byte* currentVal;
    if(instance->nodeID == 0){
        //UA_Byte data = getPubDataIDRealTime((instance->nodeID));
        //currentVal = &data;
        currentVal = getPubDataIDRealTime((instance->nodeID));
    }else {
        //UA_Byte data = getPubDataRealTime((instance->nodeID));
        //currentVal = &data;
        currentVal = getPubDataRealTime((instance->nodeID));
        //(*currentVal) += 1;
    }



    UA_Variant_setScalarCopy(&dataValue->value, currentVal , &UA_TYPES[UA_TYPES_BYTE]);

}



// num 0: Instrument ID; 1-n Daten
void addPubDataValueCallback(UA_Server *server, UA_NodeId *itemId, int num){

    UA_DataSource varDataSource;
    varDataSource.read = beforeReadDataValue;
    varDataSource.write = NULL;

    UA_StatusCode retVal;
    retVal = UA_Server_setVariableNode_dataSource(server, *itemId, varDataSource);
    //UA_Byte *nodeContext =  UA_malloc(sizeof(UA_Byte));
    struct PrevValueIDAndValue* nodeContext = UA_malloc(sizeof(struct PrevValueIDAndValue));

    if (num == 0){
        pubData_ID = 0;
        struct PrevValueIDAndValue init_nodecontext = {.pubDataValues = pubData_ID,
                .nodeID = num};
        *nodeContext = init_nodecontext;
    }else {
        pubDataValues[num - 1] = 0;
        struct PrevValueIDAndValue init_nodecontext = {.pubDataValues = pubDataValues[num - 1],
                .nodeID = num};
        *nodeContext = init_nodecontext;
    }

    //set nodeContext for variant
    UA_Server_setNodeContext(server, *itemId, nodeContext);

    if (retVal != UA_STATUSCODE_GOOD) {
        printf("Error_SensorUnitValueCallback:%s",UA_StatusCode_name(retVal));
    }
}



UA_StatusCode addPublishObj(UA_Server *server) {
    midiParserReady = false;


    UA_ObjectAttributes sjAttr = UA_ObjectAttributes_default;
    UA_Server_addObjectNode(server, UA_NODEID_STRING(2, "b_PubData"),
                            UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
                            UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
                            UA_QUALIFIEDNAME(2, "PubData"), UA_NODEID_NUMERIC(0, UA_NS0ID_BASEOBJECTTYPE),
                            sjAttr, NULL, &b_pubDataObj_Id);


    UA_VariableAttributes pubidAttr = UA_VariableAttributes_default;
    pubData_ID = 10;
    UA_Variant_setScalar(&pubidAttr.value, &pubData_ID, &UA_TYPES[UA_TYPES_BYTE]);
    UA_Server_addVariableNode(server, UA_NODEID_STRING(2, "b_PubData_ID"), b_pubDataObj_Id,
                              UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                              UA_QUALIFIEDNAME(2, "ID"),
                              UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE),
                              pubidAttr, NULL, &b_pubDataID_Id);
    addPubDataValueCallback(server, &b_pubDataID_Id,0);
    int i;
    int j;
    int nodeID;

    for(i=0; i < numberPubDataBytes; i++){
        char nodeName[10]  = "";
        sprintf(nodeName, "Data_%d", i+1);
        nodeID = 1000 + i+ 1;

        pubDataAttr[i] = UA_VariableAttributes_default;
        pubData_arr[i]  = i;
        UA_Variant_setScalar(&pubDataAttr[i].value, &pubData_arr[i], &UA_TYPES[UA_TYPES_BYTE]);

        UA_Server_addVariableNode(server, UA_NODEID_NUMERIC(2, nodeID), b_pubDataObj_Id,
                                  UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                                  UA_QUALIFIEDNAME(2, nodeName),
                                  UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE),
                                  pubDataAttr[i], NULL, &b_pubData_Id[i]);
        addPubDataValueCallback(server, &b_pubData_Id[i],i+1);
    }
}