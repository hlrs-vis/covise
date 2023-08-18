#include "opcua.h"



#include <open62541/client_config_default.h>
#include <open62541/client_highlevel.h>
#include <open62541/plugin/log_stdout.h>
#include <open62541/plugin/pki_default.h>
#include <open62541/client_subscriptions.h>
#include <open62541/client.h>
#include <open62541/common.h>
#include <iostream>

#include <cover/ui/VectorEditField.h>


std::map<UA_UInt32, std::function<void(double)>> doubleCallbacks;

static void handler_TheAnswerChanged(UA_Client *client, UA_UInt32 subId, void *subContext,
                         UA_UInt32 monId, void *monContext, UA_DataValue *value) {
    

    if(UA_Variant_hasScalarType(&value->value, &UA_TYPES[UA_TYPES_DOUBLE])) {
        std::cerr << "the value is: "  << *(UA_Double*)value->value.data << std::endl;
        doubleCallbacks[*(UA_UInt32 *)monContext](*(UA_Double*)value->value.data);

    }
    else{
        std::cerr << "wrong type " << value->value.type->typeKind << std::endl;
    }
}

static UA_INLINE UA_ByteString loadFile(const char *const path) {
    UA_ByteString fileContents = UA_STRING_NULL;

    /* Open the file */
    FILE *fp = fopen(path, "rb");
    if(!fp) {
        errno = 0; /* We read errno also from the tcp layer... */
        return fileContents;
    }

    /* Get the file length, allocate the data and read */
    fseek(fp, 0, SEEK_END);
    fileContents.length = (size_t)ftell(fp);
    fileContents.data = (UA_Byte *)UA_malloc(fileContents.length * sizeof(UA_Byte));
    if(fileContents.data) {
        fseek(fp, 0, SEEK_SET);
        size_t read = fread(fileContents.data, sizeof(UA_Byte), fileContents.length, fp);
        if(read != fileContents.length)
            UA_ByteString_clear(&fileContents);
    } else {
        fileContents.length = 0;
    }
    fclose(fp);

    return fileContents;
}

OpcUaClient::OpcUaClient(const std::string &name, opencover::ui::Menu *menu, opencover::config::File &config)
: m_menu(new opencover::ui::Menu(menu, name))
, m_connect(new opencover::ui::Button(m_menu, "connect"))
, m_username(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "username", "", config, name))
, m_password(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "password", "", config, name))
, m_serverIp(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "serverIp", "", config, name))
, m_certificate(std::make_unique<opencover::ui::FileBrowserConfigValue>(m_menu, "certificate", "", config, name))
, m_key(std::make_unique<opencover::ui::FileBrowserConfigValue>(m_menu, "key", "", config, name))
, m_name(name)
{
m_connect->setCallback([this](bool state){
    if(state)
    {
        if(connect())
        {
            m_connect->setText("disconnect");
        }
        else{
            m_connect->setText("connect");
            m_connect->setState(false);
        }
    } else {
        disconnect();
        m_connect->setText("connect");
    }
});


}

void OpcUaClient::onConnect(const std::function<void(void)> &cb)
{
    m_onConnect = cb;
}

void OpcUaClient::onDisconnect(const std::function<void(void)> &cb)
{
    m_onDisconnect = cb;
}

OpcUaClient::~OpcUaClient()
{
    if(m_connect->state())
    {
        for(auto cb : doubleCallbacks)
        {
        /* Delete the subscription */
        if(UA_Client_Subscriptions_deleteSingle(client, cb.first) == UA_STATUSCODE_GOOD)
            printf("Subscription removed\n");
        }
    }
    disconnect();
}

void OpcUaClient::listVariables(UA_Client* client)
{
    /* Browse some objects */
    printf("Browsing nodes in objects folder:\n");
    UA_BrowseRequest bReq;
    UA_BrowseRequest_init(&bReq);
    bReq.requestedMaxReferencesPerNode = 0;
    bReq.nodesToBrowse = UA_BrowseDescription_new();
    bReq.nodesToBrowseSize = 1;
    bReq.nodesToBrowse[0].nodeId = UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER); /* browse objects folder */
    bReq.nodesToBrowse[0].resultMask = UA_BROWSERESULTMASK_ALL; /* return everything */
    UA_BrowseResponse bResp = UA_Client_Service_browse(client, bReq);
    printf("%-9s %-16s %-16s %-16s\n", "NAMESPACE", "NODEID", "BROWSE NAME", "DISPLAY NAME");
    
    
    for(size_t i = 0; i < bResp.resultsSize; ++i) {
        for(size_t j = 0; j < bResp.results[i].referencesSize; ++j) {
            UA_ReferenceDescription *ref = &(bResp.results[i].references[j]);
            if(ref->nodeId.nodeId.identifierType == UA_NODEIDTYPE_NUMERIC) {
                printf("%-9u %-16u %-16.*s %-16.*s\n", ref->nodeId.nodeId.namespaceIndex,
                       ref->nodeId.nodeId.identifier.numeric, (int)ref->browseName.name.length,
                       ref->browseName.name.data, (int)ref->displayName.text.length,
                       ref->displayName.text.data);
                std::vector<char> v(ref->browseName.name.length + 1);
                std::copy(ref->browseName.name.data, ref->browseName.name.data + ref->browseName.name.length, v.begin());
                v[ref->browseName.name.length] = '\0';
                m_numericalNodes[v.data()] = OpcUaField{v.data(), ref->nodeId.nodeId.namespaceIndex, ref->nodeId.nodeId.identifier.numeric};
            } else if(ref->nodeId.nodeId.identifierType == UA_NODEIDTYPE_STRING) {
                printf("%-9u %-16.*s %-16.*s %-16.*s\n", ref->nodeId.nodeId.namespaceIndex,
                       (int)ref->nodeId.nodeId.identifier.string.length,
                       ref->nodeId.nodeId.identifier.string.data,
                       (int)ref->browseName.name.length, ref->browseName.name.data,
                       (int)ref->displayName.text.length, ref->displayName.text.data);
            }
            /* TODO: distinguish further types */
        }
    }
    UA_BrowseRequest_clear(&bReq);
    UA_BrowseResponse_clear(&bResp);
}

bool OpcUaClient::connect()
{
/* Create the server and set its config */
    client = UA_Client_new();
    UA_ClientConfig *cc = UA_Client_getConfig(client);

    /* Set securityMode and securityPolicyUri */
    cc->securityMode = UA_MESSAGESECURITYMODE_SIGNANDENCRYPT;
    cc->securityPolicyUri = UA_STRING_NULL;

    UA_ByteString certificate = loadFile(m_certificate->getValue().c_str());
    UA_ByteString privateKey  = loadFile(m_key->getValue().c_str());

    /* If no trust list is passed, all certificates are accepted. */
    UA_ClientConfig_setDefaultEncryption(cc, certificate, privateKey,
                                            NULL, 0, NULL, 0);
    UA_CertificateVerification_AcceptAll(&cc->certificateVerification);
    UA_ByteString_clear(&certificate);
    UA_ByteString_clear(&privateKey);
    


    /* The application URI must be the same as the one in the certificate.
     * The script for creating a self-created certificate generates a certificate
     * with the Uri specified below.*/
    UA_ApplicationDescription_clear(&cc->clientDescription);
    cc->clientDescription.applicationUri = UA_STRING_ALLOC("urn:open62541.server.application");
    cc->clientDescription.applicationType = UA_APPLICATIONTYPE_CLIENT;

    /* Connect to the server */
    UA_StatusCode retval = UA_STATUSCODE_GOOD;
    UA_ClientConfig_setAuthenticationUsername(cc, m_username->getValue().c_str(), m_password->getValue().c_str());
    retval = UA_Client_connect(client, m_serverIp->getValue().c_str());
    /* Alternative */
    //retval = UA_Client_connectUsername(client, serverurl, username, password);

    if(retval != UA_STATUSCODE_GOOD) {
        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Could not connect");
        UA_Client_delete(client);
        return false;
    }

    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Connected!");

    {
        /* Read the server-time */
        UA_Variant value;
        UA_Variant_init(&value);
        UA_Client_readValueAttribute(client,
                UA_NODEID_NUMERIC(0, UA_NS0ID_SERVER_SERVERSTATUS_CURRENTTIME),
                &value);
        
        if(UA_Variant_hasScalarType(&value, &UA_TYPES[UA_TYPES_DATETIME])) {
            UA_DateTimeStruct dts = UA_DateTime_toStruct(*(UA_DateTime *)value.data);
            UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND,
                        "The server date is: %02u-%02u-%04u %02u:%02u:%02u.%03u",
                        dts.day, dts.month, dts.year, dts.hour, dts.min, dts.sec, dts.milliSec);
        }
        UA_Variant_clear(&value);
    }
    listVariables(client);
    if(m_onConnect)
        m_onConnect();
    return true;
    // registerDouble("CURRENT|A", [](double){
    //     std::cerr << "hello world" << std::endl;
    // });


//     /* Read attribute */
//     int32_t value = 0;
//     printf("\nReading the value of node (1, \"the.answer\"):\n");
//     UA_Variant *val = UA_Variant_new();
//     retval = UA_Client_readValueAttribute(client, UA_NODEID_STRING(1, (char*)"the.answer"), val);
//     if(retval == UA_STATUSCODE_GOOD && UA_Variant_isScalar(val) &&
//        val->type == &UA_TYPES[UA_TYPES_INT32]) {
//             value = *(int32_t*)val->data;
//             printf("the value is: %i\n", value);
//     }
//     UA_Variant_delete(val);

//     /* Write node attribute */
//     value++;
//     printf("\nWriting a value of node (1, \"the.answer\"):\n");
//     UA_WriteRequest wReq;
//     UA_WriteRequest_init(&wReq);
//     wReq.nodesToWrite = UA_WriteValue_new();
//     wReq.nodesToWriteSize = 1;
//     wReq.nodesToWrite[0].nodeId = UA_NODEID_STRING_ALLOC(1, "the.answer");
//     wReq.nodesToWrite[0].attributeId = UA_ATTRIBUTEID_VALUE;
//     wReq.nodesToWrite[0].value.hasValue = true;
//     wReq.nodesToWrite[0].value.value.type = &UA_TYPES[UA_TYPES_INT32];
//     wReq.nodesToWrite[0].value.value.storageType = UA_VARIANT_DATA_NODELETE; /* do not free the integer on deletion */
//     wReq.nodesToWrite[0].value.value.data = &value;
//     UA_WriteResponse wResp = UA_Client_Service_write(client, wReq);
//     if(wResp.responseHeader.serviceResult == UA_STATUSCODE_GOOD)
//             printf("the new value is: %i\n", value);
//     UA_WriteRequest_clear(&wReq);
//     UA_WriteResponse_clear(&wResp);

//     /* Write node attribute (using the highlevel API) */
//     value++;
//     UA_Variant *myVariant = UA_Variant_new();
//     UA_Variant_setScalarCopy(myVariant, &value, &UA_TYPES[UA_TYPES_INT32]);
//     UA_Client_writeValueAttribute(client, UA_NODEID_STRING(1, (char*)"the.answer"), myVariant);
//     UA_Variant_delete(myVariant);



// #ifdef UA_ENABLE_METHODCALLS
//     /* Call a remote method */
//     UA_Variant input;
//     UA_String argString = UA_STRING((char*)"Hello Server");
//     UA_Variant_init(&input);
//     UA_Variant_setScalarCopy(&input, &argString, &UA_TYPES[UA_TYPES_STRING]);
//     size_t outputSize;
//     UA_Variant *output;
//     retval = UA_Client_call(client, UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
//                             UA_NODEID_NUMERIC(1, 62541), 1, &input, &outputSize, &output);
//     if(retval == UA_STATUSCODE_GOOD) {
//         printf("Method call was successful, and %lu returned values available.\n",
//                (unsigned long)outputSize);
//         UA_Array_delete(output, outputSize, &UA_TYPES[UA_TYPES_VARIANT]);
//     } else {
//         printf("Method call was unsuccessful, and %x returned values available.\n", retval);
//     }
//     UA_Variant_clear(&input);
// #endif

}

bool OpcUaClient::disconnect()
{
    /* Clean up */
    if(m_onDisconnect)
        m_onDisconnect();
    UA_Client_disconnect(client);
    UA_Client_delete(client);
    client = nullptr;
    return true;
}

bool OpcUaClient::registerDouble(const std::string &name, const std::function<void(double)> &cb)
{
    auto fieldIt = m_numericalNodes.find(name);
    if(fieldIt == m_numericalNodes.end())
        return false;
    /* Create a subscription */
    UA_CreateSubscriptionRequest request = UA_CreateSubscriptionRequest_default();
    UA_CreateSubscriptionResponse response = UA_Client_Subscriptions_create(client, request,
                                                                            NULL, NULL, NULL);

    UA_UInt32 subId = response.subscriptionId;
    auto cbIt = doubleCallbacks.emplace(std::make_pair(subId, cb)).first;                        
    if(response.responseHeader.serviceResult != UA_STATUSCODE_GOOD)
        return false;
    UA_MonitoredItemCreateRequest monRequest =
        UA_MonitoredItemCreateRequest_default(UA_NODEID_NUMERIC(fieldIt->second.nameSpace, fieldIt->second.nodeId));

    UA_MonitoredItemCreateResult monResponse =
    UA_Client_MonitoredItems_createDataChange(client, response.subscriptionId,
                                              UA_TIMESTAMPSTORETURN_BOTH,
                                              monRequest, const_cast<UA_UInt32*>(&cbIt->first), handler_TheAnswerChanged, NULL);
    if(monResponse.statusCode != UA_STATUSCODE_GOOD)
        return false;
    return true;

    /* The first publish request should return the initial value of the variable */
   
}

void OpcUaClient::update()
{
    if(client)
    {
        UA_Client_run_iterate(client, 0);
    }
}



