#include "opcuaClient.h"
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
#include <cover/coVRMSController.h>
#include <cover/ui/SelectionList.h>

#include <algorithm>

using namespace opencover::opcua;
using namespace opencover::opcua::detail;

struct DoubleCallback
{
 std::function<void(double)> cb;
 double val = 0;
};

std::map<UA_UInt32, DoubleCallback> doubleCallbacks;
auto msController = opencover::coVRMSController::instance();

static void handler_TheAnswerChanged(UA_Client *client, UA_UInt32 subId, void *subContext,
                         UA_UInt32 monId, void *monContext, UA_DataValue *value) {
    

    if(UA_Variant_hasScalarType(&value->value, &UA_TYPES[UA_TYPES_DOUBLE])) {
        doubleCallbacks[*(UA_UInt32 *)monContext].val = *(UA_Double*)value->value.data;
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

Client::Client(const std::string &name)
: m_menu(new opencover::ui::Menu(detail::Manager::instance()->m_menu, name))
, m_connect(new opencover::ui::Button(m_menu, "connect"))
, m_username(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "username", "", *detail::Manager::instance()->m_config, name))
, m_password(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "password", "", *detail::Manager::instance()->m_config, name))
, m_serverIp(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "serverIp", "", *detail::Manager::instance()->m_config, name))
, m_certificate(std::make_unique<opencover::ui::FileBrowserConfigValue>(m_menu, "certificate", "", *detail::Manager::instance()->m_config, name))
, m_key(std::make_unique<opencover::ui::FileBrowserConfigValue>(m_menu, "key", "", *detail::Manager::instance()->m_config, name))
, m_authentificationMode(std::make_unique<opencover::ui::SelectionListConfigValue>(m_menu, "authentification", 0, *detail::Manager::instance()->m_config, name))
, m_name(name)
{
    const std::vector<std::string> authentificationModes{"username/password", "anonymous", "key"};
    m_authentificationMode->ui()->setList(authentificationModes);
    m_authentificationMode->ui()->select(m_authentificationMode->getValue());
    m_connect->setCallback([this](bool state){
        state ? connect() : disconnect();
    });


}

void Client::onConnect(const std::function<void(void)> &cb)
{
    m_onConnect = cb;
}

void Client::onDisconnect(const std::function<void(void)> &cb)
{
    m_onDisconnect = cb;
}

Client::~Client()
{
    if(m_connect->state())
    {
        for(auto cb : doubleCallbacks)
        {
            /* Delete the subscription */
            if(UA_Client_Subscriptions_deleteSingle(client, cb.first) == UA_STATUSCODE_GOOD)
                printf("Subscription removed\n");
        }
        disconnect();
    }
    delete m_menu;
}

void Client::listVariables(UA_Client* client)
{
    if(msController->isMaster())
    {
        printf("Browsing nodes in objects folder:\n");
        UA_BrowseRequest bReq;
        UA_BrowseRequest_init(&bReq);
        bReq.requestedMaxReferencesPerNode = 0;
        bReq.nodesToBrowse = UA_BrowseDescription_new();
        bReq.nodesToBrowseSize = 1;
        bReq.nodesToBrowse[0].nodeId = UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER); /* browse objects folder */
        bReq.nodesToBrowse[0].resultMask = UA_BROWSERESULTMASK_ALL; /* return everything */
        UA_BrowseResponse bResp = UA_Client_Service_browse(client, bReq);
        // printf("%-9s %-16s %-16s %-16s\n", "NAMESPACE", "NODEID", "BROWSE NAME", "DISPLAY NAME");
        
        
        for(size_t i = 0; i < bResp.resultsSize; ++i) {
            for(size_t j = 0; j < bResp.results[i].referencesSize; ++j) {
                UA_ReferenceDescription *ref = &(bResp.results[i].references[j]);
                if(ref->nodeId.nodeId.identifierType == UA_NODEIDTYPE_NUMERIC) {
                    // printf("%-9u %-16u %-16.*s %-16.*s\n", ref->nodeId.nodeId.namespaceIndex,
                    //     ref->nodeId.nodeId.identifier.numeric, (int)ref->browseName.name.length,
                    //     ref->browseName.name.data, (int)ref->displayName.text.length,
                    //     ref->displayName.text.data);
                    std::vector<char> v(ref->browseName.name.length + 1);
                    std::copy(ref->browseName.name.data, ref->browseName.name.data + ref->browseName.name.length, v.begin());
                    v[ref->browseName.name.length] = '\0';
                    m_numericalNodes[v.data()] = Field{v.data(), ref->nodeId.nodeId.namespaceIndex, ref->nodeId.nodeId.identifier.numeric};
                } else if(ref->nodeId.nodeId.identifierType == UA_NODEIDTYPE_STRING) {
                    // printf("%-9u %-16.*s %-16.*s %-16.*s\n", ref->nodeId.nodeId.namespaceIndex,
                    //     (int)ref->nodeId.nodeId.identifier.string.length,
                    //     ref->nodeId.nodeId.identifier.string.data,
                    //     (int)ref->browseName.name.length, ref->browseName.name.data,
                    //     (int)ref->displayName.text.length, ref->displayName.text.data);
                }
                /* TODO: distinguish further types */
            }
        }
        UA_BrowseRequest_clear(&bReq);
        UA_BrowseResponse_clear(&bResp);
        size_t size = m_numericalNodes.size();
        msController->syncData(&size, sizeof(size_t));
        for(auto &node : m_numericalNodes)
        {
            msController->syncString(node.second.name);
            msController->syncData(&node.second.nameSpace, sizeof(UA_UInt16));
            msController->syncData(&node.second.nodeId, sizeof(UA_UInt32));
        }
    } else{
        size_t size;
        msController->syncData(&size, sizeof(size_t));
        for (size_t i = 0; i < size; i++)
        {
            Field f;
            f.name = msController->syncString(f.name);
            f.nameSpace = msController->syncData(&f.nameSpace, sizeof(UA_UInt16));
            f.nodeId = msController->syncData(&f.nodeId, sizeof(UA_UInt32));
            m_numericalNodes[f.name] = f;
        }
    }
}

bool Client::connectMaster()
{
/* Create the server and set its config */
        client = UA_Client_new();
        UA_ClientConfig *cc = UA_Client_getConfig(client);
        /* Set securityMode and securityPolicyUri */
        UA_StatusCode retval = UA_STATUSCODE_GOOD;
        if(m_authentificationMode->getValue() == 0)
        {
            cc->securityMode = UA_MESSAGESECURITYMODE_SIGNANDENCRYPT;
            cc->securityPolicyUri = UA_STRING_NULL;
            UA_ByteString certificate = loadFile(m_certificate->getValue().c_str());
            UA_ByteString privateKey  = loadFile(m_key->getValue().c_str());

            /* If no trust list is passed, all certificates are accepted. */
            UA_ClientConfig_setDefaultEncryption(cc, certificate, privateKey,
                                                    NULL, 0, NULL, 0);
            cc->timeout = 10000;
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
            UA_ClientConfig_setAuthenticationUsername(cc, m_username->getValue().c_str(), m_password->getValue().c_str());
        } else if(m_authentificationMode->getValue() == 1)
        {
            cc->securityMode = UA_MESSAGESECURITYMODE_NONE;
        }
        retval = UA_Client_connect(client, m_serverIp->getValue().c_str());

        if(retval != UA_STATUSCODE_GOOD) {
            UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Could not connect");
            UA_Client_delete(client);
            auto b = msController->syncBool(false);
            return false;
        }

        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Connected!");

        auto b = msController->syncBool(true);
}

bool Client::connect()
{
    bool retval = false;
    if(msController->isMaster())
    {
        retval = connectMaster();
    }
    retval = msController->syncBool(retval);
    if(!retval)
    {
        m_connect->setText("connect");
        m_connect->setState(false);
        return false;    
    }
    m_connect->setText("disconnect");
    m_connect->setState(true);
    listVariables(client);
    if(m_onConnect)
        m_onConnect();
    return true;
}

bool Client::disconnect()
{
    if(opencover::coVRMSController::instance()->isMaster())
    {
        /* Clean up */
        if(m_onDisconnect)
            m_onDisconnect();
        UA_Client_disconnect(client);
        UA_Client_delete(client);
        client = nullptr;
    }
    m_connect->setText("connect");
    m_connect->setState(false);
    return true;
}

bool Client::registerDouble(const std::string &name, const std::function<void(double)> &cb)
{
    auto fieldIt = m_numericalNodes.find(name);
    if(fieldIt == m_numericalNodes.end())
        return false;
    bool retval = false;
    if(msController->isMaster())
    {
        /* Create a subscription */
        UA_CreateSubscriptionRequest request = UA_CreateSubscriptionRequest_default();
        UA_CreateSubscriptionResponse response = UA_Client_Subscriptions_create(client, request,
                                                                                NULL, NULL, NULL);

        msController->syncData(&response.subscriptionId, sizeof(UA_UInt32));
        UA_UInt32 subId = response.subscriptionId;
        auto cbIt = doubleCallbacks.emplace(std::make_pair(subId, DoubleCallback{cb})).first;                        
        if(response.responseHeader.serviceResult != UA_STATUSCODE_GOOD)
        {
            auto b = msController->syncBool(false);
            return false;
        }
        UA_MonitoredItemCreateRequest monRequest =
            UA_MonitoredItemCreateRequest_default(UA_NODEID_NUMERIC(fieldIt->second.nameSpace, fieldIt->second.nodeId));

        UA_MonitoredItemCreateResult monResponse =
        UA_Client_MonitoredItems_createDataChange(client, response.subscriptionId,
                                                UA_TIMESTAMPSTORETURN_BOTH,
                                                monRequest, const_cast<UA_UInt32*>(&cbIt->first), handler_TheAnswerChanged, NULL);
        if(monResponse.statusCode != UA_STATUSCODE_GOOD)
        {   
            auto b = msController->syncBool(false);
            return false;
        }
        auto b = msController->syncBool(true);
        return true;
    } else {
        UA_UInt32 subId;
        msController->syncData(&subId, sizeof(UA_UInt32));
        
        doubleCallbacks.emplace(std::make_pair(subId, DoubleCallback{cb}));
        bool retval = msController->syncBool(false);
        return retval;
    }
}

void Client::update()
{
    if(msController->isMaster())
    {
        if(client)
        {
            UA_Client_run_iterate(client, 0);
        }
    } 
    for(auto &val : doubleCallbacks)
    {
        msController->syncData(&val.second.val, sizeof(double));
        val.second.cb(val.second.val);
    }
}

double Client::readValue(const std::string &name)
{
    if(!m_connect->state())
        return 0;
    auto node = m_numericalNodes.find(name);
    if(node == m_numericalNodes.end())
    {
        std::cerr << "Node " << name << " not found returning 0" << std::endl;
        return 0;
    }
    UA_Double value = 0;
    if(msController->isMaster())
    {
        UA_Variant *val = UA_Variant_new();
        auto retval = UA_Client_readValueAttribute(client, UA_NODEID_NUMERIC(node->second.nameSpace, node->second.nodeId), val);
        if(retval == UA_STATUSCODE_GOOD && UA_Variant_isScalar(val) &&
        val->type == &UA_TYPES[UA_TYPES_DOUBLE]) {
                value = *(UA_Double*)val->data;
        }
        UA_Variant_delete(val);

    }
    msController->syncData(&value, sizeof(UA_Double));
    return value;
}

bool Client::isConnected() const
{
    return client != nullptr;
}

std::vector<std::string> Client::availableFields() const
{
    std::vector<std::string> fields;
    fields.reserve(m_numericalNodes.size());
    for(const auto &field : m_numericalNodes)
        fields.push_back(field.first);
    return fields;

}



