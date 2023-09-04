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

const char *opencover::opcua::NoNodeName = "None";

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
        std::cerr << "opcua wrong type " << value->value.type->typeKind << std::endl;
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
    const std::vector<std::string> authentificationModes{"anonymous", "username/password"};
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

class  Browser
{
public:
    
Browser(UA_Client* client, UA_NodeId id){
    UA_BrowseRequest_init(&m_request);
    m_request.requestedMaxReferencesPerNode = 0;
    m_request.nodesToBrowse = UA_BrowseDescription_new();
    m_request.nodesToBrowseSize = 1;
    m_request.nodesToBrowse->browseDirection = UA_BROWSEDIRECTION_FORWARD;
    m_request.nodesToBrowse->includeSubtypes = UA_TRUE;
    
    m_request.nodesToBrowse[0].nodeId = id; /* browse objects folder */
    m_request.nodesToBrowse[0].resultMask = UA_BROWSERESULTMASK_ALL; /* return everything */
    m_response = UA_Client_Service_browse(client, m_request);
}

~Browser(){
    // UA_BrowseResponse_clear(&m_response);
    // UA_BrowseRequest_clear(&m_request);

}

UA_BrowseRequest &request(){
    return m_request;
}

UA_BrowseResponse &response(){
    return m_response;
}

private:
    UA_BrowseRequest m_request;
    UA_BrowseResponse m_response;
};

std::string toString(const UA_String &s)
{
    std::vector<char> v(s.length + 1);
    std::copy(s.data, s.data + s.length, v.begin());
    v[s.length] = '\0';
    return std::string(v.data());
}

int toTypeId(const UA_DataType *type)
{
    auto begin = UA_TYPES;
    auto end = begin + UA_TYPES_COUNT;
    auto it = std::find_if(begin, end, [type](const UA_DataType &t)
    {
        return&t == type;
    });
    return it - UA_TYPES;
}

void Client::listVariablesInResponse(UA_Client* client, UA_BrowseResponse &bResp)
{
    for(size_t i = 0; i < bResp.resultsSize; ++i) {
        for(size_t j = 0; j < bResp.results[i].referencesSize; ++j) {
            UA_ReferenceDescription *ref = &(bResp.results[i].references[j]);
            printf("%-9u %-25.*s %-25.*s %-25.*s %-9u\n", ref->nodeId.nodeId.namespaceIndex,
                (int)ref->nodeId.nodeId.identifier.string.length, ref->nodeId.nodeId.identifier.string.data,
                (int)ref->browseName.name.length, ref->browseName.name.data,
                (int)ref->displayName.text.length, ref->displayName.text.data,
                (int)ref->nodeId.nodeId.identifierType);
            UA_Variant *val = UA_Variant_new();
            auto retval = UA_Client_readValueAttribute(client, ref->nodeId.nodeId, val);
            if(retval == UA_STATUSCODE_GOOD)
            {
                auto node = &m_scalarNodes;
                if(UA_Variant_hasArrayType(val, val->type))
                    node = &m_arrayNodes;
                (*node)[toTypeId(val->type)][toString(ref->browseName.name)] = ref->nodeId.nodeId;
            }

            if(ref->nodeId.nodeId.identifier.string.data)
            {
                auto s = toString(ref->nodeId.nodeId.identifier.string);
                // Browser b(client, UA_NODEID_STRING(1, (char*)"PLC1"));
                // Browser b(client, UA_NODEID_STRING(ref->nodeId.nodeId.namespaceIndex, (char*)s.c_str()));
                Browser b(client, ref->nodeId.nodeId);
                listVariablesInResponse(client, b.response());

            }
        }
    }
}

void syncNodeMap(Client::NodeMap &map)
{
    if(msController->isMaster())
    {
        size_t numTypes = map.size();
        msController->syncData(&numTypes, sizeof(size_t));
        for(const auto &type : map)
        {
            int t = type.first;
            msController->syncData(&t, sizeof(int));
            size_t numNodes = type.second.size();
            msController->syncData(&numNodes, sizeof(size_t));
            for(const auto &node : type.second)
            {
                (void)msController->syncString(node.first);
            }
        }
    } else{
        size_t numTypes;
        msController->syncData(&numTypes, sizeof(size_t));
        std::string name;
        int type;
        size_t numNodes;
        for (size_t i = 0; i < numTypes; i++)
        {
            msController->syncData(&type, sizeof(size_t));
            msController->syncData(&numNodes, sizeof(size_t));
            for (size_t j = 0; j < numNodes; j++)
            {
                name = msController->syncString(name);
                map[type][name] = UA_NodeId();
            }
        }
    }
}

void Client::listVariables(UA_Client* client)
{
    if(msController->isMaster())
    {
        printf("Browsing nodes in objects folder:\n");
        Browser browser(client, UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER));
        
        printf("%-9s %-25s %-25s %-25s %-9s\n", "NAMESPACE", "NODEID", "BROWSE NAME", "DISPLAY NAME", "TYPE");
        
        listVariablesInResponse(client, browser.response());
    }
    syncNodeMap(m_scalarNodes);
    syncNodeMap(m_arrayNodes);
}

bool Client::connectMaster()
{
        /* Create the server and set its config */
        client = UA_Client_new();
        UA_ClientConfig *cc = UA_Client_getConfig(client);
        /* Set securityMode and securityPolicyUri */
        UA_StatusCode retval = UA_STATUSCODE_BAD;
        cc->timeout = 10000;
        if(m_authentificationMode->getValue() == 1)
        {
#ifdef UA_ENABLE_ENCRYPTION
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
#else
            std::cerr << "authentification with username/password might require certificates and therefore open62541 must be uild with encryption support" << std::endl;
#endif
            /* The application URI must be the same as the one in the certificate.
            * The script for creating a self-created certificate generates a certificate
            * with the Uri specified below.*/
            UA_ApplicationDescription_clear(&cc->clientDescription);
            cc->clientDescription.applicationUri = UA_STRING_ALLOC("urn:open62541.server.application");
            cc->clientDescription.applicationType = UA_APPLICATIONTYPE_CLIENT;

            retval = UA_Client_connectUsername(client, m_serverIp->getValue().c_str(), m_username->getValue().c_str(), m_password->getValue().c_str()); 
        } else if(m_authentificationMode->getValue() == 0)
        {
            cc->securityMode = UA_MESSAGESECURITYMODE_NONE;
            retval = UA_Client_connect(client, m_serverIp->getValue().c_str());
        }

        if(retval != UA_STATUSCODE_GOOD) {
            UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Could not connect");
            UA_Client_delete(client);
            client = nullptr;
            return false;
        }

        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Connected!");

        return true;
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
    auto node = findNode(name, 0); //this function needs the type
    bool retval = false;
    if(node && msController->isMaster())
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
            UA_MonitoredItemCreateRequest_default(*node);

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

const UA_NodeId * Client::findNode(const std::string &name, int type, bool silent) const
{
    if(!m_connect->state() || name == NoNodeName || type < 0)
        return nullptr;

    auto typeIt = m_scalarNodes.find(type);
    if(typeIt == m_scalarNodes.end())
    {
        if(!silent)
            std::cerr << "opcua can noty find node " << name << " of type " << type << std::endl;
        return nullptr;
    }
    
    auto node = typeIt->second.find(name);
    if(node == typeIt->second.end())
    {
        if(!silent)
            std::cerr << "opcua node " << name << " not found" << std::endl;
        return nullptr;
    }
    return &node->second;
}

std::vector<char> Client::readData(const UA_NodeId &id, size_t size)
{
    std::vector<char> data;
    if(msController->isMaster())
    {
        UA_Variant *val = UA_Variant_new();
        auto retval = UA_Client_readValueAttribute(client, id, val);
        if(retval == UA_STATUSCODE_GOOD)
        {
            data.resize(size);
            const char* begin = (char*)val->data;
            const char* end = begin + size;
            std::copy(begin, end, data.begin());
        }
    }
    size = data.size();
    msController->syncData(&size, sizeof(size_t));
    msController->syncData(data.data(), size);
    return data;
}

constexpr std::array<int, 8> numericalTypes{UA_TYPES_INT16, UA_TYPES_UINT16, UA_TYPES_INT32, UA_TYPES_UINT32, UA_TYPES_INT64, UA_TYPES_UINT64, UA_TYPES_FLOAT, UA_TYPES_DOUBLE};

double Client::readNumericValue(const std::string &name)
{
    if(name == NoNodeName)
        return  0;
    double retval = 0;
    bool found = false;
    for_<8>([&] (auto i) {      
        typedef typename detail::Type<numericalTypes[i.value]>::type T;
        auto node = findNode(name, numericalTypes[i.value], true);
        if(node)
        {
            auto data = readData(*node, sizeof(T));
            retval = *(T*)data.data();
            found = true;
        }
    });
    if(!found)
        std::cerr << "opcua node " << name << " is not a numeric node" << std::endl; 
    return retval;
}

bool Client::isConnected() const
{
    return client != nullptr;
}

std::vector<std::string> Client::allAvailableNodes(const NodeMap &map) const
{
    std::vector<std::string> fields;
    fields.reserve(map.size() + 1);
    fields.push_back(NoNodeName);
    for(const auto &type :map)
        for(const auto &field : type.second)
            fields.push_back(field.first);
    return fields;
}

std::vector<std::string> Client::availableNumericalNodes(const NodeMap &map) const
{
    std::vector<std::string> v;
    v.push_back(NoNodeName);

    for_<8>([&] (auto i) {      
        typedef typename detail::Type<numericalTypes[i.value]>::type T;
        auto val = availableNodes<T>(map);
        v.insert(v.end(), val.begin() + 1, val.end());
    });
    
    return v;
}

std::vector<std::string> Client::allAvailableScalars() const
{
    return allAvailableNodes(m_scalarNodes);
}


std::vector<std::string> Client::availableNumericalScalars() const
{
    return availableNumericalNodes(m_scalarNodes);
}

std::vector<std::string> Client::allAvailableArrays() const
{
    return allAvailableNodes(m_arrayNodes);

}
std::vector<std::string> Client::availableNumericalArrays() const
{
    return availableNumericalNodes(m_arrayNodes);
}






