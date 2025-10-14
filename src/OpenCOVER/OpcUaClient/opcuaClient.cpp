#include "opcua.h"
#include "opcuaClient.h"

#include <open62541/client_config_default.h>
#include <open62541/client_highlevel.h>
#include <open62541/client_subscriptions.h>
#include <open62541/client.h>
#include <open62541/common.h>
#include <open62541/plugin/log_stdout.h>
#include <open62541/plugin/pki_default.h>

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/VectorEditField.h>

#include <algorithm>
#include <iostream>

using namespace opencover::opcua;
using namespace opencover::opcua::detail;

const char *opencover::opcua::NoNodeName = "None";

// struct DebugGuard
// {
//     DebugGuard(std::mutex &m)
//     : m_guard(m) {
//         std::cerr << "locking mutex in thread " << std::this_thread::get_id() << std::endl;
//     }

//     ~DebugGuard() {
//         std::cerr << "unlocking mutex in thread " << std::this_thread::get_id() << std::endl;
//     }

//     std::lock_guard<std::mutex> m_guard;
// };

// typedef DebugGuard Guard;
typedef std::lock_guard<std::mutex> Guard;


struct ClientSubscription
{
    Client* client;
    std::string nodeName;
    int typeId;
    UA_UInt32 monitoredItemId;
};

std::map<std::string, ClientSubscription> clientSubscriptions;
auto msController = opencover::coVRMSController::instance();

static void handleNodeUpdate(UA_Client *client, UA_UInt32 subId, void *subContext,
                         UA_UInt32 monId, void *monContext, UA_DataValue *value) {
    
                           
    auto &clientSubscition = clientSubscriptions[(const char *)monContext];
    clientSubscition.client->updateNode(clientSubscition.nodeName, value);
    // if(UA_Variant_hasScalarType(&value->value, &UA_TYPES[UA_TYPES_DOUBLE])) {
    //     clientSubscriptions[*(UA_UInt32 *)monContext].val = *(UA_Double*)value->value.data;
    // }
    // else{
    //     std::cerr << "opcua wrong type " << value->value.type->typeKind << std::endl;
    // }
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

Client::Client(const std::string &name, size_t queueSize)
: m_menu(new opencover::ui::Menu(detail::Manager::instance()->m_menu, name))
, m_connect(new opencover::ui::Button(m_menu, "connect"))
, m_requestedPublishingInterval(std::make_unique<opencover::ui::SliderConfigValue>(m_menu, "requestedPublishingInterval", 1, *detail::Manager::instance()->m_config, name))
, m_queueSize(std::make_unique<opencover::ui::SliderConfigValue>(m_menu, "queueSize", 1, *detail::Manager::instance()->m_config, name))
, m_samplingInterval(std::make_unique<opencover::ui::SliderConfigValue>(m_menu, "samplingInterval", 1, *detail::Manager::instance()->m_config, name))
, m_username(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "username", "", *detail::Manager::instance()->m_config, name))
, m_password(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "password", "", *detail::Manager::instance()->m_config, name))
, m_serverIp(std::make_unique<opencover::ui::EditFieldConfigValue>(m_menu, "serverIp", "", *detail::Manager::instance()->m_config, name))
, m_certificate(std::make_unique<opencover::ui::FileBrowserConfigValue>(m_menu, "certificate", "", *detail::Manager::instance()->m_config, name))
, m_key(std::make_unique<opencover::ui::FileBrowserConfigValue>(m_menu, "key", "", *detail::Manager::instance()->m_config, name))
, m_authentificationMode(std::make_unique<opencover::ui::SelectionListConfigValue>(m_menu, "authentification", 0, *detail::Manager::instance()->m_config, name))
, m_name(name)
, m_valueQueueSize(queueSize)
{
    const std::vector<std::string> authentificationModes{"anonymous", "username/password"};
    m_authentificationMode->ui()->setList(authentificationModes);
    m_authentificationMode->ui()->select(m_authentificationMode->getValue());


    m_connect->setCallback([this](bool state){
        state ? connect() : disconnect();
    });

    m_requestedPublishingInterval->ui()->setBounds(1, 100);
    m_samplingInterval->ui()->setBounds(1, 250);
    m_queueSize->ui()->setBounds(1, 100);

}

Client::~Client()
{
    disconnect();
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

UA_Variant_ptr getInitialValue(UA_Client *client, const UA_NodeId &nodeId)
{
    UA_ReadRequest request;
    UA_ReadRequest_init(&request);

    UA_ReadValueId rvi;
    UA_ReadValueId_init(&rvi);
    rvi.nodeId = nodeId; // replace with your NodeId
    rvi.attributeId = UA_ATTRIBUTEID_VALUE;

    request.nodesToRead = &rvi;
    request.nodesToReadSize = 1;

    UA_ReadResponse response = UA_Client_Service_read(client, request);
    UA_Variant_ptr retval;
    // if(response.responseHeader.serviceResult == UA_STATUSCODE_GOOD &&
    // response.resultsSize > 0 && response.results[0].hasValue)
    // {
    //     retval = &response.results[0].value;
    //     retval.timestamp = response.responseHeader.timestamp;
    // }
    // UA_ReadResponse_clear(&response);
    return retval;
}

void Client::runClient()
{
    while(!connectCommunication())
    {
        if(m_shutdown)
            return;
    }
    printf("Browsing nodes in objects folder:\n");
    Browser browser(client, UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER));
    printf("%-9s %-25s %-25s %-25s %-9s\n", "NAMESPACE", "NODEID", "BROWSE NAME", "DISPLAY NAME", "TYPE");
    fetchAvailableNodes(client, browser.response());
    /* Create a subscription */
    UA_CreateSubscriptionRequest request = UA_CreateSubscriptionRequest_default();
    request.requestedPublishingInterval = m_requestedPublishingInterval->getValue();
    m_subscription = UA_Client_Subscriptions_create(client, request, NULL, NULL, NULL);
    m_requestedPublishingInterval->setValue(m_subscription.revisedPublishingInterval);                   
    m_connected = true;
    statusChanged();
    while (!m_shutdown)
    {
        {
            std::unique_lock g(m_mutex);
            while (!m_nodesToObserve.empty())
            {
                auto registration = m_nodesToObserve.back();
                g.unlock();
                registerNode(registration);
                g.lock();
                m_nodesToObserve.pop_back();
            }
            while (!m_nodesToUnregister.empty())
            {
                auto unregister = m_nodesToUnregister.back();
                m_nodesToUnregister.pop_back();
                g.unlock();
                unregisterNode(unregister);
                g.lock();
            }
        }
        UA_Client_run_iterate(client, 1000);

    }
    if(UA_Client_Subscriptions_deleteSingle(client, m_subscription.subscriptionId) == UA_STATUSCODE_GOOD)
    printf("Subscription removed\n");
    //invalidate the SubscripteionIds
    Guard g(m_mutex);
    for(auto &node : m_availableNodes)
        for(auto sub : node.subscribers)
            *sub.second = nullptr;
    UA_Client_disconnect(client);
    UA_Client_delete(client);
    client = nullptr;
    m_connected = false;

}

void Client::fetchAvailableNodes(UA_Client* client, UA_BrowseResponse &bResp)
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
                ClientNode node{toString(ref->browseName.name) , ref->nodeId.nodeId, toTypeId(val->type)};
                if(UA_Variant_hasArrayType(val, val->type))
                {
                    size_t outArrayDimensionsSize;
                    UA_UInt32 *outArrayDimensions;
                    if(UA_Client_readArrayDimensionsAttribute(client, node.id, &outArrayDimensionsSize, &outArrayDimensions) == UA_STATUSCODE_GOOD)
                    {
                        node.dimensions.resize(outArrayDimensionsSize);
                        std::copy(outArrayDimensions, outArrayDimensions + outArrayDimensionsSize, node.dimensions.begin());
                    }
                }
                Guard g(m_mutex);
                m_availableNodes.push_back(node);
                
            }
            UA_Variant_delete(val);

            if(ref->nodeId.nodeId.identifier.string.data)
            {
                auto s = toString(ref->nodeId.nodeId.identifier.string);
                // Browser b(client, UA_NODEID_STRING(1, (char*)"PLC1"));
                // Browser b(client, UA_NODEID_STRING(ref->nodeId.nodeId.namespaceIndex, (char*)s.c_str()));
                Browser b(client, ref->nodeId.nodeId);
                fetchAvailableNodes(client, b.response());

            }
        }
    }
}

void Client::registerNode(const NodeRequest &nodeRequest)
{
    std::unique_lock<std::mutex> g(m_mutex);
    auto node = nodeRequest.node;
    if(!node)
        return;
    
    node->subscribers[nodeRequest.requestId] = nodeRequest.clientReference;
    //note that we requested this scalar for the specific user
    if(node->subscribers.size() > 1)
        return;

    UA_MonitoredItemCreateRequest monRequest =
        UA_MonitoredItemCreateRequest_default(node->id);
    monRequest.requestedParameters.samplingInterval = m_samplingInterval->getValue();
    monRequest.requestedParameters.queueSize = m_queueSize->getValue();
    // monRequest.requestedParameters.discardOldest = false;

    auto it = clientSubscriptions.insert(std::make_pair(nodeRequest.node->name, ClientSubscription{this, nodeRequest.node->name})).first;
    auto subId = m_subscription.subscriptionId;
    g.unlock();
    //this can directly call handleNodeUpdate and therefore must not be locked
    auto result = UA_Client_MonitoredItems_createDataChange(client, subId,
                                            UA_TIMESTAMPSTORETURN_BOTH,
                                            monRequest, const_cast<char*>(it->first.c_str()), handleNodeUpdate, NULL);
    m_samplingInterval->setValue(result.revisedSamplingInterval);
    it->second.monitoredItemId = result.monitoredItemId;
    auto initial = getInitialValue(client, node->id);

    // node->values.push_front(initial);
    // node->lastValue = initial;
    // node->numUpdatesPerFrame++;

}

void Client::unregisterNode(size_t id)
{
    for(auto &node : m_availableNodes)
    {
        auto &subscribers = node.subscribers;
        subscribers.erase(id);
       
        if(subscribers.empty())
        {
            
            auto subsription = std::find_if(clientSubscriptions.begin(), clientSubscriptions.end(), [this, &node](const std::pair<std::string, ClientSubscription> &sub){
                return sub.second.client == this && sub.second.nodeName == node.name;
            });
            if(subsription != clientSubscriptions.end())
            {
                // delete the monitoredItem
                UA_DeleteMonitoredItemsRequest deleteRequest;
                UA_DeleteMonitoredItemsRequest_init(&deleteRequest);
                deleteRequest.subscriptionId = m_subscription.subscriptionId;
                deleteRequest.monitoredItemIds = &subsription->second.monitoredItemId;
                deleteRequest.monitoredItemIdsSize = 1;
                UA_Client_MonitoredItems_delete(client, deleteRequest);
            }
        } 
    }
}


void Client::statusChanged()
{
    Guard g(m_mutex);
    for(auto &obs : m_statusObservers)
        obs.second = false;
}

ClientNode* Client::findNode(const std::string &name)
{
    auto node = std::find_if(m_availableNodes.begin(), m_availableNodes.end(), [&name](const ClientNode &n){return n.name == name;});
    if(node == m_availableNodes.end())
        return nullptr;
    return &*node;
}

UA_Variant_ptr Client::getValue(const std::string &name)
{
    return getValue(findNode(name));
}

UA_Variant_ptr Client::getValue(ClientNode *node)
{
    if(node)
    {
        if(node->values.empty())
            return node->lastValue;
        auto retval = node->values.back();
        node->values.pop_back();
        return retval;
    }
    return UA_Variant_ptr();
}

std::unique_ptr<opencover::dataclient::detail::MultiDimensionalArrayBase> Client::getArrayImpl(std::type_index type, const std::string &name)
{
    auto node = findNode(name);
    if(!node)
        return nullptr;
    return getArrayImpl(type, node);

}

std::unique_ptr<opencover::dataclient::detail::MultiDimensionalArrayBase> Client::getArrayImpl(std::type_index type, ClientNode* node)
{
    if(!node)
        return nullptr;
    std::lock_guard<std::mutex> g(m_mutex);
    auto variant = getValue(node);

    if(!variant.get())
        return nullptr;
    for_<8>([this, &node, &variant, &type] (auto i) {      
        typedef typename detail::Type<numericalTypes[i.value]>::type T;
        if(node->type == numericalTypes[i.value])
        {
            if(type != std::type_index(typeid(T)))
            {
                std::cerr << "opcua type mismatch " << node->type << " requested " << type.name() << std::endl;
                return;
            }
            auto array = std::make_unique<dataclient::MultiDimensionalArray<T>>();
            array->timestamp = variant.timestamp;
            auto size = std::max(size_t(1), variant->arrayLength);
            array->dimensions.push_back(size);
            array->data.resize(size);
            std::memcpy(array->data.data(), variant->data, size * sizeof(T));
            if(variant.get()->arrayDimensionsSize > 0)
            {
                array->dimensions.resize(variant->arrayDimensionsSize);
                for (size_t i = 0; i < variant.get()->arrayDimensionsSize; i++)
                {
                    array->dimensions.push_back(variant->arrayDimensions[i]);
                }
            }
        }
    });
    return nullptr;
}


double Client::getNumericScalar(const opencover::dataclient::ObserverHandle &handle, double *timestamp)
{
    return 0;
}

double Client::getNumericScalar(const std::string &nodeName, double *timestamp)
{
    return 0;
    // return getNumericScalar(findNode(nodeName), timestamp);
}

double Client::getNumericScalar(ClientNode *node, double *timestamp)
{
    double retval = 0;
    if(!node)
        return retval;
    //not very efficient
    for_<8>([this, node, &retval, timestamp] (auto i) {      
        typedef typename detail::Type<numericalTypes[i.value]>::type T;
        if(node->type == numericalTypes[i.value])
        {
            auto array = getArrayImpl(std::type_index(typeid(T)), node);
            auto v = dynamic_cast<dataclient::MultiDimensionalArray<T>*>(array.get());
            if(v->isScalar()){
                if(timestamp)
                    *timestamp = v->timestamp;
                retval = static_cast<double>(v->data[0]);
            }
        }
    });
    return retval;
}

size_t Client::numNodeUpdates(const std::string &nodeName)
{
    Guard g(m_mutex);
    auto node = findNode(nodeName);
    if(node)
        return node->values.size();
    return 0;
}


bool Client::connectCommunication()
{
        /* Create the server and set its config */
        auto tmpClient = UA_Client_new(); 
        UA_ClientConfig *cc = UA_Client_getConfig(tmpClient);
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

            retval = UA_Client_connectUsername(tmpClient, m_serverIp->getValue().c_str(), m_username->getValue().c_str(), m_password->getValue().c_str()); 
        } else if(m_authentificationMode->getValue() == 0)
        {
            cc->securityMode = UA_MESSAGESECURITYMODE_NONE;
            retval = UA_Client_connect(tmpClient, m_serverIp->getValue().c_str());
        }

        if(retval != UA_STATUSCODE_GOOD) {
            UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Could not connect");
            UA_Client_delete(tmpClient);
            return false;
        }

        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Connected!");
        Guard g(m_mutex);
        client = tmpClient;
        return true;
}

opencover::dataclient::ObserverHandle Client::observeNode(const std::string &name)
{
    Guard g(m_mutex);
    dataclient::ObserverHandle id(m_requestId, this);
    auto node = findNode(name);
    if(!node)
    {
        ++m_requestId;
        std::cerr << "could not find opcua node " << name << std::endl;
        return id;

    }
    m_nodesToObserve.push_back(NodeRequest{node, m_requestId, getClientReference(id)});
    // id.m_node = node;
    ++m_requestId;
    return id;
}

void Client::updateNode(const std::string& nodeName, UA_DataValue *value)
{
    Guard g(m_mutex);
    auto node = findNode(nodeName);
    if(!node ||node->type != toTypeId(value->value.type))
        return;
    auto v = UA_Variant_ptr(&value->value);
    v.timestamp = value->sourceTimestamp;
    while(node->values.size() > m_valueQueueSize)
        node->values.pop_back();
    node->values.push_front(v);
    node->lastValue = v;
    node->numUpdatesPerFrame++;
}

void Client::connect()
{
    m_shutdown = false;
    if(msController->isMaster())
    {
        if(m_communicationThread && m_communicationThread->joinable())
            return;
        m_communicationThread.reset(new std::thread([this](){
            runClient();
        }));
    }
    m_connect->setText("disconnect");
    m_connect->setState(true);
}

void Client::disconnect()
{
    if(opencover::coVRMSController::instance()->isMaster())
    {
        /* Clean up */
        if(!m_communicationThread || !m_communicationThread->joinable())
            return;
        statusChanged();
        m_shutdown = true;
        m_communicationThread->join();
        for(auto &node : m_availableNodes)
        {
            for(auto sub : node.subscribers)
                m_nodesToObserve.push_back(NodeRequest{&node, sub.first, sub.second});
            node.subscribers.clear();
        }
    }
    m_connect->setText("connect");
    m_connect->setState(false);
}


bool Client::isConnected() const
{
    Guard g(m_mutex);
    bool b = (client != nullptr) && m_connected;
    b= msController->syncBool(b);
    return b; 
}

std::vector<std::string> Client::getNodesWith(std::type_index type, bool isScalar) const
{
    std::vector<std::string> vec{NoNodeName};
    for(const auto &node : m_availableNodes)
    {
        if(node.isScalar() != isScalar)
            continue;
        for_<8>([this, &node, &type, &vec] (auto i) {
            typedef typename detail::Type<numericalTypes[i.value]>::type T;
            if(type == std::type_index(typeid(T)))
                vec.push_back(node.name);
        });
    }
    return msController->syncVector(vec); 
}

std::vector<std::string> Client::getNodesWith(bool isArithmetic, bool isScalar) const
{
    std::vector<std::string> vec{NoNodeName};
    for(const auto &node : m_availableNodes)
    {
        if(node.isScalar() != isScalar)
            continue;
        for_<8>([this, &node, isArithmetic, &vec] (auto i) {
            typedef typename detail::Type<numericalTypes[i.value]>::type T;
            if(std::is_arithmetic<T>::value == isArithmetic)
                vec.push_back(node.name);
        });
    }
    return msController->syncVector(vec); 
}


