#ifndef OPENCOVER_OPCUA_CLIENT_H
#define OPENCOVER_OPCUA_CLIENT_H

#include "export.h"
#include "types.h"
#include "uaVariantPtr.h"

#include <DataClient/DataClient.h>
#include <DataClient/ObserverHandle.h>

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <open62541/types.h>
#include <string>
#include <thread>

class UA_Client;

namespace opencover{
namespace opcua
{
extern OPCUACLIENTEXPORT const char *NoNodeName;

struct OPCUACLIENTEXPORT ClientNode{
    std::string name;
    UA_NodeId id;
    int type;
    std::vector<size_t> dimensions = std::vector<size_t>{1};
    std::map<size_t, opencover::dataclient::Client**> subscribers;
    size_t numUpdatesPerFrame = 0;
    std::deque<UA_Variant_ptr> values;
    UA_Variant_ptr lastValue;
    std::set<size_t> updatedSubscribers;
    bool operator==(const ClientNode &other) const{return other.name == name;}
    bool operator==(const std::string &name) const{return name == this->name;}
    bool isScalar() const{return (dimensions.size() == 1 && dimensions[0] == 1);}
    size_t numEntries(){return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());}
};


class OPCUACLIENTEXPORT Client : public opencover::dataclient::Client
{
    friend class opencover::dataclient::ObserverHandle;
private:

    typedef  std::vector<ClientNode> Nodes;
public:

    Client(const std::string &name, size_t queueSize = 0);
    ~Client();

    void connect() override;
    void disconnect() override;
    bool isConnected() const override;



    
    //register nodes to get updates pushed by the server
    //keep the ObserverHandle as long as you want to observe
    [[nodiscard]] opencover::dataclient::ObserverHandle observeNode(const std::string &name) override;
    //get a list of values that the server sent since the last get
    
    double getNumericScalar(const std::string &nodeName, double *timestamp = nullptr) override;
    double getNumericScalar(const opencover::dataclient::ObserverHandle &handle, double *timestamp = nullptr) override;
    double getNumericScalar(ClientNode *node, double *timestamp = nullptr);
    size_t numNodeUpdates(const std::string &nodeName) override;

    
    //called by the server callback
    void updateNode(const std::string& nodeName, UA_DataValue *value);
    
    //calles by ObserverHandle
    void queueUnregisterNode(size_t id);
    
private:
    std::unique_ptr<opencover::dataclient::detail::MultiDimensionalArrayBase> getArrayImpl(std::type_index type, const std::string &name) override;
    std::unique_ptr<opencover::dataclient::detail::MultiDimensionalArrayBase> getArrayImpl(std::type_index type, ClientNode* node);
    std::vector<std::string> getNodesWith(std::type_index type, bool isScalar) const override;
    std::vector<std::string> getNodesWith(bool mustBeArithmetic, bool isScalar) const override;
    
    struct NodeRequest
    {
        ClientNode* node = nullptr;
        size_t requestId = 0;
        opencover::dataclient::Client **clientReference= nullptr;
    };

    size_t m_valueQueueSize = 0;

    //methods that are run in the communication thread on the master
    void runClient();
    void fetchAvailableNodes(UA_Client* client, UA_BrowseResponse &bResp);
    void registerNode(const NodeRequest &request);
    bool connectCommunication();
    std::vector<std::string> getNodesWith(int type, bool isScalar) const;
    //called by RegisterId
    void unregisterNode(size_t id);

    void statusChanged();
    ClientNode* findNode(const std::string &name);
    UA_Variant_ptr getValue(const std::string &name);
    UA_Variant_ptr getValue(ClientNode *node);
    
    opencover::ui::Menu *m_menu;
    opencover::ui::Button *m_connect;
    //times in ms
    std::unique_ptr<opencover::ui::SliderConfigValue> m_requestedPublishingInterval, m_samplingInterval, m_queueSize;
    std::unique_ptr<opencover::ui::FileBrowserConfigValue> m_certificate, m_key;
    std::unique_ptr<opencover::ui::EditFieldConfigValue> m_password, m_username, m_serverIp;
    std::unique_ptr<opencover::ui::SelectionListConfigValue> m_authentificationMode;
    const std::string m_name;
    UA_Client *client = nullptr;

    std::unique_ptr<std::thread> m_communicationThread;
    mutable std::mutex m_mutex;
    Nodes m_availableNodes;
    UA_CreateSubscriptionResponse m_subscription;
    std::atomic_bool m_shutdown{false};
    std::deque<NodeRequest> m_nodesToObserve;
    std::vector<size_t> m_nodesToUnregister;
    size_t m_requestId = 0;
    std::map<void*, bool> m_statusObservers; //stores which objects received a changed status
    bool m_connected = false;
};

}
}




#endif // OPENCOVER_OPCUA_CLIENT_H
