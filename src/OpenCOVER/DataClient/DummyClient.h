#ifndef OPENCOVER_DUMMYCLIENT_H
#define OPENCOVER_DUMMYCLIENT_H

#include "DataClient.h"
#include "export.h"

#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Owner.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace opencover { namespace dataclient {

struct DummyNode {
    std::string name;
    ui::Slider* slider = nullptr;
    std::map<size_t, opencover::dataclient::Client**> subscribers;
    std::deque<double> values{0.0};
    std::set<size_t> updatedSubscribers;
    bool operator==(const DummyNode &other) const{return other.name == name;}
    bool operator==(const std::string &name) const{return name == this->name;}
};

class DATACLIENTEXPORT DummyClient : public Client , public opencover::ui::Owner{
public:
    DummyClient(const std::string& name = "DummyClient");
    ~DummyClient() override;

    // Connection lifecycle - for dummy, these are no-ops
    void connect() override;
    void disconnect() override;
    bool isConnected() const override;

    // Register nodes to get updates via UI sliders
    [[nodiscard]] ObserverHandle observeNode(const std::string& name) override;

    // Pull-style access
    double getNumericScalar(const std::string& name, double* timestep = nullptr) override;
    double getNumericScalar(const ObserverHandle& handle, double* timestep = nullptr) override;
    
    size_t numNodeUpdates(const std::string& name) override;

private:
    std::unique_ptr<detail::MultiDimensionalArrayBase> getArrayImpl(std::type_index type, const std::string& name) override;
    std::vector<std::string> getNodesWith(std::type_index type, bool isScalar) const override;
    std::vector<std::string> getNodesWith(bool isArithmetic, bool isScalar) const override;
    
    // Helper methods
    DummyNode* findNode(const std::string& name);
    DummyNode* findNode(size_t id);
    void createSliderForNode(DummyNode* node);
    void unregisterNode(size_t id);
    
    std::string m_name;
    std::unique_ptr<ui::Menu> m_menu;
    std::unique_ptr<ui::Button> m_connectButton;
    std::map<size_t, std::unique_ptr<DummyNode>> m_nodes;
    size_t m_nextNodeId = 1;
    bool m_connected = false;
};

}} // namespace opencover::dataclient

#endif // OPENCOVER_DUMMYCLIENT_H
