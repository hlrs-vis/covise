#include "DummyClient.h"
#include <cover/coVRPluginSupport.h>

#include <iostream>
#include <algorithm>

namespace opencover { namespace dataclient {

DummyClient::DummyClient(const std::string& name)
    : ui::Owner(name, cover->ui)
    , m_name(name)
{
    // Create the main menu for the dummy client
    m_menu = std::make_unique<ui::Menu>(m_name, this);

    // Create connect/disconnect button
    m_connectButton = std::make_unique<ui::Button>(m_menu.get(), "Connect");
    m_connectButton->setState(false);
    m_connectButton->setCallback([this](bool state) {
        if (state) {
            connect();
        } else {
            disconnect();
        }
    });
}

DummyClient::~DummyClient()
{
    disconnect();
}

void DummyClient::connect()
{
    if (m_connected)
        return;
        
    m_connected = true;
    m_connectButton->setState(true);
    m_connectButton->setText("Disconnect");
    statusChanged();
    std::cout << "DummyClient '" << m_name << "' connected" << std::endl;
}

void DummyClient::disconnect()
{
    if (!m_connected)
        return;
        
    m_connected = false;
    m_connectButton->setState(false);
    m_connectButton->setText("Connect");
    statusChanged();
    std::cout << "DummyClient '" << m_name << "' disconnected" << std::endl;
}

bool DummyClient::isConnected() const
{
    return m_connected;
}

ObserverHandle DummyClient::observeNode(const std::string& name)
{
    // Check if node already exists
    for (auto& pair : m_nodes) {
        if (pair.second && pair.second->name == name) {
            std::cout << "DummyClient: Node '" << name << "' already observed" << std::endl;
            return ObserverHandle(pair.first, this);
        }
    }
    
    // Create new node with default type (double) and as scalar
    size_t nodeId = m_nextNodeId++;
    auto node = std::make_unique<DummyNode>();
    node->name = name;

    // Create slider for this node
    createSliderForNode(node.get());
    
    m_nodes[nodeId] = std::move(node);
    
    std::cout << "DummyClient: Observing node '" << name << "' with ID " << nodeId << std::endl;
    
    return ObserverHandle(nodeId, this);
}

void DummyClient::createSliderForNode(DummyNode* node)
{
    if (!node || node->slider)
        return;
    
    // Create a slider for this node
    node->slider = new ui::Slider(m_menu.get(), node->name);
    node->slider->setBounds(0.0, 360.0);
    
    // Set callback to update the node value when slider changes
    node->slider->setCallback([node](double value, bool released) {
        node->values.push_back(value);
    });
}

double DummyClient::getNumericScalar(const std::string& name, double* timestep)
{
    auto* node = findNode(name);
    if (!node) {
        std::cerr << "DummyClient: Node '" << name << "' not found" << std::endl;
        return 0.0;
    }
    
    if (timestep) {
        *timestep = 0.0; // Dummy client doesn't track real timestamps
    }
    auto v = node->values.front();
    if(node->values.size() > 1)
        node->values.pop_front();
    return v;
}

double DummyClient::getNumericScalar(const ObserverHandle& handle, double* timestep)
{
    DummyNode *node = nullptr;
    for(const auto& pair : m_nodes) {
        if (handle == pair.first) {
            node = pair.second.get();
            break;
        }
    }
    if(!node)
        return 0.0;
        
    if (timestep) {
        *timestep = 0.0; // Dummy client doesn't track real timestamps
    }
    auto v = node->values.front();
    if(node->values.size() > 1)
        node->values.pop_front();
    return v;
}

size_t DummyClient::numNodeUpdates(const std::string& name)
{
    auto* node = findNode(name);
    if (!node) {
        return 0;
    }
    return node->values.size();
}

std::unique_ptr<detail::MultiDimensionalArrayBase> 
DummyClient::getArrayImpl(std::type_index type, const std::string& name)
{
    if(type != std::type_index(typeid(double)))
        return nullptr;
    auto node = findNode(name);
    if (!node) 
        return nullptr;
    auto value = std::make_unique<dataclient::MultiDimensionalArray<double>>();
    value->dimensions = {1};
    value->data = {getNumericScalar(name)};
    return value;
}

std::vector<std::string> DummyClient::getNodesWith(std::type_index type, bool isScalar) const
{
    std::vector<std::string> result;
    if(type != std::type_index(typeid(double)))
        return result; // Only double type supported in dummy client
    for (const auto& pair : m_nodes) {
        result.push_back(pair.second->name);
    }
    return result;
}

std::vector<std::string> DummyClient::getNodesWith(bool isArithmetic, bool isScalar) const
{
    std::vector<std::string> result;
    if(!isScalar)
        return result; // Only scalars supported in dummy client
    for (const auto& pair : m_nodes) {
        result.push_back(pair.second->name);
    }
    return result;
}

DummyNode* DummyClient::findNode(const std::string& name)
{
    for (auto& pair : m_nodes) {
        if (pair.second && pair.second->name == name) {
            return pair.second.get();
        }
    }
    return nullptr;
}

DummyNode* DummyClient::findNode(size_t id)
{
    auto it = m_nodes.find(id);
    if (it != m_nodes.end()) {
        return it->second.get();
    }
    return nullptr;
}

void DummyClient::unregisterNode(size_t id)
{
    auto it = m_nodes.find(id);
    if (it != m_nodes.end()) {
        std::cout << "DummyClient: Unregistering node '" << it->second->name << "'" << std::endl;
        m_nodes.erase(it);
    }
}

}} // namespace opencover::dataclient
