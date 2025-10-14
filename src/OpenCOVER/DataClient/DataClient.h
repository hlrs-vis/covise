#ifndef OPENCOVER_DATACLIENT_H
#define OPENCOVER_DATACLIENT_H

#include "export.h"
#include "MultiDimensionalArray.h"
#include "ObserverHandle.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>
#include <mutex>

namespace opencover { namespace dataclient {

extern DATACLIENTEXPORT const char *NoNodeName;

class DATACLIENTEXPORT Client {
public:
    virtual ~Client() = default;

    // Connection lifecycle
    virtual void connect() = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;
    enum StatusChange{ Unchanged, Connected, Disconnected};
    StatusChange statusChanged(void* caller);


    //get the available data nodes from the server
    std::vector<std::string> allAvailableScalars() const;
    std::vector<std::string> availableNumericalScalars() const;
    template<typename T>
    std::vector<std::string> availableScalars() const
    {
        return getNodesWith(std::type_index(typeid(T)), true);
    }
    std::vector<std::string> allAvailableArrays() const;
    std::vector<std::string> availableNumericalArrays() const;
    template<typename T>
    std::vector<std::string> availableArrays() const
    {
        return getNodesWith(std::type_index(typeid(T)), false);
    }


    //register nodes to get updates pushed by the server
    //keep the ObserverHandle as long as you want to observe
    virtual [[nodiscard]] ObserverHandle observeNode(const std::string &name) = 0;

    // Pull-style access
    virtual double getNumericScalar(const std::string &name, double *timestep = nullptr) = 0;
    virtual double getNumericScalar(const ObserverHandle &handle, double *timestep = nullptr) = 0;

    virtual size_t numNodeUpdates(const std::string &name) = 0;
    template<typename T>
    MultiDimensionalArray<T> getArray(const std::string &name)
    {
        auto p = getArrayImpl(std::type_index(typeid(T)), name);
        if (!p) return MultiDimensionalArray<T>{};
        if (auto casted = dynamic_cast<MultiDimensionalArray<T>*>(p.get())) {
            return *casted; // copies the MultiDimensionalArray<T> contents
        }
        return MultiDimensionalArray<T>{};
    }

    //called by ObserverHandle
    void queueUnregisterNode(size_t id);

protected:
    void statusChanged();
    Client** getClientReference(const ObserverHandle &handle);
private:

    virtual std::unique_ptr<detail::MultiDimensionalArrayBase> getArrayImpl(std::type_index type, const std::string &name) = 0;
    virtual std::vector<std::string> getNodesWith(std::type_index type, bool isScalar) const = 0;
    virtual std::vector<std::string> getNodesWith(bool isArithmetic, bool isScalar) const = 0;
    std::mutex m_mutex;
    std::vector<size_t> m_nodesToUnregister;
    std::map<void*, bool> m_statusObservers; //stores which objects received a changed status
    bool m_connected = false;
};

}} // namespace

#endif // OPENCOVER_DATACLIENT_H
