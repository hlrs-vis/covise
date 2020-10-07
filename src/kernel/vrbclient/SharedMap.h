#ifndef VISTLE_SHARED_MAP_H
#define VISTLE_SHARED_MAP_H

#include "SharedState.h"

namespace vrb
{

template <class Key, class Val>
class SharedMap : public SharedStateBase
{
typedef std::map<Key, Val> T;

public:
    SharedMap(std::string name, T value = T(), SharedStateType mode = USE_COUPLING_MODE)
        : SharedStateBase(name, mode, "SharedMap"), m_value(value)
    {
        assert(m_registry);
        covise::TokenBuffer data;
        composeData(data);
        subscribe(data.getData());
        setSyncInterval(0);
    }

    SharedMap<Key, Val> &operator=(const T &value)
    {
        if (m_value != value)
        {
            m_value = value;
            push();
        }
        return *this;
    }

    operator T() const
    {
        return m_value;
    }

    void deserializeValue(const regVar *data) override
    {

        //covise::TokenBuffer serializedMap(data->wholeMap);
        //deserialize(serializedMap, m_value);
        //auto change = data->m_changedEtries.begin();
        //while (change != data->m_changedEtries.end()) {
        //    if (change->first > m_value.size()) {
        //        std::cerr << m_className << "," << variableName << ": receive map change out of map size" << std::endl;
        //        return;
        //    }
        //    auto it = m_value.begin();
        //    std::advance(it, change->first);
        //    covise::TokenBuffer c(change->second);
        //    int type, pos;
        //    c >> type;
        //    c >> pos;
        //    if (type != 1 || pos != change->first) {
        //        std::cerr << "Shared Map " << variableName << " :changes in wrong format" << std::endl;
        //    }
        //    deserialize(c, it->second);
        //    ++change;
        //}
    }

    //! sends the value change to the vrb
    void push()
    {
        valueChanged = false;
        covise::TokenBuffer data;
        composeData(data);
        setVar(data.getData());
    }

    const T &value() const
    {
        return m_value;
    }

    ///change a single entrry of the map, the entry nust exist
    void changeEntry(const Key &k, const Val &v)
    {
        bool found = false;
        auto it = m_value.begin();
        if (lastPos > 0)
        {
            std::advance(it, lastPos);
            if (it->first == k)
            {
                found = true;
            }
        }
        if (!found)
        {
            it = m_value.find(k);
            if (it != m_value.end())
            {
                lastPos = std::distance(m_value.begin(), it);
                found = true;
            }
        }
        if (!found)
        {
            std::cerr << m_className << " " << variableName << ": couldn't find entry in map" << std::endl;
            return;
        }
        covise::TokenBuffer data;
        data << (int)ChangeType::ENTRY_CHANGE;
        data << lastPos;
        serialize(data, v);
        setVar(data.getData());
    }

private:

    T m_value;        ///the value of the SharedState
    int lastPos = -1; ///hint to find the changed

    void composeData(covise::TokenBuffer &data)
    {
        covise::TokenBuffer serializedMap;
        serialize(serializedMap, m_value);
        data << (int)WHOLE;
        data << serializedMap;
        serialize(data, std::map<int, covise::DataHandle>()); //we do not send changes since the Shared Map holds the complete value
    }
};
} // namespace vrb

#endif