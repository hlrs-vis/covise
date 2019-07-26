/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /*! Basic class to synchronize the state of any variable with other clients connectet to vrb
 for variables of types other than: bool, int, float, double, string, TokenBuffer
 the serialize and deserialize methods have to be implemented
 make sure the variable name is unique for each SharedState e.g. by naming the variable Plugin.Class.Variablename.ID

 */
#ifndef VRB_SHAREDSTATE_H
#define VRB_SHAREDSTATE_H

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <cassert>

#include <net/tokenbuffer.h>
#include <util/coExport.h>
#include "regClass.h"
#include "SharedStateSerializer.h"
#include "SessionID.h"

namespace covise {
class DataHandle;
}

namespace vrb {
class clientRegVar;
enum SharedStateType {
    USE_COUPLING_MODE, //0  write/read in/from private or puplic session depending on coupling mode 
    NEVER_SHARE, //1        only write to private session to save the state on the server
    ALWAYS_SHARE, //2       always write to public session and share the state within the group
    SHARE_WITH_ALL //3      write to session 0 to share the state within all sessions
};

class  VRBEXPORT SharedStateBase : public regVarObserver {
public:
    SharedStateBase(const std::string name, SharedStateType mode, const std::string& className = "SharedState");

    virtual ~SharedStateBase();

    //! let the SharedState call the given function when the registry entry got changed from the server
    void setUpdateFunction(std::function<void(void)> function);

    //! returns true if the last value change was made by an other client
    bool valueChangedByOther() const;

    std::string getName() const;

    //! is called from the registryAcces when the registry entry got changed from the server
    void update(clientRegVar* theChangedRegEntry) override;
    void setID(const SessionID& id);
    void setMute(bool m);
    bool getMute();
    ///resubscribe to the local registry and the vrb after sessionID has changed
    void resubscribe(const SessionID& id);
    //send value to local registry and vrb if syncInterval allows it
    void frame(double time);
    void setSyncInterval(float time);
    float getSyncInerval();
    //unmute and send value to vrb
    void becomeMaster();
protected:
    //convert tokenbuffer to datatype of the sharedState
    virtual void deserializeValue(const regVar* data) = 0;
    void subscribe(const covise::DataHandle& val);
    void setVar(const covise::DataHandle& val);
    std::string m_className;
    std::string variableName;
    bool doSend = false;
    bool doReceive = false;
    bool valueChanged = false;
    std::function<void(void)> updateCallback;

    VrbClientRegistry* m_registry = nullptr;

private:
    SessionID sessionID = 0; ///the session to send updates to 
    bool muted = false;
    bool send = false;
    float syncInterval = 0.1f; ///how often messages get sent. if >= 0 messages will be sent immediately
    double lastUpdateTime = 0.0;
    std::unique_ptr<const covise::DataHandle>  m_valueData;
};

template <class T>
class  SharedState : public SharedStateBase {
public:
    SharedState<T>(std::string name, T value = T(), SharedStateType mode = USE_COUPLING_MODE)
        : SharedStateBase(name, mode)
        , m_value(value) {
        assert(m_registry);
        covise::TokenBuffer data;
        serializeWithType(data, m_value);
        subscribe(data.getData());
    }

    SharedState<T>& operator=(T value) {
        if (m_value != value) {
            m_value = value;
            push();
        }
        return *this;
    }

    operator T() const {
        return m_value;
    }

    void deserializeValue(const regVar* data) override {
        covise::TokenBuffer d(data->getValue().data(), data->getValue().length());
        deserializeWithType(d, m_value);
    }

    //! sends the value change to the vrb
    void push() {
        valueChanged = false;
        covise::TokenBuffer data;
        serializeWithType(data, m_value);
        setVar(data.getData());
    }

    const T& value() const {
        return m_value;
    }

private:
    T m_value; ///the value of the SharedState

};

template <class Key, class Val>
class  SharedMap : public SharedStateBase {
    typedef std::map<Key, Val> T;
private:

    T m_value; ///the value of the SharedState
    int  lastPos = -1; ///hint to find the changed 

    void composeData(covise::TokenBuffer& data) {
        covise::TokenBuffer serializedMap;
        serialize(serializedMap, m_value);
        data << (int)WHOLE;
        data << serializedMap;
        serialize(data, std::map<int, covise::DataHandle>());//we do not send changes since the Shared Map holds the complete value
    }
public:
	SharedMap(std::string name, T value = T(), SharedStateType mode = USE_COUPLING_MODE)
		: SharedStateBase(name, mode, "SharedMap")
		, m_value(value)
	{
		assert(m_registry);
		covise::TokenBuffer data;
		composeData(data);
        auto l = data.get_length();
        subscribe(data.getData());
		setSyncInterval(0);
	}

    SharedMap<Key, Val>& operator=(const T &value)
	{
		if (m_value != value)
		{
			m_value = value;
			push();
		}
		return *this;
	}

    operator T() const {
        return m_value;
    }

    void deserializeValue(const regVar* data) override {
        covise::TokenBuffer serializedMap(data->wholeMap.data(), data->wholeMap.length());
        deserialize(serializedMap, m_value);
        auto change = data->m_changedEtries.begin();
        while (change != data->m_changedEtries.end()) {
            if (change->first > m_value.size()) {
                std::cerr << m_className << "," << variableName << ": receive map change out of map size" << std::endl;
                return;
            }
            auto it = m_value.begin();
            std::advance(it, change->first);
            covise::TokenBuffer c(change->second.data(), change->second.length());
            int type, pos;
            c >> type;
            c >> pos;
            if (type != 1 || pos != change->first) {
                std::cerr << "Shared Map " << variableName << " :changes in wrong format" << std::endl;
            }
            deserialize(c, it->second);
            ++change;
        }
    }

	//! sends the value change to the vrb
	void push()
	{
		valueChanged = false;
		covise::TokenBuffer data;
		composeData(data);
        setVar(DataHandle(data.getData()));
	}

    const T& value() const {
        return m_value;
    }

    ///change a single entrry of the map, the entry nust exist
    void changeEntry(const Key& k, const Val& v) {
        bool found = false;
        auto it = m_value.begin();
        if (lastPos > 0) {
            std::advance(it, lastPos);
            if (it->first == k) {
                found = true;
            }
        }
        if (!found) {
            it = m_value.find(k);
            if (it != m_value.end()) {
                lastPos = std::distance(m_value.begin(), it);
                found = true;
            }
        }
        if (!found) {
            std::cerr << m_className << " " << variableName << ": couldn't find entry in map" << std::endl;
            return;
        }
        covise::TokenBuffer data;
        data << (int)ChangeType::ENTRY_CHANGE;
        data << lastPos;
        serialize(data, v);
        setVar(covise::DataHandle(data.take_data(), data.get_length()));
    }
};
}
#endif


