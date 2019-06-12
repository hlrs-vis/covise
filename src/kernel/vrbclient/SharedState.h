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



namespace vrb
{

class clientRegVar;

enum SharedStateType
{
    USE_COUPLING_MODE, //0  write/read in/from private or puplic session depending on coupling mode 
    NEVER_SHARE, //1        only write to private session to save the state on the server
    ALWAYS_SHARE, //2       always write to public session and share the state within the group
    SHARE_WITH_ALL //3      write to session 0 to share the state within all sessions
};

class  VRBEXPORT SharedStateBase : public regVarObserver
{
public:
    SharedStateBase(const std::string name, SharedStateType mode, const std::string &className = "SharedState");

    virtual ~SharedStateBase();

    //! let the SharedState call the given function when the registry entry got changed from the server
    void setUpdateFunction(std::function<void(void)> function);

    //! returns true if the last value change was made by an other client
    bool valueChangedByOther() const;

    std::string getName() const;

    //! is called from the registryAcces when the registry entry got changed from the server
    void update(clientRegVar *theChangedRegEntry) override;
    void setID(SessionID &id);
    void setMute(bool m);
    bool getMute();
    ///resubscribe to the local registry and the vrb after sessionID has changed
    void resubscribe(SessionID& id);
    //send value to local registry and vrb if syncInterval allows it
    void frame(double time);
    void setSyncInterval(float time);
    float getSyncInerval();
    //unmute and send value to vrb
    void becomeMaster();
protected:
    //convert tokenbuffer to datatype of the sharedState
    virtual void deserializeValue(covise::TokenBuffer &data) = 0;
    void subscribe(covise::TokenBuffer &&val);
    void setVar(covise::TokenBuffer &&val);
    std::string m_className;
    std::string variableName;
    bool doSend = false;
    bool doReceive = false;
    bool valueChanged = false;
    std::function<void(void)> updateCallback;

    VrbClientRegistry *m_registry = nullptr;

private:
    SessionID sessionID = 0; ///the session to send updates to 
    bool muted = false;
    bool send = false;
    float syncInterval = 0.1f; ///how often messages get sent. if >= 0 messages will be sent immediately
    double lastUpdateTime = 0.0;
    covise::TokenBuffer tb_value;
};

template <class T>
class  SharedState : public SharedStateBase
{
public:
    SharedState<T>(std::string name, T value = T(), SharedStateType mode = USE_COUPLING_MODE)
        : SharedStateBase(name, mode)
        , m_value(value)
    {
        assert(m_registry);
        covise::TokenBuffer data;
        serializeWithType(data, m_value);
        subscribe(std::move(data));
    }

    SharedState<T> &operator=(T value)
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

    void deserializeValue(covise::TokenBuffer &data) override
    {
		m_oldValue = m_value;
		deserializeWithType(data, m_value);
    }

    //! sends the value change to the vrb
    void push()
    {
        valueChanged = false;
        covise::TokenBuffer data;
        serializeWithType(data, m_value);
        setVar(std::move(data));
    }

    const T &value() const
    {
        return m_value;
    }

	const T& oldValue() const
	{
		return m_oldValue;
	}

private:
    T m_value; ///the value of the SharedState
	T m_oldValue; ///the value the SharedState had before the last change
};

template <class Key, class Val>
class  SharedMap: public SharedStateBase
{
	typedef std::map<Key, Val> T;
private:

	T m_value; ///the value of the SharedState
	T m_oldValue; ///the value the SharedState had before the last change
	int  lastPos = -1; ///hint to find the changed 
public:
	SharedMap(std::string name, T value = T(), SharedStateType mode = USE_COUPLING_MODE)
		: SharedStateBase(name, mode, "SharedMap")
		, m_value(value)
	{
		assert(m_registry);
		covise::TokenBuffer data;
		data << (int) WHOLE;
		serializeWithType(data, m_value);
		subscribe(std::move(data));
		setSyncInterval(0);
	}

	SharedMap<Key, Val>& operator=(T value)
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

	void deserializeValue(covise::TokenBuffer& data) override
	{
		m_oldValue = m_value;
		deserializeWithType(data, m_value);
	}

	//! sends the value change to the vrb
	void push()
	{
		valueChanged = false;
		covise::TokenBuffer data;
		data << ChangeType::WHOLE;
		serialize(data, m_value);
		setVar(std::move(data));
	}

	const T& value() const
	{
		return m_value;
	}

	const T& oldValue() const
	{
		return m_oldValue;
	}
///change a single entrry of the map, the entry nust exist
	void changeEntry(const Key& k, const Val& v)
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
		setVar(std::move(data));
	}
};
}
#endif


