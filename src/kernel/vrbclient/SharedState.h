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
    SharedStateBase(std::string name, SharedStateType mode);

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
    void resubscribe(SessionID& id);
    void frame(double time);
    void setSyncInterval(float time);
    float getSyncInerval();
protected:
    virtual void deserializeValue(covise::TokenBuffer &data) = 0;
    void subscribe(covise::TokenBuffer &&val);
    void setVar(covise::TokenBuffer &&val);
    const std::string className = "SharedState";
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
    float syncInterval = 0.1f;
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

private:
    T m_value; ///the value of the SharedState

};
}
#endif


