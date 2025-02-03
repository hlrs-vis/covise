/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedState.h"
#include "SharedStateManager.h"
#include "VrbClientRegistry.h"

#include <chrono>
#include <ctime>

#include <net/tokenbuffer.h>
#include <net/dataHandle.h>



using namespace covise;
namespace vrb
{

SharedStateBase::SharedStateBase(const std::string name, SharedStateType mode, const std::string& className)
    : variableName(name)
	, m_className(className)
{
    m_registry = nullptr;
    if(SharedStateManager::instance())
    {
        m_registry = SharedStateManager::instance()->getRegistry();
        auto news = SharedStateManager::instance()->add(this, mode);
        sessionID = news.first;
        muted = news.second;
    }
    else
        std::cerr << "Warning: creation of shared state " << name << " before shared state manager has been initialized, this shared state is ignored" << std::endl;
}

SharedStateBase::~SharedStateBase()
{
	if (SharedStateManager::instance())
	{
		if (m_registry)
		{
			m_registry->unsubscribeVar(m_className, variableName);
		}
		SharedStateManager::instance()->remove(this);
	}
}

void SharedStateBase::subscribe(const DataHandle &val)
{
    if(m_registry)
    {
        m_registry->subscribeVar(sessionID, m_className, variableName, val, this);
    }
}

void SharedStateBase::setVar(const DataHandle & val)
{
    m_valueData = val;
	if (syncInterval <= 0 || SharedStateManager::instance()->frameTime() >= lastUpdateTime + syncInterval)
	{
        if(m_registry)
        {
            m_registry->setVar(sessionID, m_className, variableName, val, muted);
        }
	}
	else
	{
		send = true;
	}

}

void SharedStateBase::setUpdateFunction(std::function<void()> function)
{
    updateCallback = function;
}

bool SharedStateBase::valueChangedByOther() const
{
    return valueChanged;
}

std::string SharedStateBase::getName() const
{
    return variableName;
}

void SharedStateBase::update(clientRegVar *theChangedVar)
{
    if (theChangedVar->name() != variableName || theChangedVar->isDeleted())
    {
        return;
    }
    deserializeValue(theChangedVar);
    valueChanged = true;
    if (updateCallback != nullptr)
    {
        updateCallback();
    }
}

void SharedStateBase::setID(const SessionID &id)
{
    sessionID = id;
}

void SharedStateBase::setMute(bool m)
{
    muted = m;
}

bool SharedStateBase::getMute()
{
    return muted;
}

void SharedStateBase::resubscribe(const SessionID &id)
{
    if (m_registry)
    {
        if(m_registry->getClass(m_className)->getVar(variableName))
        {
            m_registry->unsubscribeVar(m_className, variableName, true);
        }

        m_registry->subscribeVar(id, m_className, variableName, m_valueData, this);
    }
}

void SharedStateBase::frame(double time)
{
    if(!m_registry)
        return;
    if (sessionID == SessionID())
    {
        return;
    }
    if (send && time >= lastUpdateTime + syncInterval)
    {
        m_registry->setVar(sessionID, m_className, variableName, m_valueData, muted);
        lastUpdateTime = time;
        send = false;
    }
}

void SharedStateBase::setSyncInterval(float time)
{
    syncInterval = time;
}

float SharedStateBase::getSyncInerval()
{
    return syncInterval;
}
void SharedStateBase::becomeMaster()
{
    muted = false;
    if (m_valueData.length() > 0)
    {
        send = true;
    }
}
}
