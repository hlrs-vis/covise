/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedState.h"
#include <vrbclient/regClass.h>
#include "SharedStateManager.h"
#include <vrbclient/VrbClientRegistry.h>
#include <chrono>
#include <ctime>



namespace vrb
{

SharedStateBase::SharedStateBase(std::string name, SharedStateType mode)
    : m_registry(SharedStateManager::instance()->getRegistry())
    , variableName(name)
{
    auto news = SharedStateManager::instance()->add(this, mode);
    sessionID = news.first;
    muted = news.second;
}

SharedStateBase::~SharedStateBase()
{
    m_registry->unsubscribeVar(className, variableName);
    SharedStateManager::instance()->remove(this);
}

void SharedStateBase::subscribe(covise::TokenBuffer && val)
{
    m_registry->subscribeVar(sessionID, className, variableName, std::move(val), this);
}

void SharedStateBase::setVar(covise::TokenBuffer && val)
{
    tb_value = std::move(val);
    send = true;
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
    if (theChangedVar->getName() != variableName || theChangedVar->isDeleted())
    {
        return;
    }
    theChangedVar->getValue().rewind();
    deserializeValue(theChangedVar->getValue());
    valueChanged = true;
    if (updateCallback != nullptr)
    {
        updateCallback();
    }
}

void SharedStateBase::setID(SessionID &id)
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

void SharedStateBase::resubscribe(SessionID &id)
{
    if (!m_registry->getClass(className)->getVar(variableName))
    {
        return;
    }
    m_registry->unsubscribeVar(className, variableName, true);
    covise::TokenBuffer tb;
    m_registry->subscribeVar(id, className, variableName, std::move(tb), this);
}

void SharedStateBase::frame(double time)
{
    if (sessionID == 0)
    {
        return;
    }
    if (send && time >= lastUpdateTime + syncInterval)
    {
        m_registry->setVar(sessionID, className, variableName, std::move(tb_value), muted);
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
}
