/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedState.h"
#include <vrbclient/regClass.h>
#include "SharedStateManager.h"

#include <vrbclient/VrbClientRegistry.h>

namespace vrb
{
template <>
void serialize<std::vector<std::string>>(covise::TokenBuffer &tb, const std::vector<std::string> &value)
{
    int typeID = 0;
    tb << typeID;
    uint32_t size = value.size();
    tb << size;
    for (size_t i = 0; i < size; i++)
    {
        tb << value[i];
    }
}

//template <>
//void serialize<bool>(covise::TokenBuffer &tb, const bool &value)
//{
//    int typeID = 1;
//    tb << typeID;
//    tb << value;
//}
//template <>
//void serialize<int>(covise::TokenBuffer &tb, const int &value)
//{
//    int typeID = 2;
//    tb << 4;
//    tb << value;
//}
//template <>
//void serialize<float>(covise::TokenBuffer &tb, const float &value)
//{
//    int typeID = 3;
//    tb << typeID;
//    tb << value;
//}
//template <>
//void serialize<std::string>(covise::TokenBuffer &tb, const std::string &value)
//{
//    int typeID = 4;
//    tb << typeID;
//    tb << value;
//}
//template <>
//void serialize<char>(covise::TokenBuffer &tb, const char &value)
//{
//    int typeID = 5;
//    tb << typeID;
//    tb << value;
//}

template <>
void deserialize<std::vector<std::string>>(covise::TokenBuffer &tb, std::vector<std::string> &value)
{
    int typeID;
    uint32_t size;
    tb >> typeID;
    tb >> size;
    value.clear();
    value.resize(size);
    for (size_t i = 0; i < size; i++)
    {
        std::string path;
        tb >> path;
        value[i] = path;
    }
}

SharedStateBase::SharedStateBase(std::string name, SharedStateType mode)
    : m_registry(SharedStateManager::instance()->getRegistry())
    , variableName(name)
{
    sessionID = SharedStateManager::instance()->add(this, mode);
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
    if (sessionID == 0)
    {
        return;
    }
    m_registry->setVar(sessionID, className, variableName, std::move(val));
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
    if (theChangedVar->getName() != variableName)
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

void SharedStateBase::setID(int id)
{
    sessionID = id;
}
void SharedStateBase::resubscribe(int id)
{
    if (!m_registry->getClass(className)->getVar(variableName))
    {
        return;
    }
    m_registry->unsubscribeVar(className, variableName, true);
    covise::TokenBuffer tb;
    m_registry->subscribeVar(id, className, variableName, std::move(tb), this);
}
}
