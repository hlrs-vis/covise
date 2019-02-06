/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedState.h"

namespace opencover
{
template <>
void serialize<std::vector<std::string>>(covise::TokenBuffer &tb, const std::vector<std::string> &value)
{
    uint32_t size = value.size();
    tb << size;
    for (size_t i = 0; i < size; i++)
    {
        tb << value[i];
    }
}

template <>
void deserialize<std::vector<std::string>>(covise::TokenBuffer &tb, std::vector<std::string> &value)
{
    uint32_t size;
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

SharedStateBase::SharedStateBase(std::string name)
: m_registry(coVRCommunication::instance()->registry)
{
}

SharedStateBase::~SharedStateBase()
{
    m_registry->unsubscribeVar(className.c_str(), 8, variableName.c_str());
}

void SharedStateBase::setUpdateFunction(std::function<void ()> function)
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

void SharedStateBase::update(coVrbRegEntry *theChangedRegEntry)
{
    if (variableName != theChangedRegEntry->getVar())
    {
        return;
    }

    theChangedRegEntry->getData().rewind();
    deserializeValue(theChangedRegEntry->getData());

    valueChanged = true;
    if (updateCallback)
    {
        updateCallback();
    }
}

}
