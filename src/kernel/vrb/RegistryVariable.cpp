#include "RegistryClass.h"
#include "RegistryVariable.h"

#include <net/tokenbuffer.h>
using namespace covise;
using namespace vrb;

vrb::regVar::regVar(regClass *c, const std::string &name, const DataHandle &value, bool isStatic)
    : m_class(c), m_name(name), m_isStatic(isStatic)
{
    setValue(value);
}

void regVar::sendValueChange(covise::TokenBuffer &tb)
{
    tb << m_value;
}

void regVar::sendValue(covise::TokenBuffer &tb)
{

    if (m_class->isMap())
    {
        covise::TokenBuffer v;
        v << static_cast<int>(covise::WHOLE);
        v << m_wholeMap;
        covise::serialize(v, m_changedEtries);
        tb << v;
    }
    else
    {
        sendValueChange(tb);
    }
}

/// returns the m_value
const DataHandle &regVar::value() const
{
    return m_value;
};
/// returns the class of this variable
regClass *regVar::getClass()
{
    return m_class;
};
/// set m_value
void regVar::setValue(const DataHandle &v)
{
    m_value = v;
    if (m_class->isMap())
    {
        covise::TokenBuffer tb(v);
        int type, pos;
        tb >> type;
        switch ((covise::MapChangeType)type)
        {
        case covise::WHOLE:
        {
            tb >> m_wholeMap;
            m_changedEtries.clear();
            covise::deserialize(tb, m_changedEtries); //should be empty after complete map was send from client, may be filled after session was loaded from file
            break;
        }
        case covise::ENTRY_CHANGE:
        {
            tb >> pos;
            m_changedEtries[pos] = v;
        }
        break;
        default:
            std::cerr << "unexpected SharedMap change type: " << type << std::endl;
            break;
        }
    }
}
/// returns true if this Var is static
int regVar::isStatic()
{
    return m_isStatic;
};
/// returns the Name
const std::string &regVar::name() const
{
    return m_name;
};

bool regVar::isDeleted()
{
    return m_isDeleted;
}
void regVar::setDeleted(bool isDeleted)
{
    m_isDeleted = isDeleted;
}

void regVar::serialize(covise::TokenBuffer &tb) const
{
    tb << m_name;
    if (m_class->isMap())
    {
        tb << m_wholeMap;
        covise::serialize(tb, m_changedEtries);
    }
    else
    {
        tb << m_value;
    }
}
void regVar::deserialize(covise::TokenBuffer &tb)
{
    tb >> m_name;
    if (m_class->isMap())
    {
        tb >> m_wholeMap;
        covise::deserialize(tb, m_changedEtries);
    }
    else
    {
        tb >> m_value;
    }
}

template <>
void covise::serialize(covise::TokenBuffer &tb, const regVar &value)
{
    value.serialize(tb);
}

template <>
void covise::deserialize(covise::TokenBuffer &tb, regVar &value)
{
    value.deserialize(tb);
}
