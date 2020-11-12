#include "RegistryClass.h"
#include "RegistryVariable.h"

using namespace vrb;

regClass::regClass(const std::string &name, int ID)
    : m_name(name), m_classID(ID)
{
}

regClass::Iter regClass::begin()
{
    return m_variables.begin();
}
regClass::Iter regClass::end()
{
    return m_variables.end();
}

/// get Class ID
int regClass::getID()
{
    return m_classID;
}
void regClass::setID(int id)
{
    m_classID = id;
}
const std::string &regClass::name() const
{
    return m_name;
}

bool regClass::isMap() const
{
    return m_name == sharedMapName;
}
///creates a  a regvar entry  in the map
void regClass::append(regVar *var)
{
    m_variables[var->name()].reset(var);
}
/// getVariableEntry, returns NULL if not found
regVar *regClass::getVar(const std::string &n)
{
    auto it = m_variables.find(n);
    if (it == m_variables.end())
    {
        return nullptr;
    }
    return it->second.get();
}
/// remove a Variable
void regClass::deleteVar(const std::string &n)
{
    m_variables.erase(n);
}
/// remove some Variables
void regClass::deleteAllNonStaticVars()
{
    typename Variables::iterator it = m_variables.begin();
    while (it != m_variables.end())
    {
        if (it->second->isStatic())
        {
            it = m_variables.erase(it);
        }
        else
        {
            ++it;
        }
    }
}
bool regClass::isDeleted()
{
    return m_isDel;
}
void regClass::setDeleted(bool s)
{
    m_isDel = s;
    for (const auto var : m_variables)
    {
        var.second->setDeleted(s);
    }
}

void regClass::serialize(covise::TokenBuffer &tb) const
{
    tb << m_name;
    tb << m_classID;
    tb << m_isDel;

    tb << m_variables.size();
    for (const auto var : m_variables)
    {
        covise::serialize(tb, *var.second);
    }
};

void regClass::deserialize(covise::TokenBuffer &tb)
{
    tb >> m_name;
    tb >> m_classID;
    tb >> m_isDel;

    size_t size;
    tb >> size;
    for (size_t i = 0; i < size; i++)
    {
        auto var = createVar("", covise::DataHandle{});
        covise::deserialize(tb, *var);
        m_variables[var->name()] = var;
    }
}

template <>
void covise::serialize(covise::TokenBuffer &tb, const regClass &value)
{
    value.serialize(tb);
}

template <>
void covise::deserialize(covise::TokenBuffer &tb, regClass &value)
{
    value.deserialize(tb);
}