/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "SessionID.h"
#include <net/tokenbuffer.h>
namespace vrb {
SessionID::SessionID()
    :m_owner(0)
    , m_isPrivate(true)
    , m_name(std::string())
{
}
SessionID::SessionID(const int id, const bool isPrivate)
    :m_owner(id)
    ,m_isPrivate(isPrivate)
    ,m_name(std::string())
{

}
SessionID::SessionID(int id, const std::string & name, bool isPrivate)
    :m_owner(id)
    , m_isPrivate(isPrivate)
    , m_name(name)
{
}

std::string SessionID::name() const
{
    return m_name;
}
bool SessionID::isPrivate() const
{
    return m_isPrivate;
}
int SessionID::owner() const
{
    return m_owner;
}

void SessionID::setOwner(int id) const
{
    m_owner = id;
}

void SessionID::setName(const std::string & name)
{
    m_name = name;
}

void SessionID::setPrivate(bool isPrivate)
{
    m_isPrivate = isPrivate;
}

bool SessionID::operator==(const SessionID & other) const
{
    if (m_isPrivate == other.isPrivate() && m_name == other.name())
    {
        return true;
    }
    return false;
}
bool SessionID::operator!=(const SessionID & other) const
{
    return !(*this == other);
}
bool SessionID::operator<(const SessionID & other) const
{
    if ( m_isPrivate && !other.m_isPrivate)
    {
        return true;
    }
    else if(!m_isPrivate && other.m_isPrivate)
    {
        return false;
    }
    else if (m_name < other.m_name)
    {
        return true;
    }
    else
    {
        return false;
    }
}
SessionID &SessionID::operator=(const SessionID & other) {
    m_owner = other.owner();
    m_name = other.name();
    m_isPrivate = other.isPrivate();
    return *this;
}
std::string SessionID::toText() const
{
    std::string state = "private";
    if (!m_isPrivate)
    {
        state = "public";
    }
    return m_name + "   owner: " + std::to_string(m_owner) + "  " + state;
}


covise::TokenBuffer & operator<<(covise::TokenBuffer & tb, const vrb::SessionID & id)
{
    tb << id.owner();
    tb << id.name();
    tb << id.isPrivate();
    return tb;
}

covise::TokenBuffer & operator>>(covise::TokenBuffer & tb, vrb::SessionID & id)
{
    int owner;
    std::string name;
    bool isPrivate;
    tb >> owner;
    tb >> name;
    tb >> isPrivate;
    id.setOwner(owner);
    id.setName(name);
    id.setPrivate(isPrivate);
    return tb;
}
}


