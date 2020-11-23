/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "SessionID.h"
#include <net/tokenbuffer.h>
namespace vrb {

SessionID::SessionID(const int owner, const bool isPrivate)
    : m_owner(owner)
    , m_isPrivate(isPrivate)
    , m_name(std::string())
    , m_master(owner)
{

}
SessionID::SessionID(int owner, const std::string & name, bool isPrivate)
    : m_owner(owner)
    , m_isPrivate(isPrivate)
    , m_name(name)
    , m_master(owner)
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
int SessionID::master() const
{
    return m_master;
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
void SessionID::setMaster(int master) const
{
    m_master = master;
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
    m_master = other.master();
    return *this;
}

std::ostream& operator<<(std::ostream& s, const vrb::SessionID& id) {
    s << id.name() << ", owner:" << id.owner() << (id.isPrivate() ? ", private" : ", public") << ", master:" << id.master();
    return s;
}

covise::TokenBuffer& operator<<(covise::TokenBuffer& s, const vrb::SessionID& id) {
    s << id.name() << id.owner() << id.isPrivate() << id.master();
    return s;
}
covise::TokenBuffer& operator>>(covise::TokenBuffer& s, vrb::SessionID& id) {
    int owner, master;
    std::string name;
    bool isPrivate;
    s >> name >> owner >> isPrivate >> master;

    id.setOwner(owner);
    id.setName(name);
    id.setPrivate(isPrivate);
    id.setMaster(master);
    return s;
}



}


