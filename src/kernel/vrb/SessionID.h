/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///class to identify Sessions
///SessionsID(0,"",false) is global Session
#ifndef SESSION_H
#define SESSION_H

#include <string>
#include <util/coExport.h>
#include <ostream>
namespace covise{
    class TokenBuffer;
}
namespace vrb {
class VRBEXPORT SessionID {

public:
    SessionID() = default;
    SessionID(int owner, bool isPrivate = true);
    SessionID(int owner, const std::string &name, bool isPrivate = true);

    std::string name() const;
    bool isPrivate()const ;
    int owner() const;
    int master() const;
    void setOwner(int id) const;
    void setName(const std::string &name);
    void setPrivate(bool isPrivate);
    void setMaster(int master) const;
    bool operator==(const SessionID &other) const;
    bool operator !=(const SessionID &other) const;
    bool operator <(const SessionID &other) const;
    SessionID &operator=(const SessionID &other);
    SessionID &operator=(const int other) = delete;
private:
	mutable int m_owner = 0;
    mutable int m_master = 0;
    bool m_isPrivate = true;
    std::string m_name;
};

VRBEXPORT std::ostream &operator<<(std::ostream &s, const vrb::SessionID &id);

VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &s, const vrb::SessionID &id);
VRBEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &s, vrb::SessionID &id);
}

#endif // !SESSION_H
