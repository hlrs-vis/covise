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
namespace covise {
class TokenBuffer;
}
namespace vrb {
class VRBEXPORT SessionID {

public:
    SessionID();
    SessionID(int id, bool isPrivate = true);
    SessionID(int id, std::string &name, bool isPrivate = true);
    SessionID(const SessionID &id);

    std::string name() const;
    bool isPrivate()const ;
    int owner() const;
    void setOwner(int id);
    void setName(std::string &name);
    void setPrivate(bool isPrivate);
    bool operator ==(const SessionID &other) const;
    bool operator !=(const SessionID &other) const;
    bool operator <(const SessionID &other) const;
    SessionID &operator=(const SessionID &other);
    std::string toText() const;
private:
    int m_owner = 0;
    std::string m_name = "";
    bool m_isPrivate = true;
};
VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vrb::SessionID &id);
VRBEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, vrb::SessionID &id);
}

#endif // !SESSION_H
