#ifndef NET_USERINFO_H
#define NET_USERINFO_H

#include "program_type.h"
#include "tokenbuffer_util.h"

#include <array>
#include <ostream>
#include <util/coExport.h>

namespace covise
{
    class TokenBuffer;
    struct NETEXPORT UserInfo
    {
        UserInfo(covise::TokenBuffer &tb);
        UserInfo(Program type);
        const Program userType;
        const std::string userName, ipAdress, hostName, email, url, icon;
    };

    NETEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const covise::UserInfo &userInfo);

    NETEXPORT std::ostream &operator<<(std::ostream &os, const covise::UserInfo &userInfo);



namespace detail
{

template <>
inline UserInfo get(TokenBuffer &tb)
{
    return UserInfo{tb};
}
}
}
#endif