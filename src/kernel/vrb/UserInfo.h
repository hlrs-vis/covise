#ifndef VRBLINET_USERINFO_H
#define VRBLINET_USERINFO_H

#include "ProgramType.h"
#include "SessionID.h"

#include <util/coExport.h>

#include <array>
#include <ostream>

namespace covise
{
    class TokenBuffer;
}

namespace vrb
{
struct VRBEXPORT UserInfo
{
    UserInfo(covise::TokenBuffer &tb);
    UserInfo(Program type);
    const Program userType;
    const std::string userName, ipAdress, hostName, email, url;
};

VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vrb::UserInfo &userInfo);

VRBEXPORT std::ostream &operator<<(std::ostream &os, const vrb::UserInfo &userInfo);


} // namespace vrb

#endif