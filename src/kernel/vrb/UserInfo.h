#ifndef VRBLINET_USERINFO_H
#define VRBLINET_USERINFO_H

#include <string>
#include <array>
#include <ostream>
#include <util/coExport.h>
namespace covise
{
    class TokenBuffer;
}

namespace vrb
{

enum class UserType
{
    Undefined,
    Cover,
    Covise,
    RemoteLauncher,
    LastDummy
};
constexpr std::array<const char *, static_cast<int>(UserType::LastDummy)> UserTypeNames{
    "Undefined",
    "Cover",
    "Covise",
    "RemoteLauncher",
};

VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vrb::UserType &userType);
VRBEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, vrb::UserType &userType);

VRBEXPORT std::ostream &operator<<(std::ostream &os, const vrb::UserType &userType);
struct VRBEXPORT UserInfo
{
    UserInfo() = default;
    UserInfo(UserType type);
    UserType userType = UserType::Undefined;
    std::string ipAdress, hostName, email, url;
};

VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vrb::UserInfo &userInfo);
VRBEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, vrb::UserInfo &userInfo);

VRBEXPORT std::ostream &operator<<(std::ostream &os, const vrb::UserInfo &userInfo);

} // namespace vrb

#endif