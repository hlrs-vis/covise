#include "UserInfo.h"
#include <net/tokenbuffer.h>
using namespace covise;
using namespace vrb;

TokenBuffer &vrb::operator<<(TokenBuffer &tb, const UserInfo &userInfo)
{
    tb << userInfo.ipAdress << userInfo.hostName << userInfo.email << userInfo.url;
    return tb;
}

TokenBuffer &vrb::operator>>(TokenBuffer &tb, UserInfo &userInfo)
{
    tb >> userInfo.ipAdress >> userInfo.hostName >> userInfo.email >> userInfo.url;
    return tb;
}

std::ostream &vrb::operator<<(std::ostream &os, const UserInfo &userInfo)
{
    os << "userName: " << userInfo.ipAdress << std::endl;
    os << "hostName: " << userInfo.hostName << std::endl;
    os << "email:    " << userInfo.email << std::endl;
    os << "url:      " << userInfo.url << std::endl;
    return os;
}

