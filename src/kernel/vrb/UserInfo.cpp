#include "UserInfo.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <net/tokenbuffer.h>

using namespace covise;
using namespace vrb;


VRBEXPORT covise::TokenBuffer &vrb::operator<<(covise::TokenBuffer &tb, const vrb::UserType &userType){
    tb << static_cast<int>(userType);
    return tb;
}

VRBEXPORT covise::TokenBuffer &vrb::operator>>(covise::TokenBuffer &tb, vrb::UserType &userType){
    int type;
    tb >> type;
    userType = static_cast<UserType>(type);
    return tb;
}

VRBEXPORT std::ostream &vrb::operator<<(std::ostream &os, const vrb::UserType &userType){
    os << UserTypeNames[static_cast<int>(userType)];
    return os;
}


UserInfo::UserInfo(UserType type)
: userType(type)
    ,ipAdress(covise::Host::getHostaddress())
    ,hostName(covise::Host::getHostname())
    ,email(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.Email", "covise-users@listserv.uni-stuttgart.de"))
    ,url(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.URL", "www.hlrs.de/covise"))
{

}

TokenBuffer &vrb::operator<<(TokenBuffer &tb, const UserInfo &userInfo)
{
    tb << userInfo.ipAdress << userInfo.hostName << userInfo.email << userInfo.url << userInfo.userType;
    return tb;
}

TokenBuffer &vrb::operator>>(TokenBuffer &tb, UserInfo &userInfo)
{
    tb >> userInfo.ipAdress >> userInfo.hostName >> userInfo.email >> userInfo.url >> userInfo.userType;
    return tb;
}

std::ostream &vrb::operator<<(std::ostream &os, const UserInfo &userInfo)
{
    os << "userName: " << userInfo.ipAdress << std::endl;
    os << "hostName: " << userInfo.hostName << std::endl;
    os << "email:    " << userInfo.email << std::endl;
    os << "url:      " << userInfo.url << std::endl;
    os << "type:     " << userInfo.userType << std::endl;
    return os;
}

