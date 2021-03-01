#include "UserInfo.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <net/tokenbuffer.h>


using namespace covise;
using namespace vrb;



namespace vrb{
namespace detail{

Program getUserType(covise::TokenBuffer& tb){
    Program t;
    tb >> t;
    return t;
}
std::string getString(covise::TokenBuffer& tb){
    std::string s;
    tb >> s;
    return s;
}

} //detail

} //vrb


UserInfo::UserInfo(covise::TokenBuffer &tb)
    : userType(detail::getUserType(tb))
    , userName(detail::getString(tb))
    , ipAdress(detail::getString(tb))
    , hostName(detail::getString(tb))
    , email(detail::getString(tb))
    , url(detail::getString(tb))
{
}

UserInfo::UserInfo(Program type)
    : userType(type)
    , userName(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.UserName", covise::Host::getHostname()))
    , ipAdress(covise::Host::getHostaddress())
    , hostName(covise::Host::getHostname())
    , email(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.Email", "covise-users@listserv.uni-stuttgart.de"))
    , url(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.URL", "www.hlrs.de/covise"))
{

}

TokenBuffer &vrb::operator<<(TokenBuffer &tb, const UserInfo &userInfo)
{
    tb << userInfo.userType << userInfo.userName << userInfo.ipAdress << userInfo.hostName << userInfo.email << userInfo.url;
    return tb;
}


std::ostream &vrb::operator<<(std::ostream &os, const UserInfo &userInfo)
{
    os << "name:     " << userInfo.userName << std::endl;
    os << "ip:       " << userInfo.ipAdress << std::endl;
    os << "hostName: " << userInfo.hostName << std::endl;
    os << "email:    " << userInfo.email << std::endl;
    os << "url:      " << userInfo.url << std::endl;
    os << "type:     " << userInfo.userType << std::endl;
    return os;
}


