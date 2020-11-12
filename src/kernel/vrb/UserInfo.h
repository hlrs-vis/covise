#ifndef VRBLINET_USERINFO_H
#define VRBLINET_USERINFO_H

#include <string>
#include <ostream>
#include <util/coExport.h>
namespace covise{
    class TokenBuffer;
}

namespace vrb{
struct VRBEXPORT UserInfo{
    std::string ipAdress, hostName, email, url;
};

VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vrb::UserInfo &userInfo);
VRBEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, vrb::UserInfo &userInfo);

VRBEXPORT std::ostream &operator<<(std::ostream &os, const vrb::UserInfo &userInfo);

} //vrb



#endif