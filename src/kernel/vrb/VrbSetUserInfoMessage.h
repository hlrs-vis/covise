#ifndef VRB_SET_USER_INFO_MESSAGE_H
#define VRB_SET_USER_INFO_MESSAGE_H

#include "RemoteClient.h"

#include <util/coExport.h>
#include <net/tokenbuffer.h>
#include <net/message.h>
#include <vector>
#include <memory>
namespace vrb{

struct VRBEXPORT UserInfoMessage{
    UserInfoMessage(const covise::Message *msg);
    
    //needed to compile with MSVC
    UserInfoMessage(const UserInfoMessage& other) = delete;
    UserInfoMessage(UserInfoMessage&& other) = delete;
    UserInfoMessage& operator=(const UserInfoMessage& other) = delete;
    UserInfoMessage& operator=(UserInfoMessage&& other) = delete;

    std::vector<RemoteClient> otherClients;
    int myClientID = -1;
    SessionID mySession, myPrivateSession;
    bool hasMyInfo = false;
};
} // namespace vrb

#endif // !VRB_SET_USER_INFO_MESSAGE_H