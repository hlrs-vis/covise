#include "VrbSetUserInfoMessage.h"
#include <cassert>
#include <net/message_types.h>
using namespace vrb;

UserInfoMessage::UserInfoMessage(const covise::Message *msg)
{
    std::vector<std::unique_ptr<int>> otherClients3;
    assert(msg->type == covise::COVISE_MESSAGE_VRB_SET_USERINFO);

    covise::TokenBuffer tb{msg};

    int num, myPos;
    tb >> num >> myPos;
    for (int i = 0; i < num; i++)
    {
        if (i == myPos)
        {
            tb >> myClientID >> mySession >> myPrivateSession;
            hasMyInfo = true;
        }
        else
        {
            otherClients.push_back(RemoteClient{ tb });
        }
    }
}
