/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef  VRB_MESSAGE_SENDER_INTERFACE_H
#define VRB_MESSAGE_SENDER_INTERFACE_H
#include <net/message_types.h>
#include <util/coExport.h>
namespace covise
{
class TokenBuffer;
class Message;
}
namespace vrb
{
class VRBEXPORT VrbMessageSenderInterface
{
public:
    virtual bool sendMessage(const covise::Message *msg) = 0;
    virtual bool sendMessage(covise::TokenBuffer &tb, covise::covise_msg_type type) = 0;
};
}



#endif // ! VRB_MESSAGE_SENDER_INTERFACE_H
