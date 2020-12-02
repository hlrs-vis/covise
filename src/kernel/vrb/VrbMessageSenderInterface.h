/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_MESSAGE_SENDER_INTERFACE_H
#define VRB_MESSAGE_SENDER_INTERFACE_H
#include <util/coExport.h>
namespace covise
{
  class TokenBuffer;
  class MessageBase;
  class Message;
} // namespace covise
namespace vrb
{
  class UdpMessage;
enum class Protocol { TCP, UDP};

  class VRBEXPORT VrbMessageSenderInterface
  {
  public:
    VrbMessageSenderInterface() = default;
    virtual ~VrbMessageSenderInterface() = default;
    bool send(const covise::MessageBase *msg);
    bool send(covise::TokenBuffer &tb, int type, Protocol p = Protocol::TCP);
protected:
    virtual bool sendMessage(const covise::Message *msg) = 0;
    virtual bool sendMessage(const vrb::UdpMessage *msg) = 0;
  };

} // namespace vrb

#endif // ! VRB_MESSAGE_SENDER_INTERFACE_H
