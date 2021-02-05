/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MESSAGE_SENDER_INTERFACE_H
#define MESSAGE_SENDER_INTERFACE_H
#include <util/coExport.h>
namespace covise
{
  class TokenBuffer;
  class MessageBase;
  class Message;
  class UdpMessage;
enum class Protocol { TCP, UDP};

  class NETEXPORT MessageSenderInterface
  {
  public:
    MessageSenderInterface() = default;
    virtual ~MessageSenderInterface() = default;
    bool send(const covise::MessageBase *msg) const;
    bool send(const covise::Message *msg) const;
    bool send(const UdpMessage *msg) const;
    bool send(covise::TokenBuffer &tb, int type, Protocol p = Protocol::TCP) const;
protected:
    virtual bool sendMessage(const covise::Message *msg) const = 0;
    virtual bool sendMessage(const UdpMessage *msg) const = 0;
  };

} // namespace covise

#endif // ! MESSAGE_SENDER_INTERFACE_H
