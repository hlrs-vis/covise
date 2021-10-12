/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "PROXY.h"

#include <net/message.h>
#include <net/message_sender_interface.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>

namespace covise
{
  IMPL_MESSAGE_WITH_SUB_CLASSES(PROXY, PROXY_TYPE)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateControllerProxy, int, vrbClientID)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateSubProcessProxie, uint32_t, procID, sender_type, senderType, int, timeout)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ProxyCreated, int, port)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ProxyConnected, bool, success)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateCrbProxy, uint32_t, toProcID, uint32_t, fromProcID, int, timeout)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ConnectionTest, int, fromClientID, int, toClientID, int, port, int, timeout)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ConnectionState, int, fromClientID, int, toClientID, ConnectionCapability, capability)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ConnectionCheck, int, fromClientID, int, toClientID)
  IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, Abort, std::vector<int>, processIds)

} // covise
