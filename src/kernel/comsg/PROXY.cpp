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
    IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateSubProcessProxie, int, procID)
    IMPL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ProxyCreated, int, port)

} // covise
