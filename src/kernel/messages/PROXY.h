/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMSG_PROXY_H
#define COMSG_PROXY_H

#include <net/message_macros.h>
#include <net/message_types.h>
#include <util/coExport.h>
#include <vector>
#include <vrb/client/VrbCredentials.h>
namespace covise{

enum class PROXY_TYPE
{
    CreateControllerProxy, CreateSubProcessProxie, ProxyCreated, ProxyConnected, CreateCrbProxy, CrbProxyCreated, ConnectionTest, ConnectionState, ConnectionCheck, Abort

};

enum class ConnectionCapability
{
  NotChecked,
  DirectConnectionPossible,
  ProxyRequired
};

DECL_MESSAGE_WITH_SUB_CLASSES(PROXY, PROXY_TYPE, COMSGEXPORT)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateControllerProxy, COMSGEXPORT, int, vrbClientID)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateSubProcessProxie, COMSGEXPORT, uint32_t, procID, sender_type, senderType, int , timeout)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ProxyCreated, COMSGEXPORT, int, port)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ProxyConnected, COMSGEXPORT, bool, success)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateCrbProxy, COMSGEXPORT, uint32_t, toProcID, uint32_t, fromProcID, int, timeout)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ConnectionTest, COMSGEXPORT, int, fromClientID, int, toClientID, int, port, int, timeout)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ConnectionState, COMSGEXPORT, int, fromClientID, int, toClientID, ConnectionCapability, capability)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ConnectionCheck, COMSGEXPORT, int, fromClientID, int, toClientID)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, Abort, COMSGEXPORT, std::vector<int>, processIds)





} // covise


#endif // !COMSG_PROXY_H
