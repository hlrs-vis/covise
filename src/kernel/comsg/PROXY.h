/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMSG_PROXY_H
#define COMSG_PROXY_H

#include <net/message_macros.h>
#include <util/coExport.h>
#include <vrb/client/VrbCredentials.h>

namespace covise{

enum class PROXY_TYPE
{
    CreateControllerProxy, CreateSubProcessProxie, ProxyCreated, ProxyConnected, CreateCrbProxy, CrbProxyCreated

};
DECL_MESSAGE_WITH_SUB_CLASSES(PROXY, PROXY_TYPE, COMSGEXPORT)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateControllerProxy, COMSGEXPORT, int, vrbClientID)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateSubProcessProxie, COMSGEXPORT, size_t, procID, int , timeout)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ProxyCreated, COMSGEXPORT, int, port)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, ProxyConnected, COMSGEXPORT, bool, success)
DECL_SUB_MESSAGE_CLASS(PROXY, PROXY_TYPE, CreateCrbProxy, COMSGEXPORT, size_t, toProcID, size_t, fromProcID, int, timeout)




} // covise


#endif // !COMSG_PROXY_H