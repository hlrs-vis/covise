/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NEW_UI.h"
#include <net/message.h>
#include <net/message_sender_interface.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>
namespace covise{

IMPL_MESSAGE_WITH_SUB_CLASSES(NEW_UI, NEW_UI_TYPE)

IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, HandlePartners, 
    LaunchStyle, launchStyle,
    int, timeout,
    std::vector<int>, clients)

TokenBuffer &operator<<(TokenBuffer &tb, const ClientInfo &cl)
{
    tb << cl.id << cl.hostName << cl.style;
    return tb;
}
TokenBuffer &operator>>(TokenBuffer &tb, ClientInfo &cl)
{
    tb >> cl.id >> cl.hostName >> cl.style;
    return tb;
}

IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestAvailablePartners, std::string, dummy)
IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, AvailablePartners, ClientList, clients)
IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestNewHost, char*, hostName, char*, userName, vrb::VrbCredentials, vrbCredentials)
IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, AvailableModules, std::string, coviseVersion, std::vector<std::string>, modules, std::vector<std::string>, categories)
IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, PartnerInfo, int, clientId, std::string, ipAddress, std::string, userName, std::string, coviseVersion, std::vector<std::string>, modules, std::vector<std::string>, categories)


} //covise