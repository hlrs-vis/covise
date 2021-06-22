/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMSG_NEW_UI_H
#define COMSG_NEW_UI_H

#include "coviseLaunchOptions.h"

#include <net/message_macros.h>
#include <util/coExport.h>
#include <vrb/client/VrbCredentials.h>

#include <vector>
#include <utility>

#define COMMA ,
namespace covise{
    enum class NEW_UI_TYPE
    {
        HandlePartners,
        RequestAvailablePartners,
        AvailablePartners,
        RequestNewHost,
        AvailableModules,
        PartnerInfo,
        ConnectionCompleted,
        ChangeClientId
    };
    DECL_MESSAGE_WITH_SUB_CLASSES(NEW_UI, NEW_UI_TYPE, COMSGEXPORT)

    DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, HandlePartners, COMSGEXPORT,
                           LaunchStyle, launchStyle,
                           int, timeout,
                           std::vector<int>, clients)

    DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestAvailablePartners, COMSGEXPORT, std::string, dummy)

    struct ClientInfo
    {
        int id;
        std::string hostName;
        LaunchStyle style;
};

TokenBuffer &operator<<(TokenBuffer &tb, const ClientInfo &cl);
TokenBuffer &operator>>(TokenBuffer &tb, ClientInfo &cl);

typedef std::vector<ClientInfo> ClientList; 
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, AvailablePartners, COMSGEXPORT, ClientList, clients)
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestNewHost, COMSGEXPORT, char *, hostName, char *, userName, vrb::VrbCredentials, vrbCredentials)
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, AvailableModules, COMSGEXPORT, std::string, coviseVersion, std::vector<std::string>, modules, std::vector<std::string>, categories)
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, PartnerInfo, COMSGEXPORT, int, clientId, std::string, ipAddress, std::string, userName, std::string, coviseVersion, std::vector<std::string>, modules, std::vector<std::string>, categories)
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, ConnectionCompleted, COMSGEXPORT, int, dummy);
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, ChangeClientId, COMSGEXPORT, int, oldId, int, newId);

}//covise

#endif // !COMSG_NEW_UI_H