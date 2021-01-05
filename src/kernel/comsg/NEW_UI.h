

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
    RequestNewHost

};
DECL_MESSAGE_WITH_SUB_CLASSES(NEW_UI, NEW_UI_TYPE, COMSGEXPORT)

DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, HandlePartners, COMSGEXPORT,
    LaunchStyle, launchStyle,
    int, timeout,
    std::vector<int>, clients)


DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestAvailablePartners, COMSGEXPORT, std::string, dummy)
typedef std::vector<std::pair<int, std::string>> ClientList;
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, AvailablePartners, COMSGEXPORT, ClientList, clients)
DECL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestNewHost, COMSGEXPORT, char*, hostName, char*, userName, vrb::VrbCredentials, vrbCredentials)


}//covise
