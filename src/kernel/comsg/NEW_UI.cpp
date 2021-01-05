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

IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestAvailablePartners, std::string, dummy)
IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, AvailablePartners, ClientList, clients)
IMPL_SUB_MESSAGE_CLASS(NEW_UI, NEW_UI_TYPE, RequestNewHost, char*, hostName, char*, userName, vrb::VrbCredentials, vrbCredentials)


} //covise