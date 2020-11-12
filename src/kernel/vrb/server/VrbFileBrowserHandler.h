#ifndef VRB_FILE_BROWSER_HANDLER_H
#define VRB_FILE_BROWSER_HANDLER_H

#include "VrbClientList.h"

class QString;
namespace covise {
class TokenBuffer;
class Message;
}
namespace vrb {
void handleFileBrouwserRequest(covise::Message* msg);
void handleFileBrowserRemoteRequest(covise::Message* msg);

void RerouteRequest(const char* location, int type, int senderId, int recvVRBId, QString filter, QString path);


}

#endif