#ifndef VRB_PRINT_CLIENT_LIST_H
#define VRB_PRINT_CLIENT_LIST_H

#include "RemoteClient.h"
#include <util/coExport.h>
#include <memory>
#include <vector>

namespace vrb
{

/* Prints a list of clients as a formatted table like this:
ID  Name     Email                                    Hostname         
2   visent   covise-users@listserv.uni-stuttgart.de   visent.hlrs.de
*/
VRBCLIENTEXPORT void printClientInfo(const std::vector<const RemoteClient*> &clients);
VRBCLIENTEXPORT void printClientInfo(const std::vector<RemoteClient> &clients);
VRBCLIENTEXPORT void printClientInfo(const std::vector<std::unique_ptr<RemoteClient>> &clients);

}

#endif // !VRB_PRINT_CLIENT_LIST_H