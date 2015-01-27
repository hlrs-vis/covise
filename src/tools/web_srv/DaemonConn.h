/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <sys/types.h>

#ifdef _WIN32

#include <process.h>

#else

#include <unistd.h>
#include <inttypes.h>
#endif

#include <comm/transport/coMkAddr.h>
#include <comm/transport/coMkLayer.h>
#include <comm/msg/coSendBuffer.h>
#include <comm/msg/coRecvBuffer.h>
#include <comm/msg/coMsg.h>
#include <comm/msg/coPrintMsg.h>
#include <comm/msg/coCommMsg.h>
#include <comm/logic/coCommunicator.h>
#include <comm/logic/coMkConn.h>
#include <sys/time.h>

#ifdef _STANDARD_C_PLUS_PLUS
#define istrstream std::istringstream
using std::flush;
#endif

class DaemonConn
{
public:
    DaemonConn();
    ~DaemonConn();

    void sendLaunchMsg();
    bool sendLoadMsg(const char *map_file);
    void askState(char *answer);

    void sendQuitMsg();

    void getPartnerHosts(char **partner_hosts, int *num_partners);

    bool isBad();

private:
    void addServerConn(coCommunicator &comm);
    void sendMsg(int32_t type, int32_t subType, const char *text, coCommunicator &comm);
    void sendUIFMsg(const char *text, coCommunicator &comm);
    char *getFile(const char *map_file);
    FILE *openFile(const char *filename, char **file_buf, int *size);

    coCommunicator comm_;
    coConnBase *conn_;
    bool is_bad_;
};
