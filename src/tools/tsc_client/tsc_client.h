/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:             ++
// ++                                                                     ++
// ++ Author:                  ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// ++**********************************************************************/

#ifndef TSC_CLIENT_H
#define TSC_CLIENT_H

#include <covise/covise_host.h>
#include <covise/covise_socket.h>
#include <covise/covise_connect.h>
#include <covise/covise_msg.h>

class AsyncClient
{
    int m_port;
    ClientConnection *m_pconn;

public:
    AsyncClient()
    {
        m_pconn = NULL;
    };
    ~AsyncClient()
    {
        if (m_pconn)
            delete m_pconn;
    };
    int open_conn(Host *host, int port);
    void close_conn(void);
    int send_msg(Message *msg);
};
#endif
