/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CoolEmAllClient_H
#define CoolEmAllClient_H

#include <string>
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
using namespace covise;
class CoolEmAllClient
{
public:
    CoolEmAllClient(std::string hostname);
    ~CoolEmAllClient();
    double getValue(std::string path, std::string var);
    bool setValue(std::string path, std::string var, std::string value);

private:
    std::string hostname;
    Host *databaseHost;
    SimpleClientConnection *conn;
};

#endif
