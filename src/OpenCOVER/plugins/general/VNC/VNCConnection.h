/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VNCCONNECTION_H_
#define VNCCONNECTION_H_

#include <list>
#include <string>

class VNCClient;
class VNCWindow;
class VNCWindowBuffer;

class VNCConnection
{
public:
    /**
    * create a new connection to the specified server and port
    */
    VNCConnection(const char *_server, unsigned short _port, const char *_password);
    ~VNCConnection();

    /**
    * update framebuffer if changed on remote host
    */
    void update();

    /// check whether connection exists
    bool isConnected() const
    {
        return connectSuccess;
    }

    /// get server name (either IP-address or host name)
    const std::string &getServer() const
    {
        return serverName;
    }

    /// get port of remote host
    unsigned short getPort() const
    {
        return port;
    }

private:
    bool connectToServer(const char *server, int port, const char *password);
    bool pollServer();
    void disconnect();

    std::string serverName;
    unsigned short port;

    VNCClient *vncClient;
    VNCWindow *vncWindow;
    VNCWindowBuffer *vncWindowBuffer;

    bool connectSuccess;
};

#endif /* VNCCONNECTION_H_ */
