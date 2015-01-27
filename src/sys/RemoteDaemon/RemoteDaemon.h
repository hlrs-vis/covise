/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**
 * @Desc: Remote daemon providing services for remote starting of covise and crb.
 * @author: Michael Braitmaier
 * @Date: 2004-04-20
 *
 */
#ifndef ACCESSGRID_DAEMON_H
#define ACCESSGRID_DAEMON_H

#include <iostream>
#include <fstream>

namespace covise
{
class ServerConnection;
class SimpleServerConnection;
class Connection;
class ConnectionList;
class Message;
}
class QSocketNotifier;

#ifdef _WIN32
#define strnicmp strnicmp
#else
#define strnicmp strncasecmp
#endif

//
//
//

class RemoteDaemon
{

    // check in du ...

public:
    /**
       * @Desc: Constructor for server class
       *
       *
       */
    RemoteDaemon();

    /**
       * @Desc: Destructor for server class
       *
       *
       */
    ~RemoteDaemon();

    /**
       * @Desc: Main loop of the server class doing the message loop.
       * @return: no return value
       */
    void loop();

    /**
       * @Desc: Opens the server, by opening the server connection at the port
       *        defined in the constructor. Adds connection to the connection list.
       * @return: 0 per default
       */
    int openServer();

    /**
       * @Desc: Closes the server and therefore closes the server connection
       *
       * @return: no return value
       */
    void closeServer();

    /**
       * @Desc: Prepares and performs application start of covise
       * @return: no return value
       */
    void startCovise();

private:
    covise::SimpleServerConnection *sConn;
    covise::SimpleServerConnection *toController;
    covise::Connection *toAG;
    covise::ConnectionList *connections;
    int port; // port Number (default: 31098) covise.config: ACCESSGRID_DAEMON.TCPPort

    std::streambuf *m_sbuf; // storage for original cerr streaming buffer to reset at end of execution
    std::ofstream *m_file; // file streaming buffer
    char *m_debugFile; // filename of logfile

    /**
       * @Desc: Handles messages sent by the remote client application.
       * @param msg - Message as sent by the client
       * @return: no return value
       */
    void handleClient(covise::Message *);

    /**
       * @Desc: Handles messages sent by the remote client application which do
       *		  not fit into predefined message patterns.
       * @param line - data sent to the server appliction
       * @param conn - connection on which the data was delivered
       * @return: 1 if successful, 0 else
       */
    int handleClient(const char *, covise::Connection *conn);

    /**
       * @Desc: Processes the incoming data on the registered connections and
       *        performs appropriate message processing.
       * @return: 1 if successfull, else 0
       */
    int processMessages();
    covise::Message *msg;
    int receivedMesssages;

    char *GetParameters(const char *pLine, int param);

    int GetParameterCount(const char *pLine);
};
#endif
