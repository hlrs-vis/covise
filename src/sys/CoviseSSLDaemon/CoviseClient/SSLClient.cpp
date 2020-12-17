/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SSLClient.h"
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <iostream>
#include <string.h>

using namespace std;
using namespace covise;

SSLClient::SSLClient(void)
{
    mConn = NULL;
    bIsRunning = false;
    bCanRun = false;
}

SSLClient::SSLClient(std::string server, int port)
{
    Host locHost(server.c_str());
    cerr << "SSLClient::SSLClient(): Init SSLClientConnection!" << endl;
    mConn = new SSLClientConnection(&locHost, port, sslPasswdCallback, this);
    bCanRun = false;
    if (!mConn)
    {
        cerr << "SSLClient::SSLClient(): SSL client conn failed!" << endl;
        return;
    }
    else
    {
        bCanRun = true;
    }
    bIsRunning = false;

    if (mConn->AttachSSLToSocket(mConn->getSocket()) == 0)
    {
        cerr << "SSLClient::SSLClient(): Attaching socket FD to SSL failed!" << endl;
    }

    cerr << "SSLClient::SSLClient(): Leaving SSLClient::SSLClient()" << endl;
}

SSLClient::~SSLClient(void)
{
    if (mConn)
    {
        delete mConn;
        mConn = NULL;
    }
}

void SSLClient::run(std::string command)
{
    const char *line = NULL;

    cerr << "SSLClient::run()" << endl;
    if (!bCanRun)
    {
        return;
    }

    cerr << "SSLClient::run(): Waiting for SSL_connect()" << endl;
    if (mConn->connect() <= 0)
    {
        cerr << "SSLClient::run(): SSL_Connect failed!" << endl;
        return;
    }

    //int c= getchar();

    cerr << "SSLClient::run(): SubjectUID = " << mConn->getSSLSubject() << endl;

    cerr << "SSLClient::run(): Send Command: " << command.c_str() << endl;
    command += '\n';
    mConn->send(command.c_str(), (unsigned int)(command.size()));

    bIsRunning = true;
    do
    {
        cerr << "SSLClient::run(): check_for_input()" << endl;
        if (mConn->check_for_input(0.01f))
        {
            cerr << "SSLClient::run(): readline()" << endl;
            line = mConn->readLine();
            cerr << "SSLClient::run(): Read: " << line << endl;
            const char *ackString = "ACK\n";
            //mConn->send(ackString,sizeof(char)*strlen(ackString));
            if (command.compare(std::string("check")) == 0)
            {
                cerr << "SSLClient::run(): " << line << endl;
            }
            if (strncmp(ackString, line, 4))
            {
                cerr << "SSLClient::run(): Got ACK!" << endl;
                bIsRunning = false;
            }
        }
        else
        {
            cerr << "SSLClient::run(): No data..." << endl;
        }
    } while (bIsRunning);
    cerr << "SSLClient:run(): Exit run!" << endl;
}

int SSLClient::sslPasswdCallback(char *buf, int size, int rwflag, void *userData)
{
    (void)rwflag;
    (void)userData;

    std::string password;
    cout << "Please enter your password for the certificate: ";
    cin >> password;
    cout << endl;

    if (password.size() > size)
        return 0;

    // XXX: size of buf might not be sufficient
    strncpy(buf, password.c_str(), password.size() /*should be length of buf*/);
    buf[password.size() - 1] = '\0';

    return (int)password.size();
}
