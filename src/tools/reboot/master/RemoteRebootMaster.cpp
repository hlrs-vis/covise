/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _WIN32
#include "locale.h"
#else
#include "unistd.h"
#endif

#ifndef RB_MINIMAL
#include <covise/covise_config.h>
#endif

#include "RemoteRebootMaster.h"
#include "RemoteRebootServer.h"

#include "RemoteRebootConstants.h"

#include "SpawnRemoteDaemon.h"
#include "SpawnSSH.h"

#ifdef _WIN32
#include "SpawnWMI.h"
#endif

#include "UIText.h"
#ifndef RB_MINIMAL
#include "UIGraphical.h"
#endif

#include <qapplication.h>
#include <qdns.h>
#include <qsocket.h>
#include <qtextstream.h>
#include <qtimer.h>

#include <iostream>
using namespace std;

RemoteRebootMaster::RemoteRebootMaster()
    : slaveSocket(0)
    , spawn(0)
    , server(0)
{

#ifdef _WIN32
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;
    wVersionRequested = MAKEWORD(1, 1);

    WSAStartup(wVersionRequested, &wsaData);
#endif

    method = RB_AUTO;
    defaultBoot = -1;
#ifdef RB_MINIMAL
    uiType = 't';
#else
    uiType = 'g';
#endif

    for (int ctr = 1; ctr < qApp->argc(); ++ctr)
    {
        if (ctr < qApp->argc() - 1)
        {
            if (strcmp(qApp->argv()[ctr], "-m") == 0)
            {
                QString methodStr = qApp->argv()[ctr + 1];
                methodStr = methodStr.lower();
                if (methodStr == "rd")
                {
                    method = RB_REMOTE_DAEMON;
                }
                else if (methodStr == "ssh")
                {
                    method = RB_SSH;
                }
                else if (methodStr == "wmi")
                {
                    method = RB_WMI;
                }
                ++ctr;
                continue;
            }
            if (strcmp(qApp->argv()[ctr], "-d") == 0)
            {
                defaultBoot = atoi(qApp->argv()[ctr + 1]);
                ++ctr;
                continue;
            }
            if (strcmp(qApp->argv()[ctr], "-g") == 0)
            {
                uiType = 'g';
                continue;
            }
            if (strcmp(qApp->argv()[ctr], "-t") == 0)
            {
                uiType = 't';
                continue;
            }
        }

        cerr << "RemoteRebootMaster::<init> info: adding " << qApp->argv()[ctr] << " to command line" << endl;
        argv.push_back(qApp->argv()[ctr]);
    }

    if (uiType == 't')
    {
        userInterface = new UIText(this);
    }
#ifndef RB_MINIMAL
    else
    {
        userInterface = new UIGraphical(this);
    }
#endif

    spawnSlave(method);
}

RemoteRebootMaster::~RemoteRebootMaster()
{
}

void RemoteRebootMaster::slaveSocketConnected()
{
    cerr << "RemoteRebootMaster::slaveSocketConnected info: connected" << endl;
}

void RemoteRebootMaster::slaveSocketConnectionClosed()
{
    cerr << "RemoteRebootMaster::socketConnectionClosed info: Connection closed, exiting..." << endl;
    slaveSocket = 0;
    qApp->quit();
}

void RemoteRebootMaster::slaveSocketReadyRead()
{
}

void RemoteRebootMaster::socketError(int err)
{

    QString message;

    switch (err)
    {
    case QSocket::ErrConnectionRefused:
        message = "Connection refused";
        break;
    case QSocket::ErrHostNotFound:
        message = "Host not found";
        break;
    case QSocket::ErrSocketRead:
        message = "Socket read failed";
        break;
    }

    cerr << "RemoteRebootMaster::socketError err: " << message << endl;
}

void RemoteRebootMaster::setSlaveSocket(QSocket *socket)
{
    cerr << "RemoteRebootMaster::setSlaveSocket info: Setting new socket" << endl;
    if (slaveSocket)
        delete slaveSocket;
    slaveSocket = socket;
    //cerr << "RemoteRebootMaster::setSlaveSocket info: Connecting signals...";
    connect(slaveSocket, SIGNAL(connected()),
            this, SLOT(slaveSocketConnected()));
    connect(slaveSocket, SIGNAL(connectionClosed()),
            this, SLOT(slaveSocketConnectionClosed()));
    connect(slaveSocket, SIGNAL(readyRead()),
            this, SLOT(slaveSocketReadyRead()));
    connect(slaveSocket, SIGNAL(error(int)),
            this, SLOT(socketError(int)));
    //cerr << " done" << endl;

    if (defaultBoot >= 0)
    {
        if (setDefaultBoot(defaultBoot))
        {
            reboot();
            qApp->exit(0);
            return;
        }
    }

    userInterface->exec();
}

QStringList RemoteRebootMaster::getBootEntries()
{

    QStringList list;

    if (slaveSocket)
    {

        QTextStream os(slaveSocket);
        os.setEncoding(QTextStream::Latin1);
        os << "<list>\n";

        QString line;

        while (line != "</list>")
        {

            while (slaveSocket && !slaveSocket->canReadLine())
            {
                qApp->processEvents();
            }

            line = slaveSocket->readLine().stripWhiteSpace();
            cerr << "RemoteRebootMaster::getBootEntries info: read '" << line << "'" << endl;
            if (line != "</list>")
                list.append(line);
        }
    }

    return list;
}

bool RemoteRebootMaster::setDefaultBoot(int defaultBoot)
{

    if (slaveSocket)
    {

        QTextStream os(slaveSocket);
        os.setEncoding(QTextStream::Latin1);
        os << "<set default>\n";
        os << defaultBoot << "\n";

        QString line;

        while (slaveSocket && !slaveSocket->canReadLine())
        {
            qApp->processEvents();
        }

        line = slaveSocket->readLine().stripWhiteSpace();
        cerr << "RemoteRebootMaster::setDefaultBoot info: read '" << line << "'" << endl;

        if (line == "</set ok>")
            return true;
    }

    return false;
}

int RemoteRebootMaster::getDefaultBoot()
{

    if (slaveSocket)
    {

        QTextStream os(slaveSocket);
        os.setEncoding(QTextStream::Latin1);
        os << "<get default>\n";

        QString line;

        while (slaveSocket && !slaveSocket->canReadLine())
        {
            qApp->processEvents();
        }

        line = slaveSocket->readLine().stripWhiteSpace();
        cerr << "RemoteRebootMaster::getDefaultBoot info: read '" << line << "'" << endl;

        return line.toInt();
    }

    return 0;
}

void RemoteRebootMaster::reboot()
{

    if (slaveSocket)
    {

        cerr << "RemoteRebootMaster::reboot info: rebooting slave" << endl;

        QTextStream os(slaveSocket);
        os.setEncoding(QTextStream::Latin1);
        os << "<reboot>\n";

        qApp->processEvents();
        slaveSocket->flush();
        qApp->processEvents();
    }
}

void RemoteRebootMaster::spawnSlave(RemoteBootMethod method)
{

    int localport = 0;

    QString line;

#ifdef RB_USE_COVISE_CONFIG
    line = CoviseConfig::getEntry("RemoteReboot.Port");
    if (!line.isNull())
        localport = line.toInt();
#endif

    char localhostname[1024];
    if (gethostname(localhostname, 1024) < 0)
    {
        cerr << "RemoteRebootMaster::spawnSlave err: unable to resolve hostname, can't continue..." << endl;
        qApp->exit(1);
        return;
    }

    QString localfqhostname;

    QDns dns(localhostname, QDns::A);
    QTimer timer;
    timer.start(10000, true);

    do
    {
        qApp->processEvents();
    } while (dns.isWorking() && timer.isActive());

    if (!timer.isActive() || dns.addresses().isEmpty())
    {
        cerr << "RemoteRebootMaster::spawnSlave warn: timeout resolving ip address" << endl;
        localfqhostname = localhostname;
    }
    else
    {
        localfqhostname = dns.addresses().first().toString();
        cerr << "RemoteRebootMaster::spawnSlave info: using ip address " << localfqhostname << endl;
    }

    server = 0;

    do
    {
        delete server;
        if (localport)
            ++localport;
        server = new RemoteRebootServer(localport, this);
    } while (!server->ok());

    localport = server->port();

    if (method == RB_AUTO)
    {

#ifdef RB_USE_COVISE_CONFIG
        line = CoviseConfig::getEntry("RemoteReboot.Method");
#else
        line = QString::null;
#endif

        if (line.isNull())
        {

            cerr << "RemoteRebootMaster::spawnSlave info: setting default method" << endl;

            method = RB_REMOTE_DAEMON;
        }
        else
        {

            if (line.lower() == "remotedaemon")
            {
                method = RB_REMOTE_DAEMON;
            }
            else if (line.lower() == "wmi")
            {
                method = RB_WMI;
            }
            else
            {
                cerr << "RemoteRebootMaster::spawnSlave err: unknown spawn method: " << line << endl;
                qApp->exit(-1);
                return;
            }
        }
    }

    switch (method)
    {

    case RB_AUTO:
        cerr << "SpawnRemoteDaemon::spawnSlave err: internal error: spawn method not set " << endl;
        break;

    case RB_REMOTE_DAEMON:
        cerr << "SpawnRemoteDaemon::spawnSlave info: spawning via remote daemon " << endl;
        spawn = new SpawnRemoteDaemon(this, localfqhostname, localport);
        break;

    case RB_SSH:
        cerr << "SpawnRemoteDaemon::spawnSlave info: spawning via SSH" << endl;
        spawn = new SpawnSSH(this, localfqhostname, localport);
        break;

#ifdef WIN32
    case RB_WMI:
        cerr << "SpawnRemoteDaemon::spawnSlave info: spawning via WMI" << endl;
        spawn = new SpawnWMI(this, localfqhostname, localport);
        break;
#endif

    default:
        cerr << "SpawnRemoteDaemon::spawnSlave err: unknown spawn method" << endl;
    }

    if (!spawn || !spawn->spawn())
    {
        qApp->exit(-1);
    }
}

UI *RemoteRebootMaster::getUI()
{
    return userInterface;
}

const QValueVector<QString> &RemoteRebootMaster::getArgv()
{
    return argv;
}

int main(int argc, char **argv)
{

#ifdef RB_MINIMAL
    QApplication app(argc, argv, false);
#else
    QApplication app(argc, argv);
#endif

    RemoteRebootMaster master;

    return app.exec();
}
