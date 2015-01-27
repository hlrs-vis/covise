/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef RB_USE_COVISE_CONFIG
#include <covise/covise_config.h>
#endif

#include "SpawnRemoteDaemon.h"
#include "RemoteRebootMaster.h"
#include "UI.h"

#include <qapplication.h>
#include <qsocket.h>

#include <iostream>
using namespace std;

SpawnRemoteDaemon::SpawnRemoteDaemon(RemoteRebootMaster *master,
                                     const QString &localhostname, int localport,
                                     const QString &hostname, int port)
    : Spawn(master, localhostname, localport)
{

    QStringList inputList;

    QValueVector<QString> argv = master->getArgv();

    if (hostname == "")
    {
        if (argv.size() > 0)
        {
            this->hostname = argv[0];
        }
        else
        {
            inputList.append("Remote Daemon Hostname");
            inputList.append("string");
            inputList.append("");
        }
    }
    else
    {
        this->hostname = hostname;
    }

    if (port == -1)
    {
        if (argv.size() > 1)
        {
            this->port = argv[1].toInt();
        }
        else
        {
            inputList.append("Remote Daemon Port");
            inputList.append("int");
            inputList.append("-1");
        }
    }
    else
    {
        this->port = port;
    }

    if (!inputList.empty())
    {

        const QMap<QString, QString> result = master->getUI()->getUserInputs(inputList);

        if (result["Remote Daemon Hostname"] != "")
        {
            this->hostname = result["Remote Daemon Hostname"];
        }

        if (result["Remote Daemon Port"] != "")
        {
            this->port = result["Remote Daemon Port"].toInt();
        }
    }
}

SpawnRemoteDaemon::~SpawnRemoteDaemon()
{
}

bool SpawnRemoteDaemon::spawn()
{

    if (port == -1)
    {
        const char *line = 0;
#ifdef RB_USE_COVISE_CONFIG
        line = CoviseConfig::getEntry("REMOTE_DAEMON.TCPPort");
#endif
        if (line)
        {
            port = QString(line).toInt();
        }
        else
        {
            port = 31090;
        }
    }

    if (hostname == "")
    {
        cerr << "SpawnRemoteDaemon::spawn err: hostname not given" << endl;
        return false;
    }

    rdaemonSocket = new QSocket(this);
    connect(rdaemonSocket, SIGNAL(connected()),
            this, SLOT(socketConnected()));
    connect(rdaemonSocket, SIGNAL(connectionClosed()),
            this, SLOT(socketConnectionClosed()));
    connect(rdaemonSocket, SIGNAL(readyRead()),
            this, SLOT(socketReadyRead()));
    connect(rdaemonSocket, SIGNAL(error(int)),
            this, SLOT(socketError(int)));

    cerr << "SpawnRemoteDaemon::spawn info: connecting to " << hostname << ":" << port << endl;
    rdaemonSocket->connectToHost(hostname, port);

    return true;
}

QString SpawnRemoteDaemon::getSpawnName() const
{
    return "Remote Daemon";
}

void SpawnRemoteDaemon::socketConnected()
{

    QTextStream os(rdaemonSocket);
    os.setEncoding(QTextStream::Latin1);
    os << "rebootClient " << localhostname << " " << localport << "\n";
    cerr << "SpawnRemoteDaemon::socketConnected info: sent 'rebootClient " << localhostname << " " << localport << "'" << endl;
    rdaemonSocket->close();
}

void SpawnRemoteDaemon::socketConnectionClosed()
{
    delete rdaemonSocket;
    rdaemonSocket = 0;
}

void SpawnRemoteDaemon::socketReadyRead()
{
    cerr << "SpawnRemoteDaemon::socketReadyRead warn: stub called " << endl;
}

void SpawnRemoteDaemon::socketError(int err)
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

    qApp->exit(1);
}
