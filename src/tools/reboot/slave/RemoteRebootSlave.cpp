/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise_config.h>

#include "RemoteRebootSlave.h"

#include <qapplication.h>
#include <qfile.h>
#include <qregexp.h>
#include <qsocket.h>
#include <qtextstream.h>

#include <iostream>
using namespace std;

RemoteRebootSlave::RemoteRebootSlave(const QString &host, int port)
    : grubConf(0)
{

    socket = new QSocket(this);
    connect(socket, SIGNAL(connected()),
            this, SLOT(socketConnected()));
    connect(socket, SIGNAL(connectionClosed()),
            this, SLOT(socketConnectionClosed()));
    connect(socket, SIGNAL(readyRead()),
            this, SLOT(socketReadyRead()));
    connect(socket, SIGNAL(error(int)),
            this, SLOT(socketError(int)));

    cerr << "RemoteRebootSlave::<init> info: connecting to " << host << ":" << port << endl;
    socket->connectToHost(host, port);

    QString grubConfFilename;

    const char *gkf = CoviseConfig::getEntry("RemoteReboot.GrubConfFilename");
    if (gkf)
    {
        grubConfFilename = gkf;
    }
    else
    {
        grubConfFilename = QString(getenv("COVISEDIR")) + "/config/grub.conf";
    }

    grubConf = new QFile(grubConfFilename);
    if (!grubConf->exists())
    {
        cerr << "RemoteRebootSlave::<init> err: Cannot open config file " << grubConfFilename << endl;
        exit(1);
    }
    else
    {
        grubConf->open(IO_ReadOnly);
        cerr << "RemoteRebootSlave::<init> info: Config file " << grubConfFilename << " opened" << endl;
    }
}

RemoteRebootSlave::~RemoteRebootSlave()
{
    cerr << "RemoteRebootSlave::<dest> info: exiting" << endl;
    delete grubConf;
}

void RemoteRebootSlave::socketConnected()
{
    cerr << "RemoteRebootSlave::socketConnected info: Connection established..." << endl;
}

void RemoteRebootSlave::socketConnectionClosed()
{
    cerr << "RemoteRebootSlave::socketConnectionClosed info: Connection closed, exiting..." << endl;
    qApp->quit();
}

void RemoteRebootSlave::socketReadyRead()
{
    if (socket->canReadLine())
    {
        QString line = socket->readLine().stripWhiteSpace();
        cerr << "RemoteRebootSlave::socketReadyRead info: read " << line << endl;

        if (line == "<list>")
        {
            sendList();
        }
        else if (line == "<set default>")
        {
            setDefaultBoot(socket->readLine().toInt());
        }
        else if (line == "<get default>")
        {
            getDefaultBoot();
        }
        else if (line == "<reboot>")
        {
            reboot();
        }
    }
}

void RemoteRebootSlave::socketError(int err)
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

    cerr << "RemoteRebootSlave::socketError err: " << message << endl;
}

void RemoteRebootSlave::sendList()
{

    if (!socket)
    {
        cerr << "RemoteRebootSlave::sendList warn: socket not open" << endl;
        return;
    }

    QTextStream os(socket);
    os.setEncoding(QTextStream::Latin1);

    grubConf->reset();
    QTextStream is(grubConf);

    while (!is.atEnd())
    {
        QString line = is.readLine().stripWhiteSpace();
        if (line.startsWith("title"))
        {
            QString entry = line.section(QRegExp("\\s"), 1, 0xFFFFFFFF, QString::SectionSkipEmpty);
            cerr << "RemoteRebootSlave::sendList info: sending '" << entry << "'" << endl;
            os << entry << "\n";
        }
    }
    os << "</list>\n";
}

void RemoteRebootSlave::setDefaultBoot(int defaultBoot)
{

    cerr << "RemoteRebootSlave::setDefaultBoot info: setting default to " << defaultBoot << endl;

    QStringList list;

    QTextStream os(socket);
    os.setEncoding(QTextStream::Latin1);

    grubConf->reset();
    QFile bakFile(grubConf->name() + ".bak");
    bakFile.open(IO_ReadWrite);

    QTextStream is(grubConf);
    QTextStream bs(&bakFile);

    while (!is.atEnd())
    {
        QString line = is.readLine();
        bs << line << endl;
        if (line.stripWhiteSpace().startsWith("default"))
        {
            char delim;
            if (line.find('=') >= 0)
            {
                delim = '=';
            }
            else
            {
                delim = ' ';
            }
            cerr << "RemoteRebootSlave::setDefaultBoot info: changing '" << line << "' to '";
            line = QString("default%1%2").arg(delim).arg(defaultBoot);
            cerr << line << "'" << endl;
        }
        list.append(line);
    }

    bakFile.close();
    grubConf->close();

    qApp->processEvents();

    grubConf->open(IO_ReadWrite);

    for (QStringList::iterator i = list.begin(); i != list.end(); ++i)
    {
        cerr << "RemoteRebootSlave::setDefaultBoot info: writing '" << *i << "'" << endl;
        is << *i << endl;
    }

    grubConf->close();
    grubConf->open(IO_ReadOnly);

    os << "</set ok>\n";
}

void RemoteRebootSlave::getDefaultBoot()
{

    QTextStream os(socket);
    os.setEncoding(QTextStream::Latin1);

    grubConf->reset();
    QTextStream is(grubConf);

    while (!is.atEnd())
    {
        QString line = is.readLine().stripWhiteSpace();
        if (line.startsWith("default"))
        {
            char delim;
            if (line.find('=') >= 0)
            {
                delim = '=';
            }
            else
            {
                delim = ' ';
            }
            QString defaultValue = line.section(delim, 1);
            cerr << "RemoteRebootSlave::getDefaultBoot info: sending default " << defaultValue << endl;
            os << defaultValue << "\n";
        }
    }
}

void RemoteRebootSlave::reboot()
{

    cerr << "RemoteRebootSlave::reboot info: Rebooting..." << endl;

#ifdef _WIN32
    char *args[6];
    args[0] = "shutdown";
    args[1] = "/r";
    args[2] = "/f";
    args[3] = "/t";
    args[4] = "5";
    args[5] = NULL;
    const char *executable = "shutdown";
    spawnvp(P_NOWAIT, executable, (const char *const *)args);
#else
    int pid = fork();
    if (pid == 0)
    {
        char *args[5];
        args[0] = "sudo";
        args[1] = "shutdown";
        args[2] = "-r";
        args[3] = "now";
        args[4] = 0;
        const char *executable = "sudo";
        cerr << "RemoteRebootSlave::Commandline: " << args[0] << args[1] << args[2] << args[3] << endl;
        execvp(executable, args);
    }
    else
    {
        signal(SIGCHLD, SIG_IGN);
    }
#endif
}

static void usage()
{

    cerr << "Usage: RemoteClientSlave <hostname> <port>" << endl;
    exit(-1);
}

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        usage();
    }

    QString hostname = argv[1];
    int port = atoi(argv[2]);

    if (!hostname || !port)
        usage();

    QApplication app(argc, argv, false);

    RemoteRebootSlave slave(hostname, port);

    cerr << "RemoteRebootSlave  main info: Entering main loop" << endl;

    return app.exec();
}
