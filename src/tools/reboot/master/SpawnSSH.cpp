/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SpawnSSH.h"
#include "RemoteRebootMaster.h"
#include "UI.h"

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif

#include <signal.h>

#include <iostream>
using namespace std;

SpawnSSH::SpawnSSH(RemoteRebootMaster *master,
                   const QString &localhostname, int localport,
                   const QString &hostname, const QString &user)

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
            inputList.append("Hostname");
            inputList.append("string");
            inputList.append("");
        }
    }
    else
    {
        this->hostname = hostname;
    }

    if (user == "")
    {
        if (argv.size() > 1)
        {
            this->user = argv[1];
        }
        else
        {
            inputList.append("User");
            inputList.append("string");
            inputList.append("");
        }
    }
    else
    {
        this->user = user;
    }

    if (!inputList.empty())
    {

        const QMap<QString, QString> result = master->getUI()->getUserInputs(inputList);

        if (result["Hostname"] != "")
        {
            this->hostname = result["Hostname"];
        }

        if (result["User"] != "")
        {
            this->user = result["User"];
        }
    }
}

SpawnSSH::~SpawnSSH()
{
}

bool SpawnSSH::spawn()
{

    if (hostname == "")
    {
        cerr << "SpawnSSH::spawn err: hostname not given" << endl;
        return false;
    }

    if (user == "")
    {
        cerr << "SpawnSSH::spawn err: user not given" << endl;
        return false;
    }

    char *user = new char[this->user.length() + 1];
    char *host = new char[this->hostname.length() + 1];
    char *localhost = new char[this->localhostname.length() + 1];
    char *port = new char[sizeof(int) + 1];

    strcpy(user, this->user.latin1());
    strcpy(host, this->hostname.latin1());
    strcpy(localhost, this->localhostname.latin1());
    sprintf(port, "%d", localport);

    char *args[8];
    args[0] = "ssh";
    args[1] = "-l";
    args[2] = user;
    args[3] = host;
    args[4] = "RemoteRebootSlave";
    args[5] = localhost;
    args[6] = port;
    args[7] = NULL;
    const char *executable = "ssh";

#ifdef _WIN32
    spawnvp(P_NOWAIT, executable, (const char *const *)args);
#else
    int pid = fork();
    if (pid == 0)
    {
        execvp(executable, args);
    }
    else
    {
        signal(SIGCHLD, SIG_IGN);
    }
#endif

    delete[] user;
    delete[] host;

    return true;
}

QString SpawnSSH::getSpawnName() const
{
    return "SSH";
}
