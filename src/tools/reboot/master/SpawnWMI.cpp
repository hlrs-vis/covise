/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise_config.h>

#include "SpawnWMI.h"
#include "RemoteRebootMaster.h"
#include "UI.h"

#include <covise/covise_process.h>

SpawnWMI::SpawnWMI(RemoteRebootMaster *master,
                   const QString &localhostname, int localport,
                   const QString &command, const QString &hostname, const QString &user, const QString &passwd)
    : Spawn(master, localhostname, localport)
{

    QStringList inputList;

    QValueVector<QString> argv = master->getArgv();

    QString systemUser;
#ifdef WIN32
    systemUser = getenv("USERNAME");
#endif

    QString confCommand = CoviseConfig::getEntry("RemoteReboot.SlaveCommand");

    if (hostname == "")
    {
        if (argv.size() > 0)
        {
            this->hostname = argv[0];
        }
        else
        {
            inputList.append("Remote Hostname");
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
            inputList.append("Remote User");
            inputList.append("string");
            inputList.append(systemUser);
        }
    }
    else
    {
        this->user = user;
    }

    if (command == "")
    {
        if (argv.size() > 2)
        {
            this->command = argv[2];
        }
        else
        {
            inputList.append("Remote Command");
            inputList.append("string");
            if (confCommand.isNull())
            {
                inputList.append("RemoteRebootSlave");
            }
            else
            {
                inputList.append(confCommand);
            }
        }
    }
    else
    {
        this->command = command;
    }

    if (passwd == "")
    {
        if (argv.size() > 3)
        {
            this->passwd = argv[3];
        }
        else
        {
            inputList.append("Password");
            inputList.append("passwd");
            inputList.append("");
        }
    }
    else
    {
        this->passwd = passwd;
    }

    if (!inputList.empty())
    {

        const QMap<QString, QString> result = master->getUI()->getUserInputs(inputList);

        if (result["Remote Command"] != "")
        {
            this->command = result["Remote Command"];
        }

        if (result["Remote Hostname"] != "")
        {
            this->hostname = result["Remote Hostname"];
        }

        if (result["Remote User"] != "")
        {
            if ((result["Remote User"] == systemUser) && (result["Password"] == ""))
            {
                this->user = "";
            }
            else
            {
                this->user = result["Remote User"];
            }
        }

        if ((result["Password"] != "") && (user != ""))
        {
            this->passwd = result["Password"];
        }
        else
        {
            this->passwd = "";
        }
    }
}

SpawnWMI::~SpawnWMI()
{
}

bool SpawnWMI::spawn()
{

    if (hostname == "")
    {
        cerr << "SpawnWMI::spawn err: hostname not given" << endl;
        return false;
    }

    return execProcessWMI(command.latin1(), 0, hostname.latin1(), user.latin1(), passwd.latin1()) == 0;
}

QString SpawnWMI::getSpawnName() const
{
    return "WMI";
}
