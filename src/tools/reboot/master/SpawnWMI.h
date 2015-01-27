/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SPAWNWMI_H
#define SPAWNWMI_H

#include "Spawn.h"

class SpawnWMI : public Spawn
{

public:
    SpawnWMI(RemoteRebootMaster *master,
             const QString &localhostname, int localport,
             const QString &command = "", const QString &hostname = "",
             const QString &user = "", const QString &passwd = "");

    virtual ~SpawnWMI();

    virtual bool spawn();
    virtual QString getSpawnName() const;

protected:
    QString hostname;
    QString user;
    QString passwd;
    QString command;
};
#endif
