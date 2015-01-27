/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SPAWNSSH_H
#define SPAWNSSH_H

#include "Spawn.h"

class SpawnSSH : public Spawn
{

public:
    SpawnSSH(RemoteRebootMaster *master,
             const QString &localhostname, int localport,
             const QString &hostname = 0, const QString &user = 0);

    virtual ~SpawnSSH();

    virtual bool spawn();
    virtual QString getSpawnName() const;

protected:
    QString hostname;
    QString user;
};
#endif
