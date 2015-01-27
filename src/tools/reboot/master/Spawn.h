/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SPAWN_H
#define SPAWN_H

#include <qstring.h>

class RemoteRebootMaster;

class Spawn
{

public:
    Spawn(RemoteRebootMaster *master, const QString &localhostname, int localport)
    {
        this->master = master;
        this->localhostname = localhostname;
        this->localport = localport;
    }

    virtual ~Spawn()
    {
    }

    virtual bool spawn() = 0;
    virtual QString getSpawnName() const = 0;

protected:
    RemoteRebootMaster *master;
    QString localhostname;
    int localport;
};
#endif
