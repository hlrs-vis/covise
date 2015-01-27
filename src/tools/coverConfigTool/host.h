/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** host.h
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#ifndef HOST_H
#define HOST_H

#include <qstring.h>
#include <qmap.h>
#include <qptrlist.h>
#include "pipe.h"

enum TrackingSystemType
{
    POLHEMUS,
    MOTIONSTAR,
    FOB,
    DTRACK,
    VRC,
    CAVELIB,
    SPACEBALL,
    SPACEPOINTER,
    MOUSE,
    NONE
};

class Host;

typedef QMap<QString, Host> HostMap;
typedef QMap<QString, Pipe> PipeMap;

class Host
{

public:
    Host();
    void setName(QString name);
    void setPipeMap(PipeMap &pm);
    void setControlHost(bool enabled);
    void setMasterHost(bool enabled);
    void setMasterInterface(QString s);
    void setTrackingSystem(TrackingSystemType t);
    void setTrackingSystemString(QString s);
    void setMonoView(QString s);
    void addPipe(QString pipeId, Pipe p);
    void deletePipe(QString pipeId);
    int getNumPipes();
    int getNumWindows();
    int getNumChannels();
    Pipe *getPipe(QString pipeId);

    QString getName();
    PipeMap *getPipeMap();
    bool isControlHost();
    bool isMasterHost();
    QString getMasterInterface();
    TrackingSystemType getTrackingSystem();
    QString getTrackingString();
    QString getMonoView();

private:
    QString name;
    PipeMap pipeMap;
    bool controlHost;
    bool masterHost;
    QString masterInterface;
    TrackingSystemType trackingSystem;
    QString monoView;
};
#endif // HOST_H
