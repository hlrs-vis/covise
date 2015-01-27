/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** host.cpp
 ** 2004-01-29, Matthias Feurer
 ****************************************************************************/

#include <qstring.h>
#include "host.h"
#include "pipe.h"

Host::Host()
{
    name = "";
    pipeMap = PipeMap();
    controlHost = false;
    masterHost = false;
    masterInterface = "";
    trackingSystem = NONE;
    monoView = "";
}

void Host::setName(QString s)
{
    name = s;
}

void Host::setPipeMap(PipeMap &pm)
{
    pipeMap = pm;
}

void Host::setControlHost(bool enabled)
{
    controlHost = enabled;
}

void Host::setMasterHost(bool enabled)
{
    masterHost = enabled;
}

void Host::setMasterInterface(QString s)
{
    masterInterface = s;
}

void Host::setTrackingSystem(TrackingSystemType t)
{
    trackingSystem = t;
}

void Host::setTrackingSystemString(QString s)
{
    if (s == "POLHEMUS")
        setTrackingSystem(POLHEMUS);
    else if (s == "MOTIONSTAR")
        setTrackingSystem(MOTIONSTAR);
    else if (s == "FOB")
        setTrackingSystem(FOB);
    else if (s == "DTRACK")
        setTrackingSystem(DTRACK);
    else if (s == "VRC")
        setTrackingSystem(VRC);
    else if (s == "CAVELIB")
        setTrackingSystem(CAVELIB);
    else if (s == "SPACEBALL")
        setTrackingSystem(SPACEBALL);
    else if (s == "SPACEPOINTER")
        setTrackingSystem(SPACEPOINTER);
    else if (s == "MOUSE")
        setTrackingSystem(MOUSE);
    else
        setTrackingSystem(NONE);
}

void Host::setMonoView(QString s)
{
    monoView = s;
}

void Host::addPipe(QString pipeId, Pipe p)
{
    pipeMap[pipeId] = p;
}

void Host::deletePipe(QString pipeId)
{
    pipeMap.remove(pipeId);
}

int Host::getNumPipes()
{
    return pipeMap.count();
}

int Host::getNumWindows()
{
    int noWins = 0;
    if (pipeMap.count() != 0)
    {
        PipeMap::Iterator pIt;
        for (pIt = pipeMap.begin(); pIt != pipeMap.end(); ++pIt)
        {
            if (pIt.data().getWindowMap() != 0)
            {
                noWins += pIt.data().getNumWindows();
            }
        }
    }
    return noWins;
}

int Host::getNumChannels()
{
    int noChannels = 0;
    if (pipeMap.count() != 0)
    {
        PipeMap::Iterator pIt;
        for (pIt = pipeMap.begin(); pIt != pipeMap.end(); ++pIt)
        {
            if (pIt.data().getWindowMap() != 0)
            {
                WindowMap *wm = pIt.data().getWindowMap();
                WindowMap::Iterator wIt;
                for (wIt = wm->begin(); wIt != wm->end(); ++wIt)
                {
                    noChannels += wIt.data().getNumChannels();
                }
            }
        }
    }
    return noChannels;
}

Pipe *Host::getPipe(QString pipeId)
{
    if (pipeMap.find(pipeId) != pipeMap.end())
    {
        return &pipeMap[pipeId];
    }
    else
        return 0;
}

QString Host::getName()
{
    return name;
}

PipeMap *Host::getPipeMap()
{
    return &pipeMap;
}

bool Host::isControlHost()
{
    return controlHost;
}

bool Host::isMasterHost()
{
    return masterHost;
}

QString Host::getMasterInterface()
{
    return masterInterface;
}

TrackingSystemType Host::getTrackingSystem()
{
    return trackingSystem;
}

QString Host::getTrackingString()
{
    switch (trackingSystem)
    {
    case POLHEMUS:
        return "POLHEMUS";
        break;
    case MOTIONSTAR:
        return "MOTIONSTAR";
        break;
    case FOB:
        return "FOB";
        break;
    case DTRACK:
        return "DTRACK";
        break;
    case VRC:
        return "VRC";
        break;
    case CAVELIB:
        return "CAVELIB";
        break;
    case SPACEBALL:
        return "SPACEBALL";
        break;
    case SPACEPOINTER:
        return "SPACEPOINTER";
        break;
    case MOUSE:
        return "MOUSE";
        break;
    case NONE:
        return "NONE";
        break;
    }
    return "";
}

QString Host::getMonoView()
{
    return monoView;
}
