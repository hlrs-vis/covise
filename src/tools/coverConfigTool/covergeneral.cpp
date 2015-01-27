/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** covergeneral.cpp
 ** 2004-02-04, Matthias Feurer
 ****************************************************************************/

#include "covergeneral.h"

#include <qstring.h>
#include "host.h"
#include "projectionarea.h"

CoverGeneral::CoverGeneral()
{
    stereoMode = QString();
    viewerPos[0] = 0;
    viewerPos[1] = 0;
    viewerPos[2] = 0;
    floorHeight = 0;
    stepSize = 200;
    menuPos[0] = 0;
    menuPos[1] = 0;
    menuPos[2] = 0;
    menuOrient[0] = -90.0;
    menuOrient[1] = 0.0;
    menuOrient[2] = 0.0;
    menuSize = 1.0;
    sceneSize = 1700;

    //MultiPC:
    syncMode = TCP;
    syncProc = APP;
    serialDevice = QString();
    commando0 = QString();
}

QString CoverGeneral::getStereoMode()
{
    return stereoMode;
}

int CoverGeneral::getViewerPosX()
{
    return viewerPos[0];
}

int CoverGeneral::getViewerPosY()
{
    return viewerPos[1];
}

int CoverGeneral::getViewerPosZ()
{
    return viewerPos[2];
}

int CoverGeneral::getFloorHeight()
{
    return floorHeight;
}

int CoverGeneral::getStepSize()
{
    return stepSize;
}

int CoverGeneral::getMenuPosX()
{
    return menuPos[0];
}

int CoverGeneral::getMenuPosY()
{
    return menuPos[1];
}

int CoverGeneral::getMenuPosZ()
{
    return menuPos[2];
}

double CoverGeneral::getMenuOrient_h()
{
    return menuOrient[0];
}

double CoverGeneral::getMenuOrient_p()
{
    return menuOrient[1];
}

double CoverGeneral::getMenuOrient_r()
{
    return menuOrient[2];
}

double CoverGeneral::getMenuSize()
{
    return menuSize;
}

int CoverGeneral::getSceneSize()
{
    return sceneSize;
}

SyncModeType CoverGeneral::getSyncMode()
{
    return syncMode;
}

SyncProcessType CoverGeneral::getSyncProcess()
{
    return syncProc;
}

QString CoverGeneral::getSyncModeString()
{
    switch (syncMode)
    {
    case TCP:
        return "TCP";
        break;
    case SERIAL:
        return "SERIAL";
        break;
    }
    return "";
}

QString CoverGeneral::getSyncProcessString()
{
    switch (syncProc)
    {
    case APP:
        return "APP";
        break;
    case DRAW:
        return "DRAW";
        break;
    }
    return "";
}

QString CoverGeneral::getSerialDevice()
{
    return serialDevice;
}

QString CoverGeneral::getCommando0()
{
    return commando0;
}

void CoverGeneral::setStereoMode(QString s)
{
    stereoMode = s;
}

void CoverGeneral::setViewerPosX(int i)
{
    viewerPos[0] = i;
}

void CoverGeneral::setViewerPosY(int i)
{
    viewerPos[1] = i;
}

void CoverGeneral::setViewerPosZ(int i)
{
    viewerPos[2] = i;
}

void CoverGeneral::setFloorHeight(int i)
{
    floorHeight = i;
}

void CoverGeneral::setStepSize(int i)
{
    stepSize = i;
}

void CoverGeneral::setMenuPosX(int i)
{
    menuPos[0] = i;
}

void CoverGeneral::setMenuPosY(int i)
{
    menuPos[1] = i;
}

void CoverGeneral::setMenuPosZ(int i)
{
    menuPos[2] = i;
}

void CoverGeneral::setMenuOrient_h(double d)
{
    menuOrient[0] = d;
}

void CoverGeneral::setMenuOrient_p(double d)
{
    menuOrient[1] = d;
}

void CoverGeneral::setMenuOrient_r(double d)
{
    menuOrient[2] = d;
}

void CoverGeneral::setSceneSize(int i)
{
    sceneSize = i;
}

void CoverGeneral::setMenuSize(double d)
{
    menuSize = d;
}

void CoverGeneral::setSyncModeString(QString s)
{
    if (s == "TCP")
        setSyncMode(TCP);
    else if (s == "SERIAL")
        setSyncMode(SERIAL);
}

void CoverGeneral::setSyncProcessString(QString s)
{
    if (s == "APP")
        setSyncProcess(APP);
    else if (s == "DRAW")
        setSyncProcess(DRAW);
}

void CoverGeneral::setSyncMode(SyncModeType s)
{
    syncMode = s;
}

void CoverGeneral::setSyncProcess(SyncProcessType s)
{
    syncProc = s;
}

void CoverGeneral::setSerialDevice(QString s)
{
    serialDevice = s;
}

void CoverGeneral::setCommando0(QString s)
{
    commando0 = s;
}
