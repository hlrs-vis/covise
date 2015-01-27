/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** covergeneral.h
 ** 2004-02-04, Matthias Feurer
 ****************************************************************************/

#ifndef COVERGENERAL_H
#define COVERGENERAL_H

#include "projectionarea.h"
#include "host.h"

#include <qstring.h>

enum SyncModeType
{
    TCP,
    SERIAL
};
enum SyncProcessType
{
    APP,
    DRAW
};

class CoverGeneral
{

public:
    CoverGeneral();
    QString getStereoMode();
    int getViewerPosX();
    int getViewerPosY();
    int getViewerPosZ();
    int getFloorHeight();
    int getStepSize();
    int getMenuPosX();
    int getMenuPosY();
    int getMenuPosZ();
    double getMenuOrient_h();
    double getMenuOrient_p();
    double getMenuOrient_r();
    double getMenuSize();
    int getSceneSize();
    SyncModeType getSyncMode();
    SyncProcessType getSyncProcess();
    QString getSyncModeString();
    QString getSyncProcessString();
    QString getSerialDevice();
    QString getCommando0();

    void setStereoMode(QString s);
    void setViewerPosX(int i);
    void setViewerPosY(int i);
    void setViewerPosZ(int i);
    void setFloorHeight(int i);
    void setStepSize(int i);
    void setMenuPosX(int i);
    void setMenuPosY(int i);
    void setMenuPosZ(int i);
    void setMenuOrient_h(double d);
    void setMenuOrient_p(double d);
    void setMenuOrient_r(double d);
    void setMenuSize(double d);
    void setSceneSize(int i);
    void setSyncModeString(QString s);
    void setSyncProcessString(QString s);
    void setSyncMode(SyncModeType s);
    void setSyncProcess(SyncProcessType s);
    void setSerialDevice(QString s);
    void setCommando0(QString s);

private:
    // Scene:
    QString stereoMode; // "active" or "passive"
    int viewerPos[3];
    int floorHeight;
    int stepSize;
    int menuPos[3];
    double menuOrient[3];
    double menuSize;
    int sceneSize;

    // MultiPC:
    SyncModeType syncMode;
    SyncProcessType syncProc;
    QString serialDevice;
    QString commando0;
};
#endif // COVERGENERAL
