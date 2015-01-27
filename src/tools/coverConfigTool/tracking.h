/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** tracking.h
 ** 2004-04-02, Matthias Feurer
 ****************************************************************************/

#ifndef TRACKING_H
#define TRACKING_H

#include <qstring.h>

enum DirectionType
{
    Forward,
    Backward,
    Left,
    Right,
    Top,
    Bottom
};
enum TrackerType
{
    POLHEMUS_LONGRANGE,
    POLHEMUS2,
    MOTIONSTAR2,
    FLOCK_OF_BIRDS,
    OTHER
};
enum DebugTrackingType
{
    DEBUG_RAW,
    DEBUG_APP,
    DEBUG_OFF
};

class Tracking
{

public:
    Tracking();
    int getNoSensors();
    int getAdrHeadSensor();
    int getAdrHandSensor();
    double getTransmitterOffsetX();
    double getTransmitterOffsetY();
    double getTransmitterOffsetZ();
    double getTransmitterOrientH();
    double getTransmitterOrientP();
    double getTransmitterOrientR();

    double getHeadSensorOffsetX();
    double getHeadSensorOffsetY();
    double getHeadSensorOffsetZ();
    double getHeadSensorOrientH();
    double getHeadSensorOrientP();
    double getHeadSensorOrientR();

    double getHandSensorOffsetX();
    double getHandSensorOffsetY();
    double getHandSensorOffsetZ();
    double getHandSensorOrientH();
    double getHandSensorOrientP();
    double getHandSensorOrientR();
    DirectionType getXDir();
    DirectionType getYDir();
    DirectionType getZDir();
    TrackerType getTrackerType();
    QString getTrackerTypeString(TrackerType t);

    DirectionType getDirectionType(QString dirString);
    QString getDirectionTypeString(DirectionType d);

    double getLinearMagneticFieldCorrectionX();
    double getLinearMagneticFieldCorrectionY();
    double getLinearMagneticFieldCorrectionZ();
    QString getInterpolationFile();
    DebugTrackingType getDebugTracking();
    QString getDebugTrackingString();
    bool getDebugButtons();
    QString getDebugButtonsString();
    int getDebugStation();

    void setNoSensors(int n);
    void setAdrHeadSensor(int a);
    void setAdrHandSensor(int a);
    void setTransmitterOffsetX(double x);
    void setTransmitterOffsetY(double y);
    void setTransmitterOffsetZ(double z);
    void setTransmitterOrientH(double h);
    void setTransmitterOrientP(double p);
    void setTransmitterOrientR(double r);

    void setHeadSensorOffsetX(double x);
    void setHeadSensorOffsetY(double y);
    void setHeadSensorOffsetZ(double z);
    void setHeadSensorOrientH(double h);
    void setHeadSensorOrientP(double p);
    void setHeadSensorOrientR(double r);

    void setHandSensorOffsetX(double x);
    void setHandSensorOffsetY(double y);
    void setHandSensorOffsetZ(double z);
    void setHandSensorOrientH(double h);
    void setHandSensorOrientP(double p);
    void setHandSensorOrientR(double r);

    void setDirections(DirectionType xDir,
                       DirectionType yDir,
                       DirectionType zDir);

    void setTrackerType(TrackerType t);
    void setLinearMagneticFieldCorrection(double x,
                                          double y,
                                          double z);
    void setInterpolationFile(QString fileName);
    void setDebugTracking(DebugTrackingType dt);
    void setDebugButtons(int i);
    void setDebugStation(int i);

    bool checkDirections(DirectionType xDir, DirectionType yDir);
    DirectionType getZDirection(DirectionType xDir, DirectionType yDir);
    void computeOrientation(DirectionType xDir,
                            DirectionType yDir,
                            DirectionType zDir,
                            double *h,
                            double *p,
                            double *r);

private:
    int noConnectedSensors; // "active" or "passive"
    int adrHeadSensor;
    int adrHandSensor;
    double transmitterOffset[3];
    double transmitterOrient[3];
    double headSensorOffset[3];
    double headSensorOrient[3];
    double handSensorOffset[3];
    double handSensorOrient[3];
    DirectionType xDir;
    DirectionType yDir;
    DirectionType zDir;
    TrackerType tracker;

    double fieldCorrection[3];
    QString interpolationFile;
    DebugTrackingType debugTracking;
    bool debugButtons;
    int debugStation;
};
#endif // TRACKING
