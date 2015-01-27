/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** tracking.cpp
 ** 2004-04-02, Matthias Feurer
 ****************************************************************************/

#include "tracking.h"

#include <qstring.h>
#include "covise.h"

Tracking::Tracking()
{
    noConnectedSensors = 0;
    adrHeadSensor = 0;
    adrHandSensor = 0;
    transmitterOffset[0] = 0;
    transmitterOffset[1] = 0;
    transmitterOffset[2] = 0;
    transmitterOrient[0] = 0;
    transmitterOrient[1] = 0;
    transmitterOrient[2] = 0;

    headSensorOffset[0] = 0;
    headSensorOffset[1] = 0;
    headSensorOffset[2] = 0;
    headSensorOrient[0] = 0;
    headSensorOrient[1] = 0;
    headSensorOrient[2] = 0;

    handSensorOffset[0] = 0;
    handSensorOffset[1] = 0;
    handSensorOffset[2] = 0;
    handSensorOrient[0] = 0;
    handSensorOrient[1] = 0;
    handSensorOrient[2] = 0;

    xDir = Forward;
    yDir = Forward;
    zDir = Forward;

    tracker = POLHEMUS_LONGRANGE;

    fieldCorrection[0] = 0;
    fieldCorrection[1] = 0;
    fieldCorrection[2] = 0;

    interpolationFile = "";
    debugTracking = DEBUG_RAW;
    debugButtons = false;
    debugStation = 0;
}

int Tracking::getNoSensors()
{
    return noConnectedSensors;
}

int Tracking::getAdrHeadSensor()
{
    return adrHeadSensor;
}

int Tracking::getAdrHandSensor()
{
    return adrHandSensor;
}

double Tracking::getTransmitterOffsetX()
{
    return transmitterOffset[0];
}

double Tracking::getTransmitterOffsetY()
{
    return transmitterOffset[1];
}

double Tracking::getTransmitterOffsetZ()
{
    return transmitterOffset[2];
}

double Tracking::getTransmitterOrientH()
{
    return transmitterOrient[0];
}

double Tracking::getTransmitterOrientP()
{
    return transmitterOrient[1];
}

double Tracking::getTransmitterOrientR()
{
    return transmitterOrient[2];
}

double Tracking::getHeadSensorOffsetX()
{
    return headSensorOffset[0];
}

double Tracking::getHeadSensorOffsetY()
{
    return headSensorOffset[1];
}

double Tracking::getHeadSensorOffsetZ()
{
    return headSensorOffset[2];
}

double Tracking::getHeadSensorOrientH()
{
    return headSensorOrient[0];
}

double Tracking::getHeadSensorOrientP()
{
    return headSensorOrient[1];
}

double Tracking::getHeadSensorOrientR()
{
    return headSensorOrient[2];
}

double Tracking::getHandSensorOffsetX()
{
    return handSensorOffset[0];
}

double Tracking::getHandSensorOffsetY()
{
    return handSensorOffset[1];
}

double Tracking::getHandSensorOffsetZ()
{
    return handSensorOffset[2];
}

double Tracking::getHandSensorOrientH()
{
    return handSensorOrient[0];
}

double Tracking::getHandSensorOrientP()
{
    return handSensorOrient[1];
}

double Tracking::getHandSensorOrientR()
{
    return handSensorOrient[2];
}

DirectionType Tracking::getDirectionType(QString dirString)
{
    if (dirString == "Forward")
        return Forward;
    else if (dirString == "Backward")
        return Backward;
    else if (dirString == "Left")
        return Left;
    else if (dirString == "Right")
        return Right;
    else if (dirString == "Top")
        return Top;
    else if (dirString == "Bottom")
        return Bottom;
    else
        cout << "Direction string not recognized! error!" << endl;
    return Forward;
}

QString Tracking::getDirectionTypeString(DirectionType d)
{
    switch (d)
    {
    case Forward:
        return "Forward";
        break;
    case Backward:
        return "Backward";
        break;
    case Left:
        return "Left";
        break;
    case Right:
        return "Right";
        break;
    case Top:
        return "Top";
        break;
    case Bottom:
        return "Bottom";
        break;
    }
}

DirectionType Tracking::getXDir()
{
    return xDir;
}

DirectionType Tracking::getYDir()
{
    return yDir;
}

DirectionType Tracking::getZDir()
{
    return zDir;
}

TrackerType Tracking::getTrackerType()
{
    return tracker;
}

QString Tracking::getTrackerTypeString(TrackerType t)
{
    switch (t)
    {
    case POLHEMUS_LONGRANGE:
        return "POLHEMUS LONGRANGE";
        break;
    case POLHEMUS2:
        return "POLHEMUS";
        break;
    case MOTIONSTAR2:
        return "MOTIONSTAR2";
        break;
    case FLOCK_OF_BIRDS:
        return "FLOCK OF BIRDS";
        break;
    default:
        return "OTHER";
        break;
    }
}

double Tracking::getLinearMagneticFieldCorrectionX()
{
    return fieldCorrection[0];
}

double Tracking::getLinearMagneticFieldCorrectionY()
{
    return fieldCorrection[1];
}

double Tracking::getLinearMagneticFieldCorrectionZ()
{
    return fieldCorrection[2];
}

QString Tracking::getInterpolationFile()
{
    return interpolationFile;
}

DebugTrackingType Tracking::getDebugTracking()
{
    return debugTracking;
}

QString Tracking::getDebugTrackingString()
{
    switch (debugTracking)
    {
    case DEBUG_RAW:
        return "RAW";
        break;
    case DEBUG_APP:
        return "APP";
        break;
    default:
        return "OFF";
        break;
    }
}

bool Tracking::getDebugButtons()
{
    return debugButtons;
}

QString Tracking::getDebugButtonsString()
{
    if (debugButtons == true)
        return "TRUE";
    else
        return "FALSE";
}

int Tracking::getDebugStation()
{
    return debugStation;
}

void Tracking::setNoSensors(int n)
{
    noConnectedSensors = n;
}

void Tracking::setAdrHeadSensor(int a)
{
    adrHeadSensor = a;
}

void Tracking::setAdrHandSensor(int a)
{
    adrHandSensor = a;
}

void Tracking::setTransmitterOffsetX(double x)
{
    transmitterOffset[0] = x;
}

void Tracking::setTransmitterOffsetY(double y)
{
    transmitterOffset[1] = y;
}

void Tracking::setTransmitterOffsetZ(double z)
{
    transmitterOffset[2] = z;
}

void Tracking::setTransmitterOrientH(double h)
{
    transmitterOrient[0] = h;
}

void Tracking::setTransmitterOrientP(double p)
{
    transmitterOrient[1] = p;
}

void Tracking::setTransmitterOrientR(double r)
{
    transmitterOrient[2] = r;
}

void Tracking::setHeadSensorOffsetX(double x)
{
    headSensorOffset[0] = x;
}

void Tracking::setHeadSensorOffsetY(double y)
{
    headSensorOffset[1] = y;
}

void Tracking::setHeadSensorOffsetZ(double z)
{
    headSensorOffset[2] = z;
}

void Tracking::setHeadSensorOrientH(double h)
{
    headSensorOrient[0] = h;
}

void Tracking::setHeadSensorOrientP(double p)
{
    headSensorOrient[1] = p;
}

void Tracking::setHeadSensorOrientR(double r)
{
    headSensorOrient[2] = r;
}

void Tracking::setHandSensorOffsetX(double x)
{
    handSensorOffset[0] = x;
}

void Tracking::setHandSensorOffsetY(double y)
{
    handSensorOffset[1] = y;
}

void Tracking::setHandSensorOffsetZ(double z)
{
    handSensorOffset[2] = z;
}

void Tracking::setHandSensorOrientH(double h)
{
    handSensorOrient[0] = h;
}

void Tracking::setHandSensorOrientP(double p)
{
    handSensorOrient[1] = p;
}

void Tracking::setHandSensorOrientR(double r)
{
    handSensorOrient[2] = r;
}

void Tracking::setDirections(DirectionType x,
                             DirectionType y,
                             DirectionType z)
{
    xDir = x;
    yDir = y;
    zDir = z;
}

void Tracking::setTrackerType(TrackerType t)
{
    tracker = t;
}

void Tracking::setLinearMagneticFieldCorrection(double x,
                                                double y,
                                                double z)
{
    fieldCorrection[0] = x;
    fieldCorrection[1] = y;
    fieldCorrection[2] = z;
}

void Tracking::setInterpolationFile(QString fileName)
{
    interpolationFile = fileName;
}

void Tracking::setDebugTracking(DebugTrackingType dt)
{
    debugTracking = dt;
}

void Tracking::setDebugButtons(int i)
{
    if (i == 1)
        debugButtons = true;
    else
        debugButtons = false;
}

void Tracking::setDebugStation(int i)
{
    debugStation = i;
}

bool Tracking::checkDirections(DirectionType xDir, DirectionType yDir)
{
    if (xDir == yDir)
        return false;
    if (((xDir == Left) && (yDir == Right))
        || ((xDir == Right) && (yDir == Left)))
        return false;
    if (((xDir == Top) && (yDir == Bottom))
        || ((xDir == Bottom) && (yDir == Top)))
        return false;
    if (((xDir == Forward) && (yDir == Backward))
        || ((xDir == Backward) && (yDir == Forward)))
        return false;

    return true;
}

DirectionType Tracking::getZDirection(DirectionType xDir, DirectionType yDir)
{
    switch (xDir)
    {
    case Forward:
        switch (yDir)
        {
        case Left:
            return Top;
            break;
        case Top:
            return Right;
            break;
        case Right:
            return Bottom;
            break;
        case Bottom:
            return Left;
            break;
        }
        break;
    case Backward:
        switch (yDir)
        {
        case Left:
            return Bottom;
            break;
        case Top:
            return Left;
            break;
        case Right:
            return Top;
            break;
        case Bottom:
            return Right;
            break;
        }
        break;
    case Left:
        switch (yDir)
        {
        case Top:
            return Forward;
            break;
        case Forward:
            return Top;
            break;
        case Bottom:
            return Backward;
            break;
        case Backward:
            return Bottom;
            break;
        }
        break;
    case Right:
        switch (yDir)
        {
        case Top:
            return Backward;
            break;
        case Forward:
            return Top;
            break;
        case Bottom:
            return Forward;
            break;
        case Backward:
            return Bottom;
            break;
        }
        break;
    case Top:
        switch (yDir)
        {
        case Right:
            return Forward;
            break;
        case Forward:
            return Left;
            break;
        case Left:
            return Backward;
            break;
        case Backward:
            return Right;
            break;
        }
        break;
    case Bottom:
        switch (yDir)
        {
        case Right:
            return Backward;
            break;
        case Forward:
            return Right;
            break;
        case Left:
            return Forward;
            break;
        case Backward:
            return Left;
            break;
        }
        break;
    }
    return Forward;
}

void Tracking::computeOrientation(DirectionType xDir,
                                  DirectionType yDir,
                                  DirectionType zDir,
                                  double *h,
                                  double *p,
                                  double *r)
{
    switch (xDir)
    {
    case Forward:
        switch (yDir)
        {
        case Left:
            *r = 0;
            *p = 0;
            *h = 90;
            break;
        case Top:
            *r = 90;
            *p = 90;
            *h = 0;
            break;
        case Right:
            *r = 180;
            *p = 0;
            *h = 270;
            break;
        case Bottom:
            *r = -90;
            *p = 270;
            *h = 0;
            break;
        }
        break;
    case Backward:
        switch (yDir)
        {
        case Left:
            *r = 180;
            *p = 0;
            *h = 90;
            break;
        case Top:
            *r = -90;
            *p = 90;
            *h = 0;
            break;
        case Right:
            *r = 0;
            *p = 0;
            *h = -90;
            break;
        case Bottom:
            *r = 90;
            *p = -90;
            *h = 0;
            break;
        }
        break;
    case Left:
        switch (yDir)
        {
        case Top:
            *r = 180;
            *p = 90;
            *h = 0;
            break;
        case Forward:
            *r = 180;
            *p = 0;
            *h = 0;
            break;
        case Bottom:
            *r = 180;
            *p = 270;
            *h = 0;
            break;
        case Backward:
            *r = 180;
            *p = 180;
            *h = 0;
            break;
        }
        break;
    case Right:
        switch (yDir)
        {
        case Top:
            *r = 0;
            *p = 90;
            *h = 0;
            break;
        case Forward:
            *r = 0;
            *p = 0;
            *h = 0;
            break;
        case Bottom:
            *r = 0;
            *p = 180;
            *h = 0;
            break;
        case Backward:
            *r = 0;
            *p = 270;
            *h = 0;
            break;
        }
        break;
    case Top:
        switch (yDir)
        {
        case Right:
            *r = -90;
            *p = 0;
            *h = 270;
            break;
        case Forward:
            *r = -90;
            *p = 0;
            *h = 0;
            break;
        case Left:
            *r = -90;
            *p = 0;
            *h = 90;
            break;
        case Backward:
            *r = -90;
            *p = 0;
            *h = 180;
            break;
        }
        break;
    case Bottom:
        switch (yDir)
        {
        case Right:
            *r = 90;
            *p = 0;
            *h = 270;
            break;
        case Forward:
            *r = 90;
            *p = 0;
            *h = 0;
            break;
        case Left:
            *r = 90;
            *p = 0;
            *h = 90;
            break;
        case Backward:
            *r = 90;
            *p = 90;
            *h = 0;
            break;
        }
        break;
    }
}
