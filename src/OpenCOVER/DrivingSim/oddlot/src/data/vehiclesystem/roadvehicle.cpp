/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.06.2010
**
**************************************************************************/

#include "roadvehicle.hpp"

#include "vehiclegroup.hpp"

RoadVehicle::RoadVehicle(
    const QString &name,
    const QString &id,
    RoadVehicle::RoadVehicleType type,
    RoadVehicle::RoadVehicleIntelligenceType intelligenceType,
    const QString &modelFile,
    double maxAcceleration,
    double indicatoryVelocity,
    double maxCrossAcceleration,
    double minimumGap,
    double pursueTime,
    double comfortableDecelaration,
    double saveDeceleration,
    double approachFactor,
    double laneChangeThreshold,
    double politenessFactor)
    : DataElement()
    , roadVehicleChanges_(0x0)
    , parentVehicleGroup_(NULL)
    , name_(name)
    , id_(id)
    , type_(type)
    , intelligenceType_(intelligenceType)
    , modelFile_(modelFile)
    , maxAcceleration_(maxAcceleration)
    , indicatoryVelocity_(indicatoryVelocity)
    , maxCrossAcceleration_(maxCrossAcceleration)
    , minimumGap_(minimumGap)
    , pursueTime_(pursueTime)
    , comfortableDecelaration_(comfortableDecelaration)
    , saveDeceleration_(saveDeceleration)
    , approachFactor_(approachFactor)
    , laneChangeThreshold_(laneChangeThreshold)
    , politenessFactor_(politenessFactor)
{
}

RoadVehicle::~RoadVehicle()
{
}

//##################//
// ProjectData      //
//##################//

void
RoadVehicle::setParentVehicleGroup(VehicleGroup *vehicleGroup)
{
    parentVehicleGroup_ = vehicleGroup;
    setParentElement(vehicleGroup);
    addRoadVehicleChanges(RoadVehicle::CVR_VehicleGroupChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RoadVehicle::notificationDone()
{
    roadVehicleChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
RoadVehicle::addRoadVehicleChanges(int changes)
{
    if (changes)
    {
        roadVehicleChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
* With autotraverse: visitor will be send to roads, fiddleyards, etc.
* Without: accepts visitor as 'this'.
*/
void
RoadVehicle::accept(Visitor *visitor)
{
    visitor->visit(this);
}

//###################//
// Static Functions  //
//###################//

RoadVehicle::RoadVehicleType
RoadVehicle::parseRoadVehicleType(const QString &type)
{
    if (type == "car")
        return RoadVehicle::DRV_CAR;
    //	else if(type == "xx")
    //		return RoadVehicle::DRV_XX;
    else
    {
        qDebug("WARNING: unknown road vehicle type: %s", type.toUtf8().constData());
        return RoadVehicle::DRV_NONE;
    }
}

const QString
RoadVehicle::parseRoadVehicleTypeBack(RoadVehicle::RoadVehicleType type)
{
    if (type == RoadVehicle::DRV_CAR)
        return QString("car");
    //	else if(type == RoadVehicle::DRV_XX)
    //		return QString("xx");
    else
    {
        qDebug("WARNING: unknown road vehicle type.");
        return QString("none");
    }
}

RoadVehicle::RoadVehicleIntelligenceType
RoadVehicle::parseRoadVehicleIntelligenceType(const QString &type)
{
    if (type == "human")
        return RoadVehicle::DRVI_HUMAN;
    else if (type == "agent")
        return RoadVehicle::DRVI_AGENT;
    else
    {
        qDebug("WARNING: unknown road vehicle type: %s", type.toUtf8().constData());
        return RoadVehicle::DRVI_NONE;
    }
}

const QString
RoadVehicle::parseRoadVehicleIntelligenceTypeBack(RoadVehicle::RoadVehicleIntelligenceType type)
{
    if (type == RoadVehicle::DRVI_HUMAN)
        return QString("human");
    else if (type == RoadVehicle::DRVI_AGENT)
        return QString("agent");
    else
    {
        qDebug("WARNING: unknown road vehicle type.");
        return QString("none");
    }
}

double RoadVehicle::defaultMaxAcceleration = 3.0;
double RoadVehicle::defaultIndicatoryVelocity = 30.0;
double RoadVehicle::defaultMaxCrossAcceleration = 6.0;

double RoadVehicle::defaultMinimumGap = 2.0;
double RoadVehicle::defaultPursueTime = 1.4;
double RoadVehicle::defaultComfortableDecelaration = 1.5;
double RoadVehicle::defaultSaveDeceleration = 15.0;
double RoadVehicle::defaultApproachFactor = 4.0;
double RoadVehicle::defaultLaneChangeThreshold = 0.8;
double RoadVehicle::defaultPolitenessFactor = 0.3;
