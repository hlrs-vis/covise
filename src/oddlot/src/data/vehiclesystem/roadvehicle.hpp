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

#ifndef ROADVEHICLE_HPP
#define ROADVEHICLE_HPP

#include "src/data/dataelement.hpp"

class VehicleGroup;

class RoadVehicle : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum RoadVehicleChange
    {
        CVR_VehicleGroupChanged = 0x1,
        CVR_IdChanged = 0x2,
        CVR_NameChanged = 0x4,
        CVR_TypeChanged = 0x8,
        CVR_IntelligenceChanged = 0x10,
        CVR_GeometryChanged = 0x20,
        CVR_InitialStateChanged = 0x40,
        CVR_RouteChanged = 0x80,
        CVR_DynamicsChanged = 0x100,
        CVR_BehaviourChanged = 0x200
    };

    // RoadVehicle Type //
    //
    enum RoadVehicleType
    {
        DRV_CAR,
        DRV_NONE
    };
    static RoadVehicle::RoadVehicleType parseRoadVehicleType(const QString &type);
    static const QString parseRoadVehicleTypeBack(RoadVehicle::RoadVehicleType type);

    // RoadVehicle Type //
    //
    enum RoadVehicleIntelligenceType
    {
        DRVI_HUMAN,
        DRVI_AGENT,
        DRVI_NONE
    };
    static RoadVehicle::RoadVehicleIntelligenceType parseRoadVehicleIntelligenceType(const QString &type);
    static const QString parseRoadVehicleIntelligenceTypeBack(RoadVehicle::RoadVehicleIntelligenceType type);

    static double defaultMaxAcceleration;
    static double defaultIndicatoryVelocity;
    static double defaultMaxCrossAcceleration;

    static double defaultMinimumGap;
    static double defaultPursueTime;
    static double defaultComfortableDecelaration;
    static double defaultSaveDeceleration;
    static double defaultApproachFactor;
    static double defaultLaneChangeThreshold; // Note: typo in specification! Threshold/Treshold
    static double defaultPolitenessFactor;

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadVehicle(
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
        double politenessFactor);
    virtual ~RoadVehicle();

    // RoadVehicle //
    //
    QString getName() const
    {
        return name_;
    }
    QString getId() const
    {
        return id_;
    }
    RoadVehicle::RoadVehicleType getType() const
    {
        return type_;
    }

    // Parameters //
    //
    RoadVehicle::RoadVehicleIntelligenceType getIntelligenceType() const
    {
        return intelligenceType_;
    }
    QString getModelFile() const
    {
        return modelFile_;
    }

    // TODO: INITIAL STATE

    double getMaxAcceleration() const
    {
        return maxAcceleration_;
    }
    double getIndicatoryVelocity() const
    {
        return indicatoryVelocity_;
    }
    double getMaxCrossAcceleration() const
    {
        return maxCrossAcceleration_;
    }

    double getMinimumGap() const
    {
        return minimumGap_;
    }
    double getPursueTime() const
    {
        return pursueTime_;
    }
    double getComfortableDeceleration() const
    {
        return comfortableDecelaration_;
    }
    double getSaveDeceleration() const
    {
        return saveDeceleration_;
    }
    double getApproachFactor() const
    {
        return approachFactor_;
    }
    double getLaneChangeThreshold() const
    {
        return laneChangeThreshold_;
    } // Note: typo in specification! Threshold/Treshold
    double getPolitenessFactor() const
    {
        return politenessFactor_;
    }

    // Parent //
    //
    VehicleGroup *getParentVehicleGroup() const
    {
        return parentVehicleGroup_;
    }
    void setParentVehicleGroup(VehicleGroup *vehicleGroup);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getRoadVehicleChanges() const
    {
        return roadVehicleChanges_;
    }
    void addRoadVehicleChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    RoadVehicle(); /* not allowed */
    RoadVehicle(const RoadVehicle &); /* not allowed */
    RoadVehicle &operator=(const RoadVehicle &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Change flags //
    //
    int roadVehicleChanges_;

    // Parent //
    //
    VehicleGroup *parentVehicleGroup_;

    // RoadVehicle //
    //
    QString name_;
    QString id_;
    RoadVehicle::RoadVehicleType type_;

    // Parameters //
    //
    RoadVehicle::RoadVehicleIntelligenceType intelligenceType_;
    QString modelFile_;

    // TODO: INITIAL STATE

    double maxAcceleration_;
    double indicatoryVelocity_;
    double maxCrossAcceleration_;

    double minimumGap_;
    double pursueTime_;
    double comfortableDecelaration_;
    double saveDeceleration_;
    double approachFactor_;
    double laneChangeThreshold_; // Note: typo in specification! Threshold/Treshold
    double politenessFactor_;
};

#endif // ROADVEHICLE_HPP
