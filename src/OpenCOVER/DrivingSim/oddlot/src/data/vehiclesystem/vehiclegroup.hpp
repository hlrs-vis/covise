/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   07.06.2010
**
**************************************************************************/

#ifndef VEHICLEGROUP_HPP
#define VEHICLEGROUP_HPP

#include "src/data/dataelement.hpp"

class VehicleSystem;

class VehicleGroup : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum VehicleGroupChange
    {
        CVG_VehicleSystemChanged = 0x1,
        CVG_RoadVehicleChanged = 0x2,
    };

    // Default Values //
    //
    static double defaultRangeLOD;

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit VehicleGroup(double rangeLOD);
    virtual ~VehicleGroup();

    // RoadVehicles //
    //
    QMap<QString, RoadVehicle *> getRoadVehicles() const
    {
        return roadVehicles_;
    }

    // VehicleGroup //
    //
    double getRangeLOD() const
    {
        return rangeLOD_;
    }
    void addRoadVehicle(RoadVehicle *roadVehicle);

    void setPassThreshold(double t)
    {
        passThreshold_ = t;
        passThresholdSet_ = true;
    }
    double getPassThreshold() const
    {
        return passThreshold_;
    }
    bool hasPassThreshold() const
    {
        return passThresholdSet_;
    }

    // VehicleSystem //
    //
    VehicleSystem *getParentVehicleSystem() const
    {
        return parentVehicleSystem_;
    }
    void setParentVehicleSystem(VehicleSystem *vehicleSystem);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getVehicleGroupChanges() const
    {
        return vehicleGroupChanges_;
    }
    void addVehicleGroupChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);
    virtual void acceptForRoadVehicles(Visitor *visitor);

private:
    VehicleGroup(); /* not allowed */
    VehicleGroup(const VehicleGroup &); /* not allowed */
    VehicleGroup &operator=(const VehicleGroup &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // VehicleSystem //
    //
    VehicleSystem *parentVehicleSystem_;

    // Change flags //
    //
    int vehicleGroupChanges_;

    // vehicles: //
    //
    double rangeLOD_;
    double passThreshold_;
    bool passThresholdSet_;

    // RoadVehicles //
    //
    QMap<QString, RoadVehicle *> roadVehicles_; // owned
};

#endif // VEHICLEGROUP_HPP
