/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   01.06.2010
**
**************************************************************************/

#ifndef VEHICLESYSTEM_HPP
#define VEHICLESYSTEM_HPP

#include "src/data/dataelement.hpp"

#include <QStringList>

class VehicleGroup;
class CarPool;

class VehicleSystem : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum VehicleSystemChange
    {
        CVS_ProjectDataChanged = 0x1,
        CVS_VehicleGroupsChanged = 0x2,
        CVS_CarPoolChanged = 0x4
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit VehicleSystem();
    virtual ~VehicleSystem();

    // VehicleGroups //
    //
    void addVehicleGroup(VehicleGroup *vehicleGroup);
    QList<VehicleGroup *> getVehicleGroups() const
    {
        return vehicleGroups_;
    }

    void setCarPool(CarPool *carPool);
    CarPool *getCarPool() const
    {
        return carPool_;
    };

    // IDs //
    //
    const QString getUniqueId(const QString &suggestion);

    // ProjectData //
    //
    ProjectData *getParentProjectData() const
    {
        return parentProjectData_;
    }
    void setParentProjectData(ProjectData *projectData);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getVehicleSystemChanges() const
    {
        return vehicleSystemChanges_;
    }
    void addVehicleSystemChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);
    virtual void acceptForVehicleGroups(Visitor *visitor);

private:
    //	VehicleSystem(); /* not allowed */
    VehicleSystem(const VehicleSystem &); /* not allowed */
    VehicleSystem &operator=(const VehicleSystem &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int vehicleSystemChanges_;

    // ProjectData //
    //
    ProjectData *parentProjectData_;

    // VehicleGroups //
    //
    QList<VehicleGroup *> vehicleGroups_; // owned

    // CarPool
    CarPool *carPool_;

    // IDs //
    //
    QStringList ids_;
    int idCount_;
};

#endif // VEHICLESYSTEM_HPP
