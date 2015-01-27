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

#ifndef CARPOOL_HPP
#define CARPOOL_HPP

#include "src/data/dataelement.hpp"

class VehicleSystem;
class Pool;

class CarPool : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum CarPoolChange
    {
        CVR_VehicleSystemChanged = 0x1,
        CVR_PoolChanged = 0x2
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CarPool();
    virtual ~CarPool();

    void addPool(Pool *pool);

    // Parent //
    //
    VehicleSystem *getParentVehicleSystem() const
    {
        return parentVehicleSystem_;
    }
    void setParentVehicleSystem(VehicleSystem *vehicleSystem);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getCarPoolChanges() const
    {
        return carPoolChanges_;
    }
    void addCarPoolChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);

protected:
private:
    CarPool(const CarPool &); /* not allowed */
    CarPool &operator=(const CarPool &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Change flags //
    //
    int carPoolChanges_;

    // Parent //
    //
    VehicleSystem *parentVehicleSystem_;

    // Pools //
    //
    QMap<QString, Pool *> pools_; // owned
};

#endif // CARPOOL_HPP
