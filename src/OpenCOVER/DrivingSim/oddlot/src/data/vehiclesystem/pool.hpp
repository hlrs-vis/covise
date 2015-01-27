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

#ifndef POOL_HPP
#define POOL_HPP

#include "src/data/dataelement.hpp"

class CarPool;
class PoolVehicle;

class Pool : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum PoolChange
    {
        CVR_VehiclesChanged = 0x1,
        CVR_IdChanged = 0x2,
        CVR_NameChanged = 0x4,
        CVR_DataChanged = 0x8,
        CVR_CarPoolChanged = 0x10
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Pool(
        const QString &name,
        const QString &id,
        double velocity,
        double velocityDeviance,
        double numerator);
    virtual ~Pool();

    // Pool //
    //
    QString getName() const
    {
        return name_;
    }
    QString getID() const
    {
        return id_;
    }

    QList<PoolVehicle *> getVehicles()
    {
        return vehicles_;
    };

    void addVehicle(PoolVehicle *poolVehicle);

    double getVelocity() const
    {
        return velocity_;
    }
    double getVelocityDeviance() const
    {
        return velocityDeviance_;
    }
    double getNumerator() const
    {
        return numerator_;
    }

    // Parent //
    //
    CarPool *getParentCarPool() const
    {
        return parentCarPool_;
    }
    void setParentCarPool(CarPool *carPool);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getPoolChanges() const
    {
        return poolChanges_;
    }
    void addPoolChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    Pool(); /* not allowed */
    Pool(const Pool &); /* not allowed */
    Pool &operator=(const Pool &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Change flags //
    //
    int poolChanges_;

    // Parent //
    //
    CarPool *parentCarPool_;

    // Pool //
    //
    QString name_;
    QString id_;
    double velocity_;
    double velocityDeviance_;
    double numerator_;

    QList<PoolVehicle *> vehicles_;
};

#endif // POOL_HPP
