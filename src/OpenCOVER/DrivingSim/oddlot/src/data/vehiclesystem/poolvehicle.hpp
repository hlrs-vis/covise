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

#ifndef POOLVEHICLE_HPP
#define POOLVEHICLE_HPP

#include "src/data/dataelement.hpp"

class PoolVehicle : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum PoolVehicleChange
    {
        CVR_IdChanged = 0x2,
        CVR_NumeratorChanged = 0x4,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PoolVehicle(
        const QString &id,
        double numerator);
    virtual ~PoolVehicle();

    // PoolVehicle //
    //
    QString getID() const
    {
        return id_;
    }
    double getNumerator() const
    {
        return numerator_;
    }

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getPoolVehicleChanges() const
    {
        return roadVehicleChanges_;
    }
    void addPoolVehicleChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    PoolVehicle(); /* not allowed */
    PoolVehicle(const PoolVehicle &); /* not allowed */
    PoolVehicle &operator=(const PoolVehicle &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Change flags //
    //
    int roadVehicleChanges_;

    // PoolVehicle //
    //
    QString id_;
    double numerator_;
};

#endif // POOLVEHICLE_HPP
