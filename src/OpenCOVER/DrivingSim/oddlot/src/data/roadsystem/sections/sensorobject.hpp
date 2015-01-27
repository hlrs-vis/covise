/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#ifndef SENSOROBJECT_HPP
#define SENSOROBJECT_HPP

#include "roadsection.hpp"

class Sensor : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum SensorChange
    {
        CEL_ParameterChange = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Sensor(const QString &id, double s);
    virtual ~Sensor()
    { /* does nothing */
    }

    // Sensor //
    //
    QString getId() const
    {
        return id_;
    }
    void setId(const QString &id)
    {
        id_ = id;
    }

    double getS() const
    {
        return s_;
    }
    void setS(const double s)
    {
        s_ = s;
    }

    double getSEnd() const
    {
        return s_;
    };
    double getLength() const
    {
        return 0;
    };

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getSensorChanges() const
    {
        return sensorChanges_;
    }
    void addSensorChanges(int changes);

    // Prototype Pattern //
    //
    Sensor *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    Sensor(); /* not allowed */
    Sensor(const Sensor &); /* not allowed */
    Sensor &operator=(const Sensor &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Parent //
    //
    RSystemElementRoad *parentRoad_;

    // Sensor //
    //
    // Mandatory
    QString id_;
    double s_;

    // Change flags //
    //
    int sensorChanges_;
};

#endif // SENSOROBJECT_HPP
