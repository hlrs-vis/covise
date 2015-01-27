/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RoadSensor_h
#define RoadSensor_h

#include "Element.h"

class RoadSensorTriggerAction
{
public:
    virtual void operator()(const std::string &) = 0;
    /*{
      std::cout << "RoadSensorTriggerAction::operator(): null operator..." << std::endl;
   }*/
};

class VEHICLEUTILEXPORT RoadSensor : public Element
{
public:
    RoadSensor(const std::string &, const double &);

    const double &getS() const
    {
        return s;
    }

    void setTriggerAction(RoadSensorTriggerAction *);

    void trigger(const std::string &info)
    {
        if (triggerAction)
        {
            std::cout << "RoadSensor::trigger(): Sensor " << getId() << ": TriggeringAction! Info: " << info << std::endl;
            (*triggerAction)(info);
        }
    }

protected:
    double s;

    RoadSensorTriggerAction *triggerAction;
};

#endif
