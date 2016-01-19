/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TRAFFIC_SOURCE_H
#define OSC_TRAFFIC_SOURCE_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscPosition.h>


namespace OpenScenario {

class OPENSCENARIOEXPORT distanceType: public oscEnumType
{
public:
    static distanceType *instance(); 
private:
    distanceType();
    static distanceType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTrafficSource: public oscObjectBase
{
public:
    oscTrafficSource()
    {
        OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
        OSC_ADD_MEMBER(distance);
        OSC_ADD_MEMBER(rate);
        OSC_ADD_MEMBER(velocity);
        OSC_ADD_MEMBER(velocityDeviation);
        OSC_ADD_MEMBER(forward);
        
        distance.enumType = distanceType::instance();
    };

    oscPositionMember position;
    oscEnum distance;
    oscDouble rate;
    oscDouble velocity;
    oscDouble velocityDeviation;
    oscBool forward;

    enum distance
    {
        vehicle,
        pedestrian,
    };
};

typedef oscObjectVariable<oscTrafficSource *> oscTrafficSourceMember;

}

#endif //OSC_TRAFFIC_SOURCE_H
