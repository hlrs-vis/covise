/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_PRECIPITATION_H
#define OSC_PRECIPITATION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OPENSCENARIOEXPORT precipitationType: public oscEnumType
{
public:
    static precipitationType *instance(); 
private:
    precipitationType();
    static precipitationType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPrecipitation: public oscObjectBase
{
public:
	oscDouble intensity;
    enum type
    {
        dry,
        rain,
        snow,
    };
	
    oscPrecipitation()
    {
		OSC_ADD_MEMBER(intensity);
		OSC_ADD_MEMBER(type);
		type.enumType = precipitationType::instance();
    };
	oscEnum type;
};

typedef oscObjectVariable<oscPrecipitation *> oscPrecipitationMember;

}

#endif //OSC_PRECIPITATION_H
